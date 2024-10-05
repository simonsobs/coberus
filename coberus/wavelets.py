"""
Routines for wavelet transforms.
"""

from typing import Any
from pydantic import BaseModel, ConfigDict

from pixell import uharm
from pixell import wavelets
from pixell import wcsutils
from pixell import enmap

from pathlib import Path
from dask.distributed import Client, as_completed, get_client
import dask.array as da
import numpy as np
import math, time

from typing import Callable
from coberus.core import Coadder


class Map(BaseModel):
    """
    Information about a map that will be processed by
    the coaddition pipeline.
    """

    # A useful tag for referencing the map
    tag: str
    # Path to the map on disk
    path: Path
    # Path to the mask on disk
    mask: Path
    # Minium multipole to consider
    lmin: int | None
    # Maximum multipole to consider
    lmax: int | None
    # Response value. For CMB solutions, this is 1.0
    response: float
    # Beam function. For a given ell, return the beam value.
    beam: Callable[[float], float]
    # Optional: a function to call on the loaded enmap when
    # loading it.
    postprocess_map: Callable[[enmap.ndmap], enmap.ndmap] = lambda x: x
    # Optional: a function to call on the loaded mask when
    # loading it.
    postprocess_mask: Callable[[enmap.ndmap], enmap.ndmap] = lambda x: x

    def scales(self, basis: wavelets.CosineNeedlet) -> list[int]:
        """
        Return the scales that are relevant for this map.
        """

        scales = []

        lmin = self.lmin if self.lmin is not None else -1
        lmax = self.lmax if self.lmax is not None else 10000000000

        for i in range(basis.n):
            wlmin = basis.lmins[i]
            wlmax = basis.lmaxs[i]

            if (lmin <= wlmin) and (lmax >= wlmax):
                scales.append(i)

        return scales

    def read_map(self) -> enmap.ndmap:
        """
        Read the map from disk.
        """
        return self.postprocess_map(enmap.read_map(str(self.path)))

    def read_mask(self) -> enmap.ndmap:
        """
        Read the mask from disk.
        """
        return self.postprocess_mask(enmap.read_map(str(self.mask)))


class WaveletMetadata(BaseModel):
    """
    Wavelet transform settings.
    """

    shape: tuple[int, int]
    wcs: wcsutils.WCS
    basis: wavelets.CosineNeedlet

    # The output beam FWHM
    output_beam_fwhm: float

    # Factor to smooth for in covariance maps
    cov_smooth_factor: int = 1

    io_suffix: str | None = None
    output_root: Path

    uht: uharm.UHT | None = None
    wt: wavelets.WaveletTransform | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def nwaves(self) -> int:
        return self.basis.n

    def output_beam(self, ells: float) -> float:
        """
        Calculate the output beam for a given (array) of multipoles.
        """
        fwhm_radians = np.deg2rad(self.output_beam_fwhm / 60.0)
        square_ells = ells * ells

        prefactor = -(fwhm_radians * fwhm_radians) / (16.0 * math.log(2))

        return np.exp(prefactor * square_ells)

    def model_post_init(self, __context: Any) -> None:
        self.uht = uharm.UHT(self.shape, self.wcs)
        self.wt = wavelets.WaveletTransform(self.uht, basis=self.basis)
        return super().model_post_init(__context)

    @classmethod
    def from_primary_map(
        cls,
        filename: Path,
        basis: wavelets.CosineNeedlet,
        output_beam_fwhm: float,
        output_root: Path,
        cov_smooth_factor: int = 1,
        io_suffix: str | None = None,
    ) -> "WaveletMetadata":
        """
        Create a WaveletMetadata object from a primary map.
        """
        shape, wcs = enmap.read_map_geometry(str(filename))

        return cls(
            shape=shape,
            wcs=wcs,
            basis=basis,
            output_beam_fwhm=output_beam_fwhm,
            output_root=output_root,
            cov_smooth_factor=cov_smooth_factor,
            io_suffix=io_suffix,
        )


def map_filename(metadata: WaveletMetadata, map: Map, scale: int) -> Path:
    """
    Generate a filename for a map at a given scale.
    """
    core_name = f"{map.tag}_scale_{scale}"
    core_name += ".fits" if metadata.io_suffix is None else f"{metadata.io_suffix}.fits"

    return metadata.output_root / f"wavelet_map_{core_name}"


def mask_filename(metadata: WaveletMetadata, map: Map, scale: int) -> Path:
    """
    Generate a filename for a mask at a given scale.
    """
    core_name = f"{map.tag}_scale_{scale}"
    core_name += ".fits" if metadata.io_suffix is None else f"{metadata.io_suffix}.fits"

    return metadata.output_root / f"wavelet_mask_{core_name}"


def covariance_filename(
    metadata: WaveletMetadata, map_a: Map, map_b: Map, scale: int
) -> Path:
    """
    Generate a filename for a covariance map between two maps at a given scale.
    """
    core_name = f"{map_a.tag}_{map_b.tag}_scale_{scale}"
    core_name += ".fits" if metadata.io_suffix is None else f"{metadata.io_suffix}.fits"

    return f'{metadata.output_root}_wavelet_cov_{core_name}'


def apply_wavelet_transform(
    metadata: WaveletMetadata, map_info: Map
) -> dict[int, tuple[Path, Path, float]]:
    """
    Apply a wavelet transform to a map, and save out the data to files,
    one for each scale.

    Returns a dictionary of scales to filenames of the output maps and their masks.
    """

    # Load the map and mask
    imap = map_info.read_map()
    mask = map_info.read_mask()

    ells = np.arange(max(metadata.basis.lpeaks))
    beam_ratios = metadata.output_beam(ells) / map_info.beam(ells)
    scales = map_info.scales(metadata.basis)
    print("Wavelet transform...")
    a = time.time()
    wavecs = metadata.wt.map2wave(
        imap, fl=beam_ratios, scales=scales, fill_value=np.nan
    )
    print(f"Wavelet transform done in {time.time()-a:.2f} sec.")

    # Loop over all the wavelet scales.
    filenames = {}

    for i, wmap in enumerate(wavecs.maps):
        if i not in scales:
            # Shouldn't this be unreachable state, given we provided this
            # to map2wave?
            continue

        mask_name = mask_filename(metadata, map_info, i)
        map_name = map_filename(metadata, map_info, i)

        omask = enmap.project(mask, wmap.shape, wmap.wcs, order=0)

        enmap.write_map(str(mask_name), omask)
        enmap.write_map(str(map_name), wmap)

        filenames[i] = (map_name, mask_name, map_info.response)

    return filenames


def create_covariance_map(
    metadata: WaveletMetadata, file_a: Path, file_b: Path, output_filename: Path
) -> Path:
    """
    Create a single covariance map between two maps using block smoothing
    """

    map_a = enmap.read_map(str(file_a))
    map_b = enmap.read_map(str(file_b))

    covariance_map = map_a * map_b
    print("Cov smooth..")
    downed = enmap.downgrade(
        covariance_map, metadata.cov_smooth_factor, inclusive=True, op=np.nanmean
    )
    downed[np.isnan(downed)] = 0
    output_map = enmap.upgrade(
        downed, metadata.cov_smooth_factor, inclusive=True, oshape=covariance_map.shape
    )

    enmap.write_map(str(output_filename), output_map)

    return output_filename


def create_covariance_maps(
    metadata: WaveletMetadata,
    maps: list[Map],
    scale: int,
) -> tuple[list[list[Path]], list[Path]]:
    """
    Creates all covariance maps for a given scale. Must be ran in a dask
    task as it uses get_client()
    """

    # Only need to do the upper quadrant and diagonal, since the covariance
    # matrix is symmetric.

    client = get_client()

    filtered_maps = [map for map in maps if scale in map.scales(metadata.basis)]

    covariance_maps = [[None] * len(filtered_maps) for _ in range(len(filtered_maps))]
    futures = []

    for i, map_a in enumerate(filtered_maps):
        for j, map_b in enumerate(filtered_maps):
            if i > j:
                continue

            output_filename = covariance_filename(metadata, map_a, map_b, scale)

            map_a_filename = map_filename(metadata, map_a, scale)
            map_b_filename = map_filename(metadata, map_b, scale)

            future = client.submit(
                create_covariance_map,
                metadata,
                map_a_filename,
                map_b_filename,
                output_filename,
            )

            futures.append(future)

            # Save the filenames in the arrays:
            covariance_maps[i][j] = output_filename
            covariance_maps[j][i] = output_filename

    # Return scale so we know which one it is when the future completes...
    return covariance_maps, futures, scale


def create_all_wavelet_maps(
    metadata: WaveletMetadata,
    maps: list[Map],
    client: Client,
) -> dict[int, tuple[Path, Path]]:
    """
    Create all wavelet maps for all maps. Blocks internally until
    all maps are created.
    """

    wavelet_map_futures = [
        client.submit(apply_wavelet_transform, metadata, map_info) for map_info in maps
    ]

    all_wavelet_maps = {}

    for future in as_completed(wavelet_map_futures):
        wavelet_maps = future.result()
        for scale, new_map_filenames in wavelet_maps.items():
            all_wavelet_maps[scale] = all_wavelet_maps.get(scale, []) + [
                new_map_filenames
            ]

    return all_wavelet_maps


def create_covariance_maps_all_scales(
    metadata: WaveletMetadata,
    maps: list[Map],
    scales: list[int],
    client: Client,
) -> dict[int, list[list[Path]]]:
    covariance_maps = [
        client.submit(create_covariance_maps, metadata, maps, scale) for scale in scales
    ]

    all_covariance_maps = {
        scale: cov_maps
        for (_, (cov_maps, _, scale)) in as_completed(covariance_maps, with_results=True)
    }

    return all_covariance_maps


def wavelet_prepare(
    client: Client,
    metadata: WaveletMetadata,
    maps: list[Map],
) -> dict[int, Coadder]:
    """
    Prepares your input maps for a multi-scale coaddition by
    wavelet transforming them.
    """

    # Create all wavelet maps
    all_wavelet_maps = create_all_wavelet_maps(metadata, maps, client)

    # Create all covariance maps
    all_covariance_maps = create_covariance_maps_all_scales(
        metadata,
        maps,
        scales=list(all_wavelet_maps.keys()),
        client=client,
    )

    # Create Coadder objects for each scale
    coadders = {}

    for scale, covariance_maps in all_covariance_maps.items():
        map_filenames, mask_filenames, responses = list(zip(*all_wavelet_maps[scale]))

        coadder = Coadder(
            maps=map_filenames,
            masks=mask_filenames,
            covariance_maps=covariance_maps,
            responses=responses,
        )

        coadders[scale] = coadder

    return coadders


def wavelet_to_map(
    primary_map: Path,
    primary_mask: Path,
    metadata: WaveletMetadata,
    coadd_results: dict[int, da.Array],
) -> enmap.ndmap:
    """
    Convert a set of coadded wavelet maps back to the pixel domain.
    """

    # We need to generate an empty wavelet transform.
    primary = enmap.read_map(str(primary_map))

    wavecs = (
        metadata.wt.map2wave(primary, scales=coadd_results.keys(), fill_value=np.nan)
        * 0.0
    )

    for scale, wavelet_map in coadd_results.items():
        wavecs.maps[scale] = enmap.enmap(wavelet_map.compute(), metadata.wcs)

    coadded_map = metadata.wt.wave2map(wavecs)

    mask = enmap.read_map(str(primary_mask))

    coadded_map[mask == 0] = 0.0

    return coadded_map


if __name__ == "__main__":
    # Simple test...
    from coberus.core import coadd

    metadata = WaveletMetadata.from_primary_map(
        Path("example/map1.fits"),
        basis=wavelets.CosineNeedlet([800.0, 1000.0, 2000.0, 3000.0, 4000.0]),
        output_beam_fwhm=1.6,
        output_root=Path("example/wavelet"),
        cov_smooth_factor=64,
    )

    def beam(ells: float) -> float:
        """
        Calculate the output beam for a given (array) of multipoles.
        """
        fwhm_radians = np.deg2rad(5.0 / 60.0)
        square_ells = ells * ells

        prefactor = -(fwhm_radians * fwhm_radians) / (16.0 * math.log(2))

        return np.exp(prefactor * square_ells)

    maps = [
        Map(
            tag="map1",
            path=Path("example/map1.fits"),
            mask=Path("example/map1_mask.fits"),
            lmin=0,
            lmax=3000,
            response=1.0,
            beam=beam,
        ),
        Map(
            tag="map2",
            path=Path("example/map2.fits"),
            mask=Path("example/map2_mask.fits"),
            lmin=0,
            lmax=3000,
            response=1.0,
            beam=beam,
        ),
    ]

    client = Client()

    coadders = wavelet_prepare(client, metadata, maps)

    print(coadders)

    result_maps = {s: coadd(client, coadder) for s, coadder in coadders.items()}

    coadded_map = wavelet_to_map(
        Path("example/map1.fits"),
        Path("example/map1_mask.fits"),
        metadata,
        result_maps,
    )

    enmap.write_map("example/coadded_map.fits", coadded_map)

    # Clean up
    for _, coadder in coadders.items():
        coadder.cleanup()

    client.close()
