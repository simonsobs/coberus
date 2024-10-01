"""
Routines for wavelet transforms.
"""

from typing import Any
from pydantic import BaseModel

from pixell import uharm
from pixell import wavelets
from pixell import wcsutils
from pixell import enmap

from pathlib import Path
import numpy as np
import math

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
    lmin: int
    # Maximum multipole to consider
    lmax: int
    # Response value. For CMB solutions, this is 1.0
    response: float
    # Beam function. For a given ell, return the beam value.
    beam: callable[float, float]
    # Scales for this map
    scales: list[int]


class WaveletMetadata(BaseModel):
    """
    Wavelet transform settings.
    """

    shape: tuple[int]
    wcs: wcsutils.WCS
    basis: wavelets.CosineNeedlet

    # The output beam FWHM
    output_beam_fwhm: float

    # Factor to smooth for in covariance maps
    cov_smooth_factor: int = 1

    io_suffix: str | None = None
    output_root: Path

    uht: uharm.UHT | None
    wt: wavelets.WaveletTransform | None = None

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
        self.wt = wavelets.WaveletTransform(self.uht, basis=self.basis)
        self.uht = uharm.UHT(self.shape, self.wcs)
        return super().model_post_init(__context)
    
    @classmethod
    def from_primary_map(cls, filename: Path, basis: wavelets.CosineNeedlet, output_beam_fwhm: float, output_root: Path, cov_smooth_factor: int = 1, ) -> "WaveletMetadata":
        """
        Create a WaveletMetadata object from a primary map.
        """
        shape, wcs = enmap.read_map_geometry(filename)

        return cls(
            shape=shape,
            wcs=wcs, 
            basis=basis,
            output_beam_fwhm=output_beam_fwhm,
            output_root=output_root,
            cov_smooth_factor=cov_smooth_factor,
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

    return metadata.output_root / f"wavelet_cov_{core_name}"


def apply_wavelet_transform(
    metadata: WaveletMetadata, map_info: Map
) -> dict[int, tuple[Path, Path]]:
    """
    Apply a wavelet transform to a map, and save out the data to files,
    one for each scale.

    Returns a dictionary of scales to filenames of the output maps and their masks.
    """

    # Load the map and mask
    imap = enmap.read_map(map_info.path)
    mask = enmap.read_map(map_info.mask)

    ells = np.arange(metadata.lmax)
    beam_ratios = metadata.output_beam(ells) / map_info.beam(ells)
    wavecs = wavelets.map2wave(
        imap, fl=beam_ratios, scales=map_info.scales, fill_value=np.nan
    )

    # Loop over all the wavelet scales.
    filenames = {}

    for i, wmap in enumerate(wavecs.maps):
        if i not in map_info.scales:
            # Shouldn't this be unreachable state, given we provided this
            # to map2wave?
            continue

        mask_name = mask_filename(metadata, map_info, i)
        map_name = map_filename(metadata, map_info, i)

        omask = enmap.project(mask, wmap.shape, wmap.wcs, order=0)

        enmap.write_map(str(mask_name), omask)
        enmap.write_map(str(map_name), wmap)

        filenames[i] = (map_name, mask_name)

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

    downed = enmap.downgrade(
        covariance_map, metadata.cov_smooth_factor, inclusive=True, op=np.nanmean
    )
    downed[np.isnan(downed)] = 0
    output_map = enmap.upgrade(
        downed, metadata.cov_smooth_factor, inclusive=True, op=np.nanmean
    )

    enmap.write_map(str(output_filename), output_map)

    return output_filename


def create_covariance_maps(
    metadata: WaveletMetadata,
    maps: list[Map],
    scale: int,
) -> list[list[Path]]:
    """
    Creates all covariance maps for a given scale.
    """

    # TODO: Write this as a parallel operation? Submit futures instead of
    # waiting for each one to finish?

    # Only need to do the upper quadrant and diagonal, since the covariance
    # matrix is symmetric.

    covariance_maps = [[None] * len(maps) for _ in range(len(maps))]

    for i, map_a in enumerate(maps):
        for j, map_b in enumerate(maps):
            if i > j:
                continue

            output_filename = covariance_filename(metadata, map_a, map_b, scale)

            map_a_filename = map_filename(metadata, map_a, scale)
            map_b_filename = map_filename(metadata, map_b, scale)

            # TODO: Submit this as a task to the dask client
            _ = create_covariance_map(
                metadata, map_a_filename, map_b_filename, output_filename
            )

            # Save the filenames in the arrays:
            covariance_maps[i][j] = output_filename
            covariance_maps[j][i] = output_filename

    return covariance_maps


def create_all_wavelet_maps(
    metadata: WaveletMetadata, maps: list[Map]
) -> dict[int, tuple[Path, Path]]:
    """
    Create all wavelet maps for all maps.
    """

    all_wavelet_maps = {}

    for map_info in maps:
        # TODO: Submit this as a task to the dask client
        wavelet_maps = apply_wavelet_transform(metadata, map_info)

        for scale, new_map_filenames in wavelet_maps.items():
            # TODO: This is not actually parallel as this is a synchronous operation.
            # You'll need to come back and do the dictionary updates afterwards.
            all_wavelet_maps[scale] = all_wavelet_maps.get(scale, []).append(
                new_map_filenames
            )

    return all_wavelet_maps


def create_covariance_maps_all_scales(
    metadata: WaveletMetadata,
    maps: list[Map],
    scales: list[int],
) -> dict[int, list[list[Path]]]:
    all_covariance_maps = {}

    for scale in scales:
        covariance_maps = create_covariance_maps(metadata, maps, scale)

        all_covariance_maps[scale] = covariance_maps

    return all_covariance_maps


def wavelet_prepare(
    metadata: WaveletMetadata,
    maps: list[Map],
) -> list[Coadder]:
    """
    Prepares your input maps for a multi-scale coaddition by
    wavelet transforming them.
    """

    # Create all wavelet maps
    all_wavelet_maps = create_all_wavelet_maps(metadata, maps)

    # Create all covariance maps
    all_covariance_maps = create_covariance_maps_all_scales(
        metadata, maps, scales=list(all_wavelet_maps.keys())
    )

    # Create Coadder objects for each scale
    coadders = []
    
    for scale, covariance_maps in all_covariance_maps.items():
        map_filenames, mask_filenames = list(zip(*all_wavelet_maps[scale]))
        responses = [map_info.response for map_info in maps]

        coadder = Coadder(
            maps=map_filenames,
            masks=mask_filenames,
            covariance_maps=covariance_maps,
            responses=responses,
        )

        coadders.append(coadder)
    
    return coadders


if __name__ == "__main__":
    # Simple test...

    

    metadata = WaveletMetadata.from_primary_map(

    )
