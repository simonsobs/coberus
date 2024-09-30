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

        core_name = f"{map_info.tag}_scale_{i}"
        core_name += (
            ".fits" if metadata.io_suffix is None else f"{metadata.io_suffix}.fits"
        )

        mask_name = metadata.output_root / f"wavelet_mask_{core_name}"
        map_name = metadata.output_root / f"wavelet_map_{core_name}"

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
