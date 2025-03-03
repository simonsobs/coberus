"""
Routines for saving the dask array to a FITS file.
"""

import pixell
from pathlib import Path
from dask.array import Array
from astropy.wcs import WCS

import pixell.enmap


def extract_wcs(input: Path) -> WCS:
    """
    Extract the WCS from a FITS file.
    """

    return pixell.enmap.read_map(str(input), delayed=True).wcs


def save_to_fits(output: Path, array: Array, wcs: WCS):
    """
    Save the dask array to a FITS file.

    Parameters
    ----------

    output: Path
        The path to the output FITS file.
    array: Array
        The dask array to save.
    wcs: WCS
        The WCS object that describes your map. You should extract
        this from one of the co-added maps with ``extract_wcs``.
    """

    array = array.compute()

    output_map = pixell.enmap.ndmap(array, wcs=wcs)

    output_map.write(str(output))

    return
