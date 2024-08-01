"""
Coadder for maps. Uses linear algebra magic to add things, and dask
to parallelise it all.
"""

from pydantic import BaseModel
from pathlib import Path

import numpy as np
import numba
import dask.array as da
from astropy.io import fits

Chunk = tuple[tuple[int], tuple[int]]


class Coadder(BaseModel):
    """
    Inputs are ordered.

    maps: list[Path]
        A list of paths to the real maps.

    masks: list[Path]
        A list of paths to masks for the real maps.

    responses: list[float]
        The responses to use for those masks

    covariance_maps: list[list[Path]]
        A 2D matrix of paths stating the covariance maps for the real maps.
        E.g. covariance_maps[1][2] is the covariance between maps[1] and maps[2].
        This is by construction a symmetric matrix, so the ordering between the
        two indicies doesn't matter.
    """

    # Input variables describing data
    maps: list[Path]
    masks: list[Path]
    responses: list[float]

    covariance_maps: list[list[Path]]

    # Settings for the map coadder.
    pixels_per_chunk: int = 800

    def chunk_meta(self) -> dict[str, any]:
        with fits.open(self.maps[0]) as hdul:
            meta = hdul[0].data

        chunks_size = (self.pixels_per_chunk, self.pixels_per_chunk)

        return {"meta": meta, "chunks": chunks_size}
    
    def chunk_simple(self) -> tuple[tuple[int]]:
        with fits.open(self.maps[0]) as hdul:
            n_y, n_x = hdul[0].data.shape

        # Need to be careful - pixels_per_chunk may not be aligned with
        # the map size.
        chunks = []

        for i in range(0, n_y, self.pixels_per_chunk):
            for j in range(0, n_x, self.pixels_per_chunk):
                chunks.append(
                    (self.pixels_per_chunk, self.pixels_per_chunk)
                )

        return chunks

    def chunk_task_list(self) -> list[Chunk]:
        """
        Opens the first map and returns a list of tuples that describe
        the chunks of the map that need to be processed.
        """

        with fits.open(self.maps[0]) as hdul:
            n_y, n_x = hdul[0].data.shape

        # Need to be careful - pixels_per_chunk may not be aligned with
        # the map size.
        chunks = []

        for i in range(0, n_y, self.pixels_per_chunk):
            for j in range(0, n_x, self.pixels_per_chunk):
                chunks.append(
                    (
                        (i, j),
                        (
                            min(i + self.pixels_per_chunk, n_y),
                            min(j + self.pixels_per_chunk, n_x),
                        ),
                    )
                )

        return chunks


def read_maps_chunk(maps: list[Path], chunk: Chunk) -> np.ndarray:
    """
    Reads a chunk of maps from the disk.
    """

    data = []

    for map in maps:
        with fits.open(map) as hdul:
            data.append(
                hdul[0].data[chunk[0][0] : chunk[1][0], chunk[0][1] : chunk[1][1]].astype(np.float32)
            )

    return np.array(data)

def read_mask_chunk(masks: list[Path], chunk: Chunk) -> np.ndarray:
    """
    Reads a chunk of masks from the disk. Includes the conversion to bool.
    """

    data = []

    for mask in masks:
        with fits.open(mask) as hdul:
            data.append(
                hdul[0].data[chunk[0][0] : chunk[1][0], chunk[0][1] : chunk[1][1]].astype(bool)
            )

    return np.array(data)


def read_covariances_chunk(maps: list[list[Path]], chunk: Chunk) -> np.ndarray:
    """
    Reads the covariances from the disk. Note that we only read the upper
    left triangle of the covariance matrix and repeat that for the lower right.
    """

    data = np.empty((len(maps), len(maps), chunk[1][0] - chunk[0][0], chunk[1][1] - chunk[0][1]), dtype=np.float32)

    for i, map in enumerate(maps):
        for j in range(i + 1):
            with fits.open(map[j]) as hdul:
                data[i, j] = hdul[0].data[chunk[0][0] : chunk[1][0], chunk[0][1] : chunk[1][1]]
                data[j, i] = data[i, j]

    return data

@numba.njit(fastmath=True)
def coadd_maps_pixels(
    maps: np.ndarray,
    covariance_maps: np.ndarray,
    masks: np.ndarray,
    responses: np.ndarray,
) -> np.ndarray:
    """
    Co-adds the maps in the pixel domain. Assumes all masks and maps are the same size.
    """

    n_x, n_y = maps.shape[-2:]
    output = np.zeros((n_y, n_x))

    for i in range(n_y):
        for j in range(n_x):
            mask = masks[:, i, j]

            masked_maps = maps[mask, i, j]
            n_out = masked_maps.size

            cov = covariance_maps[:, :, i, j][:,mask][mask,:].reshape((n_out, n_out))
            a = responses[mask]

            cinva = np.linalg.solve(cov, a)
            denom = np.dot(a, cinva)

            cinvd = np.linalg.solve(cov, masked_maps)
            numer = np.dot(a, cinvd)

            output[i, j] = numer / denom

    return output


def write_to_main_array(data: np.ndarray, chunk: Chunk, main_array: da.array):
    """
    Writes the data to the main array.
    """

    main_array[chunk[0][0] : chunk[1][0], chunk[0][1] : chunk[1][1]] = data

    return


def run_tasks_for_block(x=None, block_info=None) -> da.array:
    dask_chunk = block_info[None]["array-location"]
    chunk = ((dask_chunk[0][0], dask_chunk[1][0]), (dask_chunk[0][1], dask_chunk[1][1]))

    # From block info, transform into our idea of a chunk.
    maps_chunk = read_maps_chunk(coadder.maps, chunk) 
    masks_chunk = read_mask_chunk(coadder.masks, chunk)
    covariances_chunk = read_covariances_chunk(coadder.covariance_maps, chunk)

    return coadd_maps_pixels(maps_chunk, covariances_chunk, masks_chunk, np.array(coadder.responses, dtype=np.float32))

if __name__ == "__main__":
    from dask.distributed import Client
    client = Client() 

    TEST_DATA_LOCATION = Path("/Users/borrow-adm/Globus/needlet_test/")

    maps = [TEST_DATA_LOCATION / f"map{i}.fits" for i in range(1, 4)]
    masks = [TEST_DATA_LOCATION / f"map{i}_mask.fits" for i in range(1, 4)]
    responses = [1.0, 1.0, 1.0]
    covariance_maps = [
        [
            TEST_DATA_LOCATION
            / (f"cov_map{i}_map{j}.fits" if i < j else f"cov_map{j}_map{i}.fits")
            for j in range(1, 4)
        ]
        for i in range(1, 4)
    ]

    coadder = Coadder(
        maps=maps, masks=masks, responses=responses, covariance_maps=covariance_maps
    )

    meta = coadder.chunk_meta()["meta"]
    chunks = coadder.chunk_meta()["chunks"]

    # We have to create this dummy array for the dask array to play nice.
    dask_array = da.zeros(meta.shape, chunks=chunks, dtype=np.float32)

    # Test it all.
    output = da.map_blocks(
        run_tasks_for_block,
        dask_array,
        dtype=np.float32,
        chunks=chunks,
        meta=np.array((), dtype=np.float32)
    ).compute()

    print(output)

    # Test the whole thing.
