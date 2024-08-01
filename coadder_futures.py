"""
Coadder for maps. Uses linear algebra magic to add things, and dask
to parallelise it all.
"""

from pydantic import BaseModel
from pathlib import Path

import numpy as np
import numba
import dask.array as da
import dask
from dask.distributed import Client
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
    pixels_per_chunk: int = 400

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
                chunks.append((self.pixels_per_chunk, self.pixels_per_chunk))

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
                hdul[0]
                .data[chunk[0][0] : chunk[1][0], chunk[0][1] : chunk[1][1]]
                .astype(np.float32)
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
                hdul[0]
                .data[chunk[0][0] : chunk[1][0], chunk[0][1] : chunk[1][1]]
                .astype(bool)
            )

    return np.array(data)


def read_covariances_chunk(maps: list[list[Path]], chunk: Chunk) -> np.ndarray:
    """
    Reads the covariances from the disk. Note that we only read the upper
    left triangle of the covariance matrix and repeat that for the lower right.
    """

    data = np.empty(
        (len(maps), len(maps), chunk[1][0] - chunk[0][0], chunk[1][1] - chunk[0][1]),
        dtype=np.float32,
    )

    for i, map in enumerate(maps):
        for j in range(i + 1):
            with fits.open(map[j]) as hdul:
                data[i, j] = hdul[0].data[
                    chunk[0][0] : chunk[1][0], chunk[0][1] : chunk[1][1]
                ]
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
    output = np.zeros((n_x, n_y))

    for i in range(n_x):
        for j in range(n_y):
            mask = masks[:, j, i]

            masked_maps = maps[mask, j, i]
            n_out = masked_maps.size

            if n_out == 0:
                output[i, j] = 0.0
                continue

            cov = covariance_maps[:, :, j, i][:, mask][mask, :].reshape((n_out, n_out))
            a = responses[mask]

            cinva = np.linalg.solve(cov, a)
            denom = np.dot(a, cinva)

            if denom == 0.0:
                output[i, j] = 0.0
                continue

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


def coadded_map_wrapper(
    maps: np.ndarray,
    covariance_maps: np.ndarray,
    masks: np.ndarray,
    responses: np.ndarray,
    chunk: Chunk,
) -> np.ndarray:
    """
    Wrapper for the coadd_maps_pixels function that allows you to return
    the chunk to.
    """

    return coadd_maps_pixels(maps, covariance_maps, masks, responses), chunk


def create_tasks_for_chunk(
    chunk: Chunk, client: Client, coadder: Coadder, main_array: da.Array
):
    maps = client.submit(read_maps_chunk, coadder.maps, chunk)

    masks = client.submit(read_mask_chunk, coadder.masks, chunk)

    covariances = client.submit(read_covariances_chunk, coadder.covariance_maps, chunk)

    coadded_map = client.submit(
        coadded_map_wrapper,
        maps,
        covariances,
        masks,
        np.array(coadder.responses, dtype=np.float32),
        chunk,
    )

    return coadded_map


if __name__ == "__main__":
    client = Client()

    TEST_DATA_LOCATION = Path("/Users/borrow-adm/Globus/needlet_test/")

    maps = [TEST_DATA_LOCATION / f"map{i}.fits" for i in range(1, 4)]
    masks = [TEST_DATA_LOCATION / f"map1_mask.fits" for i in range(1, 4)]
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

    chunks = coadder.chunk_task_list()

    main_array = np.zeros(coadder.chunk_meta()["meta"].shape, dtype=np.float32)

    results = [
        create_tasks_for_chunk(chunk, client, coadder, main_array) for chunk in chunks
    ]

    for future in dask.distributed.as_completed(results):
        image, chunk = future.result()

        write_to_main_array(image, chunk, main_array)

    np.save("output.npy", main_array)
