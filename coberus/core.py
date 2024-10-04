"""
Coadder for maps. Uses linear algebra magic to add things, and dask
to parallelise it all.
"""

import dask.array
from pydantic import BaseModel
from pathlib import Path

import numpy as np
import numba
import dask.array as da
import dask
from dask.distributed import Client
from astropy.io import fits

debug = False
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

    def cleanup(self):
        """
        Cleans up all files associated with the Coadder (i.e.
        deletes them from disk!)
        """

        # Generate a list of all files to delete.
        files: set[Path] = set()

        files.update(*self.maps)
        files.update(*self.masks)

        for cov in self.covariance_maps:
            files.update(*cov)

        # Delete all files.
        for file in files:
            file.unlink()

        return


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


def masks_to_skips(masks: np.ndarray) -> list[bool]:
    """
    Converts the masks to skips. If the mask is all False, we skip the map.
    """

    return [not np.any(mask) for mask in masks]


def read_covariances_chunk(
    maps: list[list[Path]], chunk: Chunk, skip: list[bool]
) -> np.ndarray:
    """
    Reads the covariances from the disk. Note that we only read the upper
    left triangle of the covariance matrix and repeat that for the lower right.
    """

    data = np.eye(len(maps), dtype=np.float32)[..., None, None] * np.ones(
        (chunk[1][0] - chunk[0][0], chunk[1][1] - chunk[0][1]), dtype=np.float32
    )

    for i, map in enumerate(maps):
        if skip[i]:
            continue
        for j in range(i + 1):
            if skip[j]:
                continue
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

    n_y, n_x = maps.shape[-2:]
    output = np.zeros((n_y, n_x), dtype=np.float32)

    for i in range(n_x):
        for j in range(n_y):
            mask = masks[:, j, i]

            masked_maps = maps[mask, j, i]
            n_out = masked_maps.size

            if n_out == 0:
                output[j, i] = 0.0
                continue

            if debug:
                print(covariance_maps[:, :, j, i])
            cov = covariance_maps[:, :, j, i][:, mask][mask, :].reshape((n_out, n_out))
            if debug:
                print(cov)
            a = responses[mask]

            cinva = np.linalg.solve(cov, a)
            denom = np.dot(a, cinva)

            if denom == 0.0:
                output[j, i] = 0.0
                continue

            cinvd = np.linalg.solve(cov, masked_maps)
            numer = np.dot(a, cinvd)

            output[j, i] = numer / denom

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

    skips = client.submit(masks_to_skips, masks)

    covariances = client.submit(
        read_covariances_chunk, coadder.covariance_maps, chunk, skips
    )

    coadded_map = client.submit(
        coadded_map_wrapper,
        maps,
        covariances,
        masks,
        np.array(coadder.responses, dtype=np.float32),
        chunk,
    )

    return coadded_map


def coadd(client: Client, coadder: Coadder) -> da.Array:
    """
    The main function for co-adding maps. Takes your Coadder object, and
    a Dask client, and returns a Dask array filled with your coadded map.
    This function uses Dask futures to coadd your maps.
    """
    chunks = coadder.chunk_task_list()

    main_array = da.zeros(coadder.chunk_meta()["meta"].shape, dtype=np.float32)

    results = [
        create_tasks_for_chunk(chunk, client, coadder, main_array) for chunk in chunks
    ]

    for future in dask.distributed.as_completed(results):
        image, chunk = future.result()

        write_to_main_array(image, chunk, main_array)

    return main_array
