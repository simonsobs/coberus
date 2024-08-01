"""
A simple example trying to generate a big dask array.
"""

import numpy as np
import dask.array as da

output_dimensions = (2000, 2000)
chunk_size = (256, 256)

def make_chunk(x=None, block_info=None):
    print(block_info)
    return np.random.rand(*block_info[None]["chunk-shape"])



output = da.map_blocks(
    make_chunk,
    da.zeros(output_dimensions, chunks=chunk_size),
    chunks=chunk_size,
    dtype=np.float64,
    meta=np.array((), dtype=np.float64)
).compute()

print(output.shape)
print(output.mean())