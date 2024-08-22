# Coberus

A python library for co-adding maps in parallel. Coberus works by splitting your
map up into chunks (by default of 400x400 pixels). Each thread is handed one
chunk, and it goes to read the constituent chunks of the underlying maps from
disk. Co-addition then happens on a pxiel-by-pixel level, with the chunking used
to reduce file access overhead.

### Setup

You can install this repository using `uv` or `pip`.

```
{uv} pip install git+https://github.com/simonsobs/map-coaddition
```

You can then use the library in your code as `coberus`.

### Example usage

There are multiple ways to use coberus.

#### Simple - CLI

This relies on you having a pre-prepared directory with all of your maps
having the correct names, with the following format:

1. `map{N}.fits` for maps with index 1-N.
2. `map{N}_mask.fits` for masks for maps.
3. `cov_map{N}_map{M}.fits` for covariance maps between N and M.
4. `responses.json` which is a list of floats (e.g. just `[1.0, 1.0, 1.0, ...]`)
   giving the responses of the maps in order.

You can then run, assuming there are 5 maps and they live in `maps`:

```
coberus --input=./maps --number=5 --output=coadded.fits
```

#### Advanced - Creating your own Coadder

This is useful if you have non-conforming filenames or you are trying to
connect to an existing dask cluster.

`coberus` is made up of one core object, `Coadder`, and one main
function `coadd`.

```python
from coberus import Coadder, coadd
from dask.distributed import Client

maps = ["mymap_a.fits", "mymapb.fits"]
masks = ["ma_a.fits", "ma-b.fits"]
covariance_maps = [
    ["a_a.fits", "a_b.fits"],
    ["a_b.fits", "b_b.fits"],
]
repsonses = [0.1, 0.9]

coadder = Coadd(
    maps=maps,
    masks=masks,
    covariance_maps=covariance_maps,
    responses=responses
)

client = Client()

# Result is a dask array
result = coadd(client, coadder)

# This is now a numpy array
arr = result.compute()
```

There are utility functions in `coberus.fits` to write the dask array to file.


#### Naming

The name `coberus` is a play on `coadder`. 'Adders' are a common species of
[snake in Europe](https://en.wikipedia.org/wiki/Adder), and have the species
designation of _Vipera berus_.