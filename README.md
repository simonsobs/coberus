# Coberus

A python library for co-adding maps in parallel. Coberus works by splitting your
map up into chunks (by default of 400x400 pixels). Each thread is handed one
chunk, and it goes to read the constituent chunks of the underlying maps from
disk. Co-addition then happens on a pixel-by-pixel level, with the chunking used
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


#### Needlet ILC Pipeline - `needlet_coadd`

For Needlet Internal Linear Combination (NILC) map coaddition, we recommend
using the `pipeline.needlet_coadd` function. It will wavelet-decompose
a set of input maps, empirically estimate the covariance and co-add
the maps using Dask-distributed parallelism.

```python
from coberus import pipeline
import numpy as np

# Tags identify each input dataset
tags = ['a', 'b']
base_tag = 'a'  # the map tag corresponding to the final geometry

# Functions that return file paths given a tag
map_fname_func  = lambda tag: f'maps/imap_{tag}.fits'
mask_fname_func = lambda tag: f'masks/mask_{tag}.fits'

# Beam: returns the harmonic-space beam for a tag at multipoles ells
# e.g. all maps have fwhm = 5.0 arcmin
fwhm = 5.0
beam_func = lambda tag, ells: np.exp(-0.5 * ells * (ells + 1) * (fwhm * np.pi / 180 / 60) ** 2 / (8 * np.log(2)))

# Response: SED of the component to preserve (1 for CMB) evaluated
# for each input tag (corresponding to that frequency or passband)
response_func = lambda tag: 1.0

# Cosine needlet peaks define the wavelet basis
lpeaks = [0, 500, 750, 1000, 1250, 1500]

# Multipole range each tag contributes to
lmins = [0, 0]
lmaxs = [1500, 1500]

# Output beam FWHM in arcminutes
out_beam_fwhm = 5.0

# Directory for intermediate wavelet/covariance files (use RAMdisk if possible)
# and index with your simulation number
out_root = '/dev/shm/my_job_0_'

result = pipeline.needlet_coadd(
    map_fname_func, mask_fname_func, tags, base_tag,
    lpeaks, lmins, lmaxs, response_func, beam_func,
    out_beam_fwhm, out_root,
    cov_smooth_type='gaussian',  # 'block', 'gaussian', or 'tophat'
    n_workers=10, # For Dask parallelization
)
coadd_map = result['coadd']
```

**Key inputs:**
- **`map_fname_func(tag)`** — path to the input map (pre-masked and apodized for stable wavelet transforms).
- **`mask_fname_func(tag)`** — path to a *binary* mask indicating which pixels to include.
- **`beam_func(tag, ells)`** — harmonic-space beam profile; maps are reconvolved to `out_beam_fwhm`.
- **`response_func(tag)`** — SED response of the preserved component (1 for CMB).
- **`lpeaks`** — peak multipoles defining the cosine needlet basis.
- **`lmins` / `lmaxs`** — per-tag multipole limits controlling which needlet scales each tag enters.
- **`out_root`** — path prefix for intermediate files; `/dev/shm/` is recommended for speed.

**Optional features:**
- **`deproj_response_funcs`** — list of response functions for constrained ILC deprojection (e.g. tSZ removal).
- **`cov_smooth_type`** — covariance smoothing method: `'block'` (default), `'gaussian'` ([2307.01043](https://arxiv.org/abs/2307.01043)), or `'tophat'` ([2307.01258](https://arxiv.org/abs/2307.01258)).
- **`nmap_labels` / `nmap_label_fname_func`** — coadd additional maps (e.g. simulations) using weights derived from the data maps; results returned as `result['<label>_coadd']`.
- **`delete_intermediate=True`** — clean up wavelet/covariance files after completion.

See `test_notebook.ipynb` for a complete worked example with simulated CMB maps.


#### Naming

The name `coberus` is a play on `coadder`. 'Adders' are a common species of
[snake in Europe](https://en.wikipedia.org/wiki/Adder), and have the species
designation of _Vipera berus_.