from __future__ import print_function
from orphics import maps, io
from pixell import enmap, curvedsky as cs, utils as u
import numpy as np

"""
This script saves a small (47 GB) test dataset for map
coaddition pipeline plumbing tests.

map{i}.fits  # the map
map{i}_mask.fits # the mask  (mostly ones and zeros, but not always)
cov_map{i}_map{j}.fits  # covariance maps
The covariance is trivial (just ones on the diagonal, zeros on the offdiagonal).

But this is just a plumbing exercise anyway.  You can probably create as many copies of these maps 
as you want without making the linear algebra break, to test the scalability of the pipeline.
The maps are CAR projection at 0.5 arcmin pixel resolution, and go from declination -70 to 30 deg.
"""

tshape, twcs = enmap.band_geometry(
    np.asarray((-70, 30)) * u.degree, res=0.5 * u.arcmin, variant="fejer1"
)
mlmax = 30000


def wmap(fname, oname):
    print(fname)
    omap = cs.alm2map(
        cs.map2alm(enmap.read_map(fname)[0], lmax=mlmax), enmap.empty(tshape, twcs)
    )
    io.hplot(omap, oname.replace(".fits", ""), downgrade=4, mask=0)
    enmap.write_map(oname, omap)
    return omap


def wmap_ivar(fname, oname, rmin=1e-3, rmax=90.0):
    print(fname)
    ivar = enmap.read_map(fname)
    rms = maps.rms_from_ivar(ivar)
    rms[ivar <= 0] = 0
    omap = enmap.project(rms, tshape, twcs, order=0)

    omap[omap < rmin] = 0
    omap[omap > rmax] = 0
    omap[np.logical_and(omap >= rmin, omap <= rmax)] = 1
    io.hplot(omap, oname.replace(".fits", ""), downgrade=4, mask=0)
    enmap.write_map(oname, omap)


# ProductDB entry: https://www.productdb.actcmb.org/product/map/planck_npipe_v1
omap = wmap("planck_npipe_143_coadd_srcfree.fits", "map1.fits")
enmap.write_map("map1_mask.fits", omap * 0 + 1)

# ProductDB entries: https://www.productdb.actcmb.org/product/map/act_dr6v4
wmap("cmb_daydeep_pa6_f090_3pass_4way_coadd_map_srcfree.fits", "map2.fits")
wmap("cmb_daywide_pa6_f150_3pass_4way_coadd_map_srcfree.fits", "map3.fits")

wmap_ivar(
    "cmb_daydeep_pa6_f090_3pass_4way_coadd_ivar.fits", "map2_mask.fits", rmax=40.0
)
wmap_ivar(
    "cmb_daywide_pa6_f150_3pass_4way_coadd_ivar.fits", "map3_mask.fits", rmax=250.0
)

ones = omap * 0 + 1
zeros = omap * 0

for i in range(1, 4):
    for j in range(i, 4):
        enmap.write_map(f"cov_map{i}_map{j}.fits", ones if i == j else zeros)
