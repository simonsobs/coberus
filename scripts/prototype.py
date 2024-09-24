from pixell import enmap,curvedsky as cs, wavelets as wv,uharm,multimap,utils as u
import numpy as np
import utils
import os,sys
from orphics import io,maps
from coberus import Coadder, coadd
from coberus import pipeline
from dask.distributed import Client

"""

To do
- Covariance smoothing factors
- Data model
- Beam-bandpass factors in responses

Properties of a tag:
- lmin and lmax
- beam
- bandpass / central freq


"""


if __name__ == '__main__':


    out_root = utils.out_root
    outname = 'test'
    gal = '80'

    map_fname_func = lambda tag: f'{out_root}/{outname}_{tag}_map.fits'
    mask_fname_func = lambda tag: f'{out_root}/{outname}_{tag}_mask{gal}.fits'
    base_tag = 'night_pa5_f090' # We will extract on to this geometry and use its mask for the final mask
    tags = [base_tag,'143','daydeep_pa5_f150']
    #lpeaks = [0.,100.,500.,800.,1000.,2000.,3000.,4000., 5000., 6000., 8000,10000.]
    lpeaks = [0.,100.,500.,800.,1000.,2000.,3000.,4000.]
    lmins = [500,0,500]
    lmaxs = [None,3000,None]
    fwhm = [2.2,7.0,1.4]
    beam_func = lambda tag, ells: maps.gauss_beam(ells, fwhm[tags.index(tag)])
    response_func = lambda tag: 1.0
    out_beam_fwhm = 1.6
    # cutbox = [[4.-2,4.-9],[-4.-2,-4.-9]]

    coadd_map = pipeline.needlet_coadd(map_fname_func, mask_fname_func, tags, base_tag,
                   lpeaks, lmins, lmaxs, response_func, beam_func, out_beam_fwhm, out_root)

    print(coadd_map.shape, coadd_map.wcs)
    # plot(coadd_map,"all",0,mtype='coadd',colorbar=True,grid=True,ticks=10)
    # smap = coadd_map.submap(np.asarray(cutbox)*u.degree)
    # plot(smap,"all",0,mtype='coadd_submap',colorbar=True,grid=True,ticks=0.5) # these are input maps
    enmap.write_map(f'{out_root}/{outname}_coadd_map.fits',coadd_map)
