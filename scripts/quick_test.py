from coberus import pipeline
import numpy as np
from orphics import maps,io,cosmology
from pixell import enmap,curvedsky as cs,utils as u

"""
A quick self-contained test of most coberus pipeline features.
"""

if __name__ == '__main__':

    # Let's use the RAMdisk
    out_root = '/dev/shm/'

    tags = ['m1','m2']
    base_tag = 'm1'
    lpeaks = [0.,100.,500.,800.,1000.]
    lmins = [0,0]
    lmaxs = [1000,1000]
    response_func = lambda tag: 1.0
    deproj_response_funcs = [lambda tag: [-1.,0.][tags.index(tag)]]
    fwhms = [4.0,4.0]
    noises = [10.0,12.0]
    out_beam_fwhm = 4.0

    map_fname_func = lambda tag: f'{out_root}map_{tag}.fits'
    mask_fname_func = lambda tag: f'{out_root}mask_{tag}.fits'
    beam_func = lambda tag,ells: maps.gauss_beam(ells,fwhms[tags.index(tag)])

    shape,wcs = enmap.band_geometry(np.array((-70,35.))*u.degree,res=16.0*u.arcmin,variant='fejer1')
    theory = cosmology.default_theory()
    ells = np.arange(1000.)
    cltt = theory.lCl('TT',ells)*maps.gauss_beam(4.0,ells)**2.
    cmap = cs.rand_map(shape,wcs,cltt)
    for i,tag in enumerate(tags):
        print("Making maps...")
        enmap.write_map(map_fname_func(tag),cmap + maps.white_noise(shape,wcs,noises[i]))
        enmap.write_map(mask_fname_func(tag),cmap*0+1)


    outmaps = pipeline.needlet_coadd(map_fname_func, mask_fname_func, tags, base_tag,
                      lpeaks, lmins, lmaxs, response_func, 
                      beam_func, out_beam_fwhm, out_root, deproj_response_funcs = deproj_response_funcs,
                      cov_smooth_type='block', cov_smooth_factor=64, ilc_bias_tol=0.01, 
                      fft_smooth=False, smooth_mean_cov=True, cov_smooth_scales=None, 
                      use_annulus=False, annulus_fwhm_ratio=0.5, map_postprocess_func=None, 
                      mask_postprocess_func=None, n_workers=None,delete_intermediate=False,
                      nmap_labels=[], nmap_label_fname_func=None)

    print(outmaps)
