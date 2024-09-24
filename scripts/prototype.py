from pixell import enmap,curvedsky as cs, wavelets as wv,uharm,multimap,utils as u
import numpy as np
import utils
import os,sys
from orphics import io,maps
from coberus import Coadder, coadd
from coberus import pipeline

import argparse


if __name__ == '__main__':
    # Parse command line
    parser = argparse.ArgumentParser(description='Make a mask.')
    parser.add_argument("out_name", type=str,help='Name of outputs. Could include a path.')
    parser.add_argument("--nworkers", type=int,default=None,help='Number of workers.')
    parser.add_argument("--basetag", type=str,default='night_pa5_f090',help='Base tag.')
    parser.add_argument("--basis", type=str,default='lensmode',help='Base tag.')
    parser.add_argument("--nomask", action='store_true',help='Whether to mask the sky at all.')
    parser.add_argument("--tags",     type=str,  default=None,help="Comma separated list of tags.")
    args = parser.parse_args()

    if args.tags is None:
        tags = utils.act_tags + utils.planck_tags
        tags.remove('daywide_pa4_f220')
        tags.remove('daywide_pa6_f090')
        tags.remove('daywide_pa6_f150')
        tags.remove('030')
        tags.remove('044')
        tags.remove('070')
        tags.remove('545')
    else:
        tags = args.tags.split(',')

    out_root = utils.out_root
    outname = args.out_name

    if args.nomask:
        def fmproc(x):
            out = x.copy()
            out[x<=0] = 0
            out[x>0] = 1
            return out
        mask_fname_func = lambda tag: f'{out_root}/{outname}_{tag}_ivar.fits'
    else:
        gal = '80'
        fmproc = None
        mask_fname_func = lambda tag: f'{out_root}/{outname}_{tag}_mask{gal}.fits'


    map_fname_func = lambda tag: f'{out_root}/{outname}_{tag}_map.fits'
    base_tag = args.basetag # We will extract on to this geometry and use its mask for the final mask

    if args.basis=='lensmode':
        lpeaks = [0.,100.,500.,800.,1000.,2000.,3000.,4000.]
    elif args.basis=='szmode':
        lpeaks = np.append([0.,100.,500.,800.,1000.,2000.,3000.,4000.],np.arange(6000.,30000.,4000.))
    elif args.basis=='debug':
        lpeaks = [0,100.,500.,1000.]

    lmins = []
    lmaxs = []
    fwhms = []

    c = io.config_from_yaml('data.yaml')
    print(c)
    def _clean(item):
        if item=='None': return None
        return float(item)

    for tag in tags:
        if ('night' in tag) or ('daywide' in tag) or ('daydeep' in tag):
            freq = tag.split('_')[-1]
            key = f'act_{freq}'
        else:
            key = tag
        lmin = _clean(c[key]['lmin'])
        lmax = _clean(c[key]['lmax'])
        fwhm = _clean(c[key]['beam'])
        lmins.append(lmin)
        lmaxs.append(lmax)
        fwhms.append(fwhm)


    beam_func = lambda tag, ells: maps.gauss_beam(ells, fwhms[tags.index(tag)])
    response_func = lambda tag: 1.0
    out_beam_fwhm = 1.6
    # cutbox = [[4.-2,4.-9],[-4.-2,-4.-9]]

    coadd_map = pipeline.needlet_coadd(map_fname_func, mask_fname_func, tags, base_tag,
                                       lpeaks, lmins, lmaxs, response_func, beam_func, out_beam_fwhm, out_root, mask_postprocess_func=fmproc,n_workers=args.nworkers)

    print(coadd_map.shape, coadd_map.wcs)
    # plot(coadd_map,"all",0,mtype='coadd',colorbar=True,grid=True,ticks=10)
    # smap = coadd_map.submap(np.asarray(cutbox)*u.degree)
    # plot(smap,"all",0,mtype='coadd_submap',colorbar=True,grid=True,ticks=0.5) # these are input maps
    enmap.write_map(f'{out_root}/{outname}_coadd_map.fits',coadd_map)
