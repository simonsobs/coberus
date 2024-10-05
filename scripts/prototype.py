from pixell import enmap,curvedsky as cs, wavelets as wv,uharm,multimap,utils as u
import numpy as np
import utils
import os,sys
from orphics import io,maps
from coberus import coadd
from coberus.wavelets import wavelet_prepare, wavelet_to_map
from dask.distributed import Client

import argparse


if __name__ == '__main__':
    # Parse command line
    parser = argparse.ArgumentParser(description='Make a mask.')
    parser.add_argument("out_name", type=str,help='Name of outputs. Could include a path.')
    parser.add_argument("--fwhm", type=float,default=1.6,help='Output FWHM.')
    parser.add_argument("--cov-smooth-factor", type=int,default=64,help='Covariance smooth factor.')
    parser.add_argument("--nworkers", type=int,default=None,help='Number of workers.')
    parser.add_argument("--basetag", type=str,default='night_pa5_f090',help='Base tag.')
    parser.add_argument("--basis", type=str,default='lensmode',help='Base tag.')
    parser.add_argument("--nomask", action='store_true',help='Whether to mask the sky at all.')
    parser.add_argument("--tags",     type=str,  default=None,help="Comma separated list of tags.")
    args = parser.parse_args()

    tags = utils.parse_tags(args.tags)
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
    lmins,lmaxs,fwhms, c = utils.get_properties('data.yaml',tags)
    beam_func = lambda tag, ells: maps.gauss_beam(ells, fwhms[tags.index(tag)])
    response_func = lambda tag: 1.0

    data_dict = {}
    data_dict['lpeaks'] = utils.get_lpeaks(args.basis)
    data_dict['output_beam_fwhm'] = args.fwhm
    data_dict['cov_smooth_factor'] = args.cov_smooth_factor
    data_dict['primary_map_tag'] = args.basetag # We will extract on to this geometry and use its mask for the final mask
    data_dict['output_root'] = f'{out_root}/{outname}_data_covsmooth_{args.cov_smooth_factor}'
    data_dict['output_map'] = f'{out_root}/{outname}_data_covsmooth_{args.cov_smooth_factor}_coadd_map.fits'
    data_dict['map_metadata'] = {}
    for i,tag in enumerate(tags):
        data_dict['map_metadata'][tag] = {}
        data_dict['map_metadata'][tag]['path'] = map_fname_func(tag)
        data_dict['map_metadata'][tag]['mask'] = mask_fname_func(tag)
        data_dict['map_metadata'][tag]['response'] = 1.0
        data_dict['map_metadata'][tag]['lmin'] = lmins[i]
        data_dict['map_metadata'][tag]['lmax'] = lmaxs[i]
        data_dict['map_metadata'][tag]['beam'] = {'fwhm': fwhms[i]}
        data_dict['map_metadata'][tag]['postprocess_mask'] = fmproc



    input_data = CoberusInput.parse_obj(data_dict)
    wavelet_metadata = input_data.to_wavelet_metadata()
    maps = input_data.to_maps()
    primary_map = [m.path for m in maps if m.tag == input_data.primary_map_tag][0]
    primary_mask = [m.mask for m in maps if m.tag == input_data.primary_map_tag][0]

    with Client(n_workers=input_data.n_workers) as client:
        coadders = wavelet_prepare(client=client, metadata=wavelet_metadata, maps=maps)

        result_maps = {
            s: coadd(client=client, coadder=coadder) for s, coadder in coadders.items()
        }

        coadded_map = wavelet_to_map(
            primary_map=primary_map,
            primary_mask=primary_mask,
            metadata=wavelet_metadata,
            coadd_results=result_maps,
        )

        # Save the final map
        coadded_map.write(str(input_data.output_map))

    # Clean up intermediate files.
    for _, coadder in coadders.items():
        coadder.cleanup()

    
    # # cutbox = [[4.-2,4.-9],[-4.-2,-4.-9]]
    # coadd_map = pipeline.needlet_coadd(map_fname_func, mask_fname_func, tags, base_tag,
    #                                    lpeaks, lmins, lmaxs, response_func, beam_func,
    #                                    out_beam_fwhm, out_root,cov_smooth_factor=args.cov_smooth_factor,
    #                                    mask_postprocess_func=fmproc,
    #                                    n_workers=args.nworkers,io_suffix=f'_data_covsmooth_{args.cov_smooth_factor}',
    #                                    delete_intermediate=True)


    # print(coadd_map.shape, coadd_map.wcs)
    # # plot(coadd_map,"all",0,mtype='coadd',colorbar=True,grid=True,ticks=10)
    # # smap = coadd_map.submap(np.asarray(cutbox)*u.degree)
    # # plot(smap,"all",0,mtype='coadd_submap',colorbar=True,grid=True,ticks=0.5) # these are input maps
    # enmap.write_map(f'{out_root}/{outname}_coadd_map.fits',coadd_map)
