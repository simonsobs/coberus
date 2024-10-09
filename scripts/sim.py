from pixell import enmap,curvedsky as cs, wavelets as wv,uharm,multimap,utils as u
import numpy as np
import utils
import os,sys
from orphics import io,maps
from coberus import pipeline
import healpy as hp

import argparse


if __name__ == '__main__':
    # Parse command line
    parser = argparse.ArgumentParser(description='Make a mask.')
    parser.add_argument("out_name", type=str,help='Name of outputs. Could include a path.')
    parser.add_argument("--fwhm", type=float,default=1.6,help='Output FWHM.')
    parser.add_argument("--cov-smooth-factor", type=int,default=64,help='Covariance smooth factor.')
    parser.add_argument("--sim-index", type=int,default=0,help='Sim index.')
    parser.add_argument("--iset", type=int,default=0,help='Sim set index.')
    parser.add_argument("--nworkers", type=int,default=None,help='Number of workers.')
    parser.add_argument("--basetag", type=str,default='night_pa5_f090',help='Base tag.')
    parser.add_argument("--basis", type=str,default='lensmode',help='Base tag.')
    parser.add_argument("--tags",     type=str,  default=None,help="Comma separated list of tags.")
    parser.add_argument("--skip-noise", action='store_true',help='Skip noise.')
    parser.add_argument("--do-noise-only", action='store_true',help='Do noise only as well.')
    parser.add_argument("--ramdisk",     type=str,  default=None,help="Environment variable for RAM-disk.")
    args = parser.parse_args()


    tags = utils.parse_tags(args.tags)
    print(tags)
    out_root = utils.out_root
    outname = args.out_name
    simid = args.sim_index

    if args.ramdisk:
        temp_out_dir = os.environ[args.ramdisk]
        if temp_out_dir.strip()=="": raise ValueError
        delete_files = False
    else:
        temp_out_dir = out_root
        delete_files = True


    gal = '80'
    fmproc = None
    mask_fname_func = lambda tag: f'{out_root}/{outname}_{tag}_mask{gal}.fits'
    map_fname_func = lambda tag: f'{temp_out_dir}/{outname}_{tag}_sim_index_{simid}_iset_{args.iset}_map.fits'
    nmap_fname_func = (lambda tag: f'{temp_out_dir}/{outname}_{tag}_noise_sim_index_{simid}_iset_{args.iset}_map.fits') if args.do_noise_only else None


    lmins,lmaxs,fwhms, c = utils.get_properties('data.yaml',tags)
    beam_func = lambda tag, ells: maps.gauss_beam(ells, fwhms[tags.index(tag)])
    base_tag = args.basetag # We will extract on to this geometry and use its mask for the final mask
    shape,wcs = enmap.read_map_geometry(utils.get_filename(base_tag,maptype='map',splitnum=None,srcfree=True))
    shape = shape[-2:]
    dfact = 2
    shape,wcs = enmap.downgrade_geometry(shape, wcs, dfact) # TODO: improve; this doesnt simulate pixwin

    # Make signal sim
    alm = hp.read_alm(utils.cmb_sim_fname(simid,args.iset),hdu=1)
    for i,tag in enumerate(tags):
        print(tag)
        balm = cs.almxfl(alm,lambda ells: beam_func(tag,ells))
        imap = cs.alm2map(balm,enmap.empty(shape,wcs,dtype=np.float32))

        if not(args.skip_noise):
            if c[tag]['exp']=='act':
                ivar = enmap.read_map(utils.get_filename(tag,maptype='ivar',splitnum=None,srcfree=True))
                if ivar.ndim==3: ivar = ivar[0]
                ivar = utils.downgrade(ivar,dfact,op=np.sum)
                nsim = maps.modulated_noise_map(ivar,lknee=3000,alpha=-4,lmax=10000,
                                                seed=(1,i,simid,args.iset),lmin=200)
            elif c[tag]['exp']=='planck':
                nsim = maps.white_noise(shape,wcs,c[tag]['sim_noise'],seed=(2,i,simid,args.iset))
            else:
                raise ValueError
        else:
            nsim = 0.
        
        # Make and save sims
        omap = imap + nsim
        if not(args.skip_noise):
            if c[tag]['exp']=='act':
                omap[ivar<=0] = 0
                nsim[ivar<=0] = 0
                del ivar
        enmap.write_map(map_fname_func(tag),omap.astype(np.float32))
        if args.do_noise_only:
            enmap.write_map(nmap_fname_func(tag),nsim.astype(np.float32))
            
    
    lpeaks = utils.get_lpeaks(args.basis)
    response_func = lambda tag: 1.0
    out_beam_fwhm = args.fwhm
    
    # cutbox = [[4.-2,4.-9],[-4.-2,-4.-9]]
    del omap, imap, nsim, balm
    coadd_map = pipeline.needlet_coadd(map_fname_func, mask_fname_func, tags, base_tag,
                                       lpeaks, lmins, lmaxs, response_func, beam_func,
                                       out_beam_fwhm, temp_out_dir,cov_smooth_factor=args.cov_smooth_factor,
                                       mask_postprocess_func=fmproc,
                                       n_workers=args.nworkers,io_suffix=f'_simid_{simid}_iset_{args.iset}_covsmooth_{args.cov_smooth_factor}',
                                       delete_intermediate=delete_files,nmap_fname_func=nmap_fname_func)


    for key in coadd_map.keys():
        enmap.write_map(f'{out_root}/sim_{simid}_iset_{args.iset}_{outname}_{key}_covsmooth_{args.cov_smooth_factor}.fits',coadd_map[key])


    # Cleanup and delete files
    if not(args.ramdisk):
        for tag in tags:
            os.remove(map_fname_func(tag))
            if args.do_noise_only:
                os.remove(nmap_fname_func(tag))
    
