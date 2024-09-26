from pixell import enmap,curvedsky as cs, wavelets as wv,uharm,multimap,utils as u
import numpy as np
import utils
import os,sys
from orphics import io,maps,stats
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
    parser.add_argument("--nworkers", type=int,default=None,help='Number of workers.')
    parser.add_argument("--basetag", type=str,default='night_pa5_f090',help='Base tag.')
    parser.add_argument("--basis", type=str,default='lensmode',help='Base tag.')
    parser.add_argument("--tags",     type=str,  default=None,help="Comma separated list of tags.")
    args = parser.parse_args()

    gal = '80'
    out_root = utils.out_root
    outname = args.out_name
    mask_fname_func = lambda tag: f'{out_root}/{outname}_{tag}_mask{gal}.fits'
    base_tag = args.basetag # We will extract on to this geometry and use its mask for the final mask
    mask = enmap.read_map(mask_fname_func(base_tag))
    shape,wcs = mask.shape,mask.wcs
    lmax = 4000
    out_beam_fwhm = args.fwhm

    Nsims = 4
    bin_edges = np.arange(100,lmax,40)
    binner = stats.bin1D(bin_edges)

    r = 0.
    for simid in range(Nsims):
        print(simid)
        alm = hp.read_alm(utils.cmb_sim_fname(simid),hdu=1)
        alm = cs.almxfl(alm,lambda x: maps.gauss_beam(x,out_beam_fwhm))
        imap = cs.alm2map(alm,enmap.empty(shape,wcs,dtype=np.float32))
        imap[mask<=0] = 0
        ialm = cs.map2alm(imap,lmax=lmax)
        omap = enmap.read_map(f'{out_root}/sim_{simid}_{outname}_coadd_covsmooth_{args.cov_smooth_factor}.fits')
        dalm = cs.map2alm(omap,lmax=lmax)

        xcls = cs.alm2cl(dalm,ialm)
        icls = cs.alm2cl(ialm,ialm)
        ls = np.arange(xcls.size)
        cents,bxcls = binner.bin(ls,xcls)
        cents,bicls = binner.bin(ls,icls)

        rcls = (bxcls-bicls)/bicls
        r = r + rcls

    r = rcls/Nsims

    pl = io.Plotter('rCl')
    pl.add(cents,r,marker='o')
    pl.hline(y=0)
    pl.done('rcls.png')
