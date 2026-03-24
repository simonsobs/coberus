from pixell import enmap,curvedsky as cs, wavelets as wv,uharm,multimap,utils as u
import numpy as np
import os,sys
from orphics import io,maps
from coberus import Coadder, coadd
from coberus import pipeline
from dask.distributed import Client
from contextlib import nullcontext
import healpy as hp

import time, psutil

def free_mem():
    return f"{psutil.virtual_memory()[1]/1024/1024/1024:.1f} GiB"


def get_scales(basis,tags,ellmins,ellmaxs):
    """
    Given a pixell.wavelets.basis wavelet basis
    object and minimum and maximum multipoles
    for datasets tagged by tags, return a dictionary
    mapping the tag names to a list of wavelet
    coefficient indices that we should calculate
    wavelet maps for.
    """
    ntags = len(ellmins)
    scales = {}
    if len(ellmaxs)!=ntags: raise ValueError
    for tag,ellmin,ellmax in zip(tags,ellmins,ellmaxs):
        scales[tag] = []
        for i in range(basis.n):
            wlmin = basis.lmins[i]
            wlmax = basis.lmaxs[i]
            if ellmin is None: ellmin = 0
            if ellmax is None: ellmax = np.inf
            if (ellmin>wlmin): continue
            if (ellmax<wlmax): continue
            scales[tag].append(i)
    return scales


# Helper for dictionaries
def update(d,key,item):
    try:
        d[key]
    except KeyError:
        d[key] = []
    d[key].append(item)


def compute_tophat_beam(w_rad, lmax, w_rad_in=None, n_theta=20000):
    """
    Computes harmonic space beam for top hat filter used in 2307.01258.

    Parameters
    ----------

    w_rad: float
        Smoothing scale in radians
    
    lmax: int
        Maximum multipole

    w_rad_in: float, optional
        Optional float that sets a second filter scale and excludes modes below
        this filter scale. Used for choosing an annulus in angular space

    n_theta: float, optional
        Number of theta bins to use when computing the angular smoothing
        function.
    
    Returns
    -------

    filt_harm: float array
        Harmonic-space filter beam for multipoles up to lmax. 

    """
    
    theta_max = 10 * w_rad
    theta = np.linspace(0, theta_max, n_theta)
    filt_ang = 1 / ((1+(theta /w_rad ))**6)
    
    if w_rad_in is not None:
        filt_ang -= 1 / ((1+(theta/w_rad_in))**6)
    
    filt_harm = hp.sphtfunc.beam2bl(filt_ang, theta, int(lmax))
    filt_harm /= filt_harm[0]
    
    return filt_harm


def needlet_coadd(map_fname_func, mask_fname_func, tags, base_tag,
                  lpeaks, lmins, lmaxs, response_func, 
                  beam_func, out_beam_fwhm, out_root, deproj_response_funcs = None,
                  cov_smooth_type='block', cov_smooth_factor=64, ilc_bias_tol=0.01, 
                  fft_smooth=False, smooth_mean_cov=True, cov_smooth_scales=None, 
                  use_annulus=False, annulus_fwhm_ratio=0.5, map_postprocess_func=None, 
                  mask_postprocess_func=None, client=None, n_workers=None,io_suffix='', delete_intermediate=False,
                  nmap_labels=[], nmap_label_fname_func=None, apply_mask=False):

    """
    Generic function for coadding maps using an empirical
    covariance determined from smoothed products of maps in a
    needlet basis. Each input map is identified
    by a string called 'tag'. The properties of these maps are specified
    through functions of the tag name.

    However, if nmap_label_fname_func are provided, covariances +
    weights estimated from the data map are then used to coadd the
    "nmap"s and they are returned as well, formatted as:
    output[f'{nmap_label}_coadd'] for [nmap_label,...] in nmap_labels.

    Parameters
    ----------

    map_fname_func : func
        Accepts the tag name and returns a path to the input map.
        We recommend you mask and apodize these maps.

    mask_fname_func : func
        Accepts the tag name and returns a path to *binary* masks
        for each tag. These binary masks denote which regions of
        each map to include in the final coadd. We recommend these
        masks exclude regions where the apodization of the input
        maps drops below some threshold, say 0.99.

    tags : list
        Ordered list of tags to coadd

    base_tag : str
        Name of the tag whose geometry all other tags are extracted
        to and whose mask is used as the final mask.

    lpeaks : list
        List of peak multipoles that define a cosine needlet basis

    lmins : list
        List of minimum multipoles beyond which a tag is not used
        in a wavelet scale
    
    lmaxs : list
        List of maximum multipoles beyond which a tag is not used
        in a wavelet scale
    
    response_func : func
        Accepts the tag name and returns the map response value.
        Use lambda x: 1 for the CMB solution.

    beam_func : func
        Accepts the tag name and the multipole as arguments and
        returns the value of the beam.

    out_beam_fwhm : float
        FWHM in arcminutes for the beam of the final map

    out_root : str
        Root path for intermediate outputs. We highly recommend
        using a RAMdisk (/dev/shm/) if you have enough memory.
        This will significantly speed up the code. Remember to
        append a unique code for each job if you have multiple
        jobs sharing memory on a node.
        e.g. out_root = '/dev/shm/sim_1_'

    deproj_response_funcs : list of funcs
        List of response functions to deproject. Each function should accepts
        the tag name and return the map response value.
    
    cov_smooth_type : optional,str
        Type of smoothing to use when constructing the covariance. Current
        options are 'block', 'gaussian', and 'tophat'. Default value is 'block'.

    cov_smooth_factor : optional,int
        Factor by which to block downgrade the covariance maps if using block
        smoothing.

    ilc_bias_tol : optional, float
        Bias tolerance used to determine FWHM for Gaussian smoothing
        of covariance if using Gaussian smoothing. Based on Eq. 43 of 
        2307.01043.

    fft_smooth : optional, boolean
        If true, use FFT's instead of SHT's to smooth the map. Default 
        option is to use SHT's.

    smooth_mean_cov: optional, boolean
        Only used for gaussian or top-hat smoothing.
        If True, computes the covariance from <(A-A_smooth)(B-B_smooth)>, as
        in pyilc. Otherwise, computes the covariance from <AB>. Default is True.
        
    map_postprocess_func : optional,func
        A function to apply to each loaded map
    
    mask_postprocess_func : optional,func
        A function to apply to each loaded mask
        
    n_workers : optional,int
        Number of workers for distributed Dask tasks
    
    delete_intermediate : optional,bool
        Whether to delete intermediate outputs

    nmap_labels : optional,list[str]
        List of possible optional maps' labels

    nmap_label_fname_func : optional,func | (nmap_label, fname) -> nmap
        Optional maps not used for covariance, but coadded with
        the same weights. Accepts the optional map's label and filename
        and returns a path

    apply_mask : optional, boolean
        If true, zero out regions of the input maps based on their masks.
        This will make the maps masked sharply before wavelet transforms,
        and is hence not recommended.
        

    Returns
    -------

    coadd_map : ndmap
       The final coadded map.
    
    """
    start_time = time.time()
    lmax = max(lpeaks) # Cosine needlets have zero support beyond lpeak
    ells = np.arange(lmax)
    shape,wcs = enmap.read_map_geometry(map_fname_func(base_tag))
    n_deproj = len(deproj_response_funcs) if (deproj_response_funcs is not None) else 0
    n_map    = len(tags) # Number of maps

    # Compute fsky from base mask. This is used to determine covariance smoothing scales
    base_mask = enmap.read_map(mask_fname_func(base_tag))
    fsky = (np.sum(base_mask**2)/np.prod(base_mask.shape))*base_mask.area()/(4*np.pi)
    print("fsky={:.2f}".format(fsky))
    

    # Initialize Wavelets
    uht  = uharm.UHT(shape, wcs,  mode="curved") 
    basis = wv.CosineNeedlet(lpeaks = lpeaks)
    scales = get_scales(basis,tags,lmins,lmaxs)
    nwaves = basis.n
    wt = wv.WaveletTransform(uht, basis = basis)

    # Compute number of tags used at each needlet scale
    n_tag_per_scale = np.sum([np.bincount(scales[tag], minlength=basis.n) 
                              for tag in tags], axis=0)
    
    # if using optional additional maps to coadd
    do_nmaps = len(nmap_labels) > 0 and (nmap_label_fname_func is not None)

    def _get_wave(fname_func, itag, imask):
        gmap = enmap.read_map(fname_func(itag))

        if map_postprocess_func is not None:
            gmap = map_postprocess_func(gmap)
        if itag != base_tag:
            gmap = enmap.extract(gmap,shape,wcs)

        if apply_mask: gmap[imask==0] = 0
        # Reconvolve to common beam
        out_beam = maps.gauss_beam(ells, out_beam_fwhm)
        in_beam = beam_func(itag,ells)
        if in_beam.ndim!=1: raise ValueError
        beam_ratio = out_beam / in_beam
        wavecs = wt.map2wave(gmap,fl=beam_ratio,scales=scales[itag],fill_value=np.nan)
        return wavecs

    # These will hold file names for maps, masks and covariance maps
    # for use by the Coberus coadder
    fmasks = {}
    fmaps = {}
    fcovs = {}
    filenames = []
    # store additional maps if desired
    if do_nmaps: nfmaps = {label: {} for label in nmap_labels}

    print(f"Free memory: {free_mem()}")
    totgibytes = 0.

    # Covariance smoothing
    cov_smooth_types = ['block', 'gaussian', 'tophat']
    assert cov_smooth_type in cov_smooth_types, f'Unrecognized covariance smoothing type.  Available options are: {", ".join(cov_smooth_types)}'

    ############################################################################
    # Determine smoothing scales
    ############################################################################
    # If the user does not input a set of smoothing scales for the covariance
    # calculation, then we comptue them here. For gaussian smoothing, we compute
    # the scales from the ILC bias tolerance (see 2307.01043). For tophat 
    # filters, we use Eq. 13 of 2307.01258. 
    # 
    # In principle, these calculations should account for the fact that the 
    # number of maps that contribute to each pixel can vary over the sky. This 
    # would require introducing an anisotropic smoothing. Instead, we use the 
    # total number of maps considered in the ILC at a given needlet scale.
    # This is conservative -- it overestimates the smoothing scale needed for a 
    # given ILC bias tolerance in regions that are only covered by a subset of 
    # the input maps.
    
    if cov_smooth_scales is None:
        if cov_smooth_type == 'gaussian':
            print(f"Covariance smoothing scales not specified. Determining scales with ILC bias tolerance of {ilc_bias_tol}")
            n_modes_eff = np.asarray([np.sum((2*ells+1)*basis(i, ells)**2) 
                                    for i in range(basis.n)])*fsky
            n_freq_eff  = n_tag_per_scale 

            cov_smooth_scales = np.sqrt( 2 * abs(1+n_deproj-n_freq_eff) / (ilc_bias_tol*n_modes_eff) ) # Radians

            assert all(cov_smooth_scales < np.pi), "Not enough modes to satisfy ILC bias tolerance." 

        elif cov_smooth_type =='tophat':
            # Note that this differs from Nmodes used in Gaussian smoothing. Here, we approximate the
            # needlets as top-hats in harmonic space. The number of modes used in the Gaussian smoothing
            
            # Note that this implementation is slightly different than the one 
            # in 2307.01258 (https://github.com/ACTCollaboration/NILC/blob/main/NILC/ilc.py)
            # which uses: 
            # n_modes_eff = np.asarray([np.sum((2*ells+1)*np.where(basis(i, ells) !=0 , 1, 0)) 
            #                         for i in range(basis.n)]) 
            
            n_modes_eff = np.asarray([np.sum((2*ells+1)*basis(i, ells)**2) 
                                    for i in range(basis.n)])*fsky
            n_freq_eff  = n_tag_per_scale

            # Eq 13 of 2307.01258
            arccos_arg = 1. - 2. * (20 * n_freq_eff/n_modes_eff)
            arccos_arg[arccos_arg<-1] = -1
            cov_smooth_scales = 2*np.arccos(arccos_arg) 

    else:
        assert len(cov_smooth_scales)==nwaves, "Number of covariance smoothing scales does not match number of wavelets"
    
    start_time_wavelets = time.time()
    
    # Loop through arrays
    for i,tag in enumerate(tags):
        mask = enmap.read_map(mask_fname_func(tag))
        if mask_postprocess_func is not None:
            mask = mask_postprocess_func(mask)
        if tag!=base_tag:
            mask = enmap.extract(mask,shape,wcs)
        else:
            base_mask = mask

        wavecs = _get_wave(map_fname_func,tag,mask)
        nwavecs = {label: _get_wave(lambda fname: nmap_label_fname_func(label, fname),
                                    tag, mask) for label in nmap_labels}

        if i==0:
            # Save multimap template for final coadded map
            owave = wavecs*0.

        # Loop through wavelet scales
        for j,wmap in enumerate(wavecs.maps):
            if (j not in scales[tags[i]]): continue
            print("Projecting mask and writing wavelet map...")
            # Project masks on to wavelet map geometries
            omask = enmap.project(mask,wmap.shape,wmap.wcs,order=0)

            mfname = f'{out_root}wavelet_mask_{tags[i]}_scale_{j}.fits'
            filenames.append(mfname)
            update(fmasks, j, mfname)
            enmap.write_map(mfname,omask)

            wfname = f'{out_root}wavelet_map_{tags[i]}_scale_{j}.fits'
            filenames.append(wfname)
            update(fmaps, j, wfname)
            enmap.write_map(wfname,wmap)
            
            totgibytes = totgibytes + (wmap.nbytes/1024/1024./1024.*2.)

            # optional maps
            for label in nmap_labels:
                nwfname = f'{out_root}wavelet_{label}_{tags[i]}_scale_{j}.fits'
                filenames.append(nwfname)
                update(nfmaps[label], j, nwfname)
                enmap.write_map(nwfname,nwavecs[label].maps[j])
                totgibytes = totgibytes + (wmap.nbytes/1024/1024./1024.)

    elapsed_time = time.time() - start_time_wavelets
    print(f"Wavelets finished in {elapsed_time/60.:.2f} minutes.")
    print(f"Total disk: {totgibytes:.1f} GiB")
    print(f"Free memory: {free_mem()}")
    print("Building covariance...")
    start_time_covariance = time.time()
    included_tags = {}
    for k in range(nwaves):

        fcovs[k] = [[''] * len(fmaps[k]) for h in range(len(fmaps[k]))]
        # Tags to be included in wavelet scale
        itags = []
        for i,tag in enumerate(tags):
            if k not in scales[tag]: continue
            itags.append(tag)
        included_tags[k] = list(itags)

        for i in range(len(itags)):
            wmap1 = enmap.read_map(f'{out_root}wavelet_map_{itags[i]}_scale_{k}.fits')
            
            for j in range(i,len(itags)):


                if i==j: # Potentially minor speedup?
                    wmap2 = wmap1
                else:
                    wmap2 = enmap.read_map(f'{out_root}wavelet_map_{itags[j]}_scale_{k}.fits')

                if cov_smooth_type == 'block':
                    cov = maps.block_smooth(wmap1*wmap2,cov_smooth_factor,slow=False) 
                
                elif cov_smooth_type == 'gaussian':
                    # Applies Gaussian smoothing procedure from 2307.01043. 
                    sigma_rad = cov_smooth_scales[k]
                    
                    if smooth_mean_cov:
                        if fft_smooth:
                            wmap1_smooth = enmap.smooth_gauss(wmap1, sigma_rad)

                            if i==j:
                                wmap2_smooth = wmap1_smooth
                            else:
                                wmap2_smooth = enmap.smooth_gauss(wmap2, sigma_rad)

                            cov = enmap.smooth_gauss((wmap1-wmap1_smooth)*(wmap2-wmap2_smooth), sigma_rad) 
                        
                        else:
                            gauss_beam = np.exp(-0.5 * ells * (ells + 1) * sigma_rad**2)
                            wmap1_smooth = cs.filter(wmap1, gauss_beam, lmax=lmax)

                            if i==j:
                                wmap2_smooth = wmap1_smooth
                            else:
                                wmap2_smooth = cs.filter(wmap2, gauss_beam, lmax=lmax)

                            cov = cs.filter((wmap1-wmap1_smooth)*(wmap2-wmap2_smooth), gauss_beam, lmax=lmax) 

                    else:
                        # Compute covariance without smoothing the maps. This
                        # uses only one SHT/FFT, but is less stable than the
                        # smooth_mean_cov_approach
                        if fft_smooth:
                            cov = enmap.smooth_gauss(wmap1*wmap2, sigma_rad) 
                        else:
                            cov = cs.filter((wmap1)*(wmap2), gauss_beam, lmax=lmax) 

                elif cov_smooth_type == 'tophat':
                    
                    # Compute beam for top-hat smoothing procedure from 2307.01258
                    w_rad = cov_smooth_scales[k]

                    if use_annulus:
                        w_rad_in = annulus_fwhm_ratio*w_rad
                    else:
                        w_rad_in = None

                    tophat_beam =  compute_tophat_beam(w_rad, lmax, w_rad_in=w_rad_in)

                    if smooth_mean_cov:
                        wmap1_smooth = cs.filter(wmap1, tophat_beam, lmax=lmax)

                        if i==j:
                            wmap2_smooth = wmap1_smooth
                        else:
                            wmap2_smooth = cs.filter(wmap2, tophat_beam, lmax=lmax)

                        cov = cs.filter((wmap1-wmap1_smooth)*(wmap2-wmap2_smooth), 
                                        tophat_beam, lmax=lmax) 

                    else:
                        cov = cs.filter((wmap1)*(wmap2), tophat_beam, lmax=lmax) # Eq. 11

                fcovname = f'{out_root}wavelet_cov_scale_{k}_{itags[i]}_{itags[j]}.fits'
                fcovs[k][i][j] = fcovname
                fcovs[k][j][i] = fcovname
                enmap.write_map(fcovname,cov)
                filenames.append(fcovname)
                totgibytes = totgibytes + (cov.nbytes/1024/1024./1024.)
                
    elapsed_time = time.time() - start_time_covariance
    print(f"Covariance finished in {elapsed_time/60.:.2f} minutes.")

    print(f"Total disk: {totgibytes:.1f} GiB")
    print(f"Free memory: {free_mem()}")

    outmaptypes = ['coadd'] + [label + '_coadd' for label in nmap_labels]
    outmaps = {}

    client_ctx = Client(n_workers=n_workers) if client is None else nullcontext(client)
    with client_ctx as client:
        for outmaptype in outmaptypes:
            # This part uses Coberus to do distributed Dask
            # pixel-space coadding of the maps for each
            # wavelet scale
            for j in range(nwaves):
                print(f"Coadding {outmaptype} scale {j}...")
                if outmaptype == 'coadd':
                    lmaps = fmaps[j]
                else:
                    lmaps = nfmaps[outmaptype[:-6]][j]

                masks = fmasks[j]
                covs = fcovs[j]
                responses = [response_func(tag) for tag in included_tags[j]]

                if n_deproj==0:
                    deproj_responses = []
                else:
                    deproj_responses = [[deproj_func_i(tag) for tag in included_tags[j]] for deproj_func_i in deproj_response_funcs] # N_deproj x N_freq 

                coadder = Coadder(
                    maps=lmaps,
                    masks=masks,
                    covariance_maps=covs,
                    responses=responses,
                    deproj_responses=deproj_responses
                )

                print("Number of workers: ", len(client.scheduler_info()['workers']))
                # Result is a dask array
                result = coadd(client, coadder)
                # This is now a numpy array
                arr = result.compute()
                owave.maps[j] = enmap.enmap(arr.copy(), owave.maps[j].wcs)

            coadd_map = wt.wave2map(owave)
            coadd_map[base_mask==0] = 0
            outmaps[outmaptype] = coadd_map.copy()

    
    print(f"Free memory: {free_mem()}")
    if delete_intermediate:
        for filename in filenames:
            os.remove(filename)
    elapsed_time = time.time() - start_time
    print(f"Done in {elapsed_time/60.:.2f} minutes.")
    return outmaps
