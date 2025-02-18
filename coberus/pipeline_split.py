from pixell import enmap,curvedsky as cs, wavelets as wv,uharm,multimap,utils as u
import numpy as np
import os,sys
from orphics import io,maps,cosmology
from coberus import Coadder, coadd
from coberus import pipeline
from dask.distributed import Client
import time, psutil

# temporary workaround for pip installing
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../scripts/")
import utils

# Helper for dictionaries
def update(d,key,item):
    try:
        d[key]
    except KeyError:
        d[key] = []
    d[key].append(item)

def needlet_coadd_extra(dmap_fname_func, smap_fname_func, nmap_fname_func,
                        mask_fname_func, tags, base_tag, lpeaks, lmins,
                        lmaxs, response_func,
                        beam_func, out_beam_fwhm, out_root,
                        cov_smooth_factor=64,
                        map_postprocess_func=None, mask_postprocess_func=None,
                        n_workers=None,
                        io_suffix='', delete_intermediate=False):

    """
    Modification to generic coadding function (coberus/pipeline.py)
    using an empirical covariance determined from smoothed products
    of maps in a needlet basis. Each input map is identified
    by a string called 'tag'. The properties of these maps are specified
    through functions of the tag name.

    This function does the following:
    
    Take a signal map (from smap_fname_func), add it to a noise map
    (from nmap_fname_func) to build a combined data (signal+noise) map.
    However, the empirical covariance + weights estimated from the data map
    is then used to coadd the signal-only maps as well as the noise-only maps.
    All three coadds are then returned.

    Parameters
    ----------

    dmap_fname_func : func
        Accepts the tag name and returns a path to the input signal+noise map

    smap_fname_func : func
        Accepts the tag name and returns a path to the input signal map
    
    nmap_fname_func : func
        Accepts the tag name and returns a path to the input noise map

    mask_fname_func : func
        Accepts the tag name and returns a path to the mask

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
        Root path for outputs

    cov_smooth_factor : optional,int
        Factor by which to block downgrade the covariance maps
    
    map_postprocess_func : optional,func
        A function to apply to each loaded map
    
    mask_postprocess_func : optional,func
        A function to apply to each loaded mask
    
    n_workers : optional,int
        Number of workers for distributed Dask tasks
    
    io_suffix : optional,str
        Suffix for intermediate outputs (use a different one for each simulation)
    
    delete_intermediate : optional,bool
        Whether to delete intermediate outputs

    
    Returns
    -------

    coadd_map : ndmap
       The final coadded map.
    
    """
    start_time = time.time()
    
    lmax = max(lpeaks)
    ells = np.arange(lmax)
    shape,wcs = enmap.read_map_geometry(map_fname_func(base_tag))

    # Initialize Wavelets
    uht  = uharm.UHT(shape, wcs)
    basis = wv.CosineNeedlet(lpeaks = lpeaks)
    scales = pipeline.get_scales(basis,tags,lmins,lmaxs)
    nwaves = basis.n
    wt = wv.WaveletTransform(uht, basis = basis)

    def _get_wave(fname_func, itag, imask):
        gmap = enmap.read_map(fname_func(itag))

        if map_postprocess_func is not None:
            gmap = map_postprocess_func(gmap)
        if itag != base_tag:
            gmap = enmap.extract(gmap,shape,wcs)

        gmap[imask==0] = 0
        # Reconvolve to common beam
        out_beam = maps.gauss_beam(ells, out_beam_fwhm)
        in_beam = beam_func(itag,ells)
        beam_ratio = out_beam / in_beam
        wavecs = wt.map2wave(gmap,fl=beam_ratio,scales=scales[itag],fill_value=np.nan)
        return wavecs

    # These will hold file names for maps, masks and covariance maps
    # for use by the Coberus coadder
    fmasks = {}
    fmaps = {}
    nfmaps = {}
    sfmaps = {}
    fcovs = {}
    filenames = []

    print(f"Free memory: {pipeline.free_mem()}")
    totgibytes = 0.

    # Loop through arrays
    for i,tag in enumerate(tags):
        mask = enmap.read_map(mask_fname_func(tag))
        if mask_postprocess_func is not None:
            mask = mask_postprocess_func(mask)
        if tag!=base_tag:
            mask = enmap.extract(mask,shape,wcs)
        else:
            base_mask = mask

        swavecs = _get_wave(smap_fname_func,tag,mask)
        nwavecs = _get_wave(nmap_fname_func,tag,mask)
        wavecs = _get_wave(dmap_fname_func,tag,mask)

        if i==0:
            # Save multimap template for final coadded map
            owave = wavecs*0.

        # Loop through wavelet scales
        for j,wmap in enumerate(wavecs.maps):
            if (j not in scales[tags[i]]): continue
            print("Projecting mask and writing wavelet map...")
            # Project masks on to wavelet map geometries
            omask = enmap.project(mask,wmap.shape,wmap.wcs,order=0)

            mfname = f'{out_root}/wavelet_mask_{tags[i]}_scale_{j}{io_suffix}.fits'
            filenames.append(mfname)
            update(fmasks, j, mfname)
            enmap.write_map(mfname,omask)

            wfname = f'{out_root}/wavelet_map_{tags[i]}_scale_{j}{io_suffix}.fits'
            filenames.append(wfname)
            update(fmaps, j, wfname)
            enmap.write_map(wfname,wmap)

            totgibytes = totgibytes + (wmap.nbytes/1024/1024./1024.*2.)

            # noise only
            nwfname = f'{out_root}/wavelet_nmap_{tags[i]}_scale_{j}{io_suffix}.fits'
            filenames.append(nwfname)
            update(nfmaps, j, nwfname)
            enmap.write_map(nwfname,nwavecs.maps[j])
            totgibytes = totgibytes + (wmap.nbytes/1024/1024./1024.)

            # signal only
            swfname = f'{out_root}/wavelet_smap_{tags[i]}_scale_{j}{io_suffix}.fits'
            filenames.append(swfname)
            update(sfmaps, j, swfname)
            enmap.write_map(swfname,swavecs.maps[j])
            totgibytes = totgibytes + (wmap.nbytes/1024/1024./1024.)


    print(wmap.dtype)
    print(f"Total disk: {totgibytes:.1f} GiB")
    print(f"Free memory: {pipeline.free_mem()}")
    print("Building covariance...")
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
            for j in range(i,len(itags)):
                print(f"Smoothing {k}: {i} x {j}..")
                wmap1 = enmap.read_map(f'{out_root}/wavelet_map_{itags[i]}_scale_{k}{io_suffix}.fits')
                wmap2 = enmap.read_map(f'{out_root}/wavelet_map_{itags[j]}_scale_{k}{io_suffix}.fits')
                
                if cov_smooth_factor!=1:
                    cov = maps.block_smooth(wmap1*wmap2,cov_smooth_factor,slow=False) # this factor needs to be adjusted
                else:
                    cov = wmap1*wmap2
                    
                fcovname = f'{out_root}/wavelet_cov_scale_{k}_{itags[i]}_{itags[j]}{io_suffix}.fits'
                filenames.append(fcovname)
                fcovs[k][i][j] = fcovname
                fcovs[k][j][i] = fcovname
                enmap.write_map(fcovname,cov)
                totgibytes = totgibytes + (cov.nbytes/1024/1024./1024.)

    print(f"Total disk: {totgibytes:.1f} GiB")
    print(f"Free memory: {pipeline.free_mem()}")

    outmaptypes = ['coadd', 'noise_coadd', 'signal_coadd']
    outmaps = {}

    for outmaptype in outmaptypes:
        # This part uses Coberus to do distributed Dask
        # pixel-space coadding of the maps for each
        # wavelet scale
        for j in range(nwaves):
            print(f"Coadding {outmaptype} scale {j}...")
            if outmaptype == 'coadd':
                lmaps = fmaps[j] 
            elif outmaptype == 'noise_coadd':
                lmaps = nfmaps[j]
            elif outmaptype == 'signal_coadd':
                lmaps = sfmaps[j]

            masks = fmasks[j]
            covs = fcovs[j]
            responses = [response_func(tag) for tag in included_tags[j]]

            coadder = Coadder(
                maps=lmaps,
                masks=masks,
                covariance_maps=covs,
                responses=responses
            )

            with Client(n_workers=n_workers) as client:
                print("Number of workers: ", len(client.scheduler_info()['workers']))
                # Result is a dask array
                result = coadd(client, coadder)
                # This is now a numpy array
                arr = result.compute()
                owave.maps[j] = enmap.enmap(arr.copy(),wcs)

        coadd_map = wt.wave2map(owave)
        coadd_map[base_mask==0] = 0
        outmaps[outmaptype] = coadd_map.copy()

    print(f"Free memory: {pipeline.free_mem()}")
    if delete_intermediate:
        for filename in filenames:
            os.remove(filename)
    
    elapsed_time = time.time() - start_time
    print(f"Done in {elapsed_time/60.:.2f} minutes.")
    return outmaps
    
