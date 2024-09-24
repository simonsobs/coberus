from pixell import enmap,curvedsky as cs, wavelets as wv,uharm,multimap,utils as u
import numpy as np
import utils
import os,sys
from orphics import io,maps
from coberus import Coadder, coadd
from coberus import pipeline
from dask.distributed import Client

# Quick plots
# def plot(imap,tag,ind,mtype='map',**kwargs):
#     io.hplot(imap,f'{out_root}/wavelet_{mtype}_{tag}_scale_{ind}',mask=0,**kwargs)

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


def needlet_coadd(map_fname_func, mask_fname_func, tags, base_tag,
          lpeaks, lmins, lmaxs, response_func, beam_func,
          out_beam_fwhm, out_root):

    """
    Generic function for coadding maps using an empirical
    covariance determined from smoothed products of maps in a
    needlet basis. Each input map is identified
    by a string called 'tag'. The properties of these maps are specified
    through functions of the tag name.

    Parameters
    ----------

    map_fname_func : func
        Accepts the tag name and returns a path to the input map

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

    Returns
    -------

    coadd_map : ndmap
       The final coadded map.
    
    """

    lmax = max(lpeaks)
    cutbox = [[4.-2,4.-9],[-4.-2,-4.-9]]

    shape,wcs = enmap.read_map_geometry(map_fname_func(base_tag))

    # Initialize Wavelets
    uht  = uharm.UHT(shape, wcs)
    basis = wv.CosineNeedlet(lpeaks = lpeaks)
    scales = get_scales(basis,tags,lmins,lmaxs)
    nwaves = basis.n
    wt = wv.WaveletTransform(uht, basis = basis)

    # These will hold file names for maps, masks and covariance maps
    # for use by the Coberus coadder
    fmasks = {}
    fmaps = {}
    fcovs = {}

    # Loop through arrays
    for i,tag in enumerate(tags):
        omap = enmap.read_map(map_fname_func(tag))
        mask = enmap.read_map(mask_fname_func(tag))
        if tag!=base_tag:
            omap = enmap.extract(omap,shape,wcs)
            mask = enmap.extract(mask,shape,wcs)
        else:
            base_mask = mask

        omap[mask==0] = 0
        print("Wavelet transform...")
        # Reconvolve to common beam
        ells = np.arange(lmax)
        out_beam = maps.gauss_beam(ells, out_beam_fwhm)
        in_beam = beam_func(tag,ells)
        beam_ratio = out_beam / in_beam
        wavecs = wt.map2wave(omap,fl=beam_ratio,scales=scales[tags[i]],fill_value=np.nan)
        # plot(omap,tags[i],0,mtype='input_map',colorbar=True,grid=True,ticks=10) # these are input maps
        smap = omap.submap(np.asarray(cutbox)*u.degree)
        # plot(smap,tags[i],0,mtype='submap',colorbar=True,grid=True,ticks=0.5) # these are input maps

        if i==0:
            # Save multimap template for final coadded map
            owave = wavecs*0.

        # Loop through wavelet scales
        for j,wmap in enumerate(wavecs.maps):
            if (j not in scales[tags[i]]): continue
            #plot(wmap,tags[i],j) # these are wavelet coefficient maps
            print("Projecting mask and writing wavelet map...")
            # Project masks on to wavelet map geometries
            omask = enmap.project(mask,wmap.shape,wmap.wcs,order=0)
            # plot(omask,tags[i],j,mtype='mask',colorbar=True) # these are wavelet coefficient maps

            mfname = f'{out_root}/wavelet_mask_{tags[i]}_scale_{j}.fits'
            update(fmasks, j, mfname)
            enmap.write_map(mfname,omask)

            wfname = f'{out_root}/wavelet_map_{tags[i]}_scale_{j}.fits'
            update(fmaps, j, wfname)
            # print("Map MB: ",wmap.nbytes/1024/1024.)
            enmap.write_map(wfname,wmap)


    print("Building covariance")
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
                print("Smoothing..")
                wmap1 = enmap.read_map(f'{out_root}/wavelet_map_{itags[i]}_scale_{k}.fits')
                wmap2 = enmap.read_map(f'{out_root}/wavelet_map_{itags[j]}_scale_{k}.fits')

                cov = maps.block_smooth(wmap1*wmap2,8) # this factor needs to be adjusted
                # if (k<2 or k>4)  and ('night' in itags[i]):
                #     plot(cov,f'{itags[i]}_{itags[j]}',k,mtype='cov')
                fcovname = f'{out_root}/wavelet_cov_scale_{k}_{itags[i]}_{itags[j]}.fits'
                fcovs[k][i][j] = fcovname
                fcovs[k][j][i] = fcovname
                enmap.write_map(fcovname,cov)


    # This part uses Coberus to do distributed Dask
    # pixel-space coadding of the maps for each
    # wavelet scale
    for j in range(nwaves):
        print(f"Coadding scale {j}...")
        lmaps = fmaps[j]
        masks = fmasks[j]
        covs = fcovs[j]
        responses = [response_func(tag) for tag in included_tags[j]]

        coadder = Coadder(
            maps=lmaps,
            masks=masks,
            covariance_maps=covs,
            responses=responses
        )

        # n_workers=int(os.environ['OMP_NUM_THREADS'])
        with Client() as client:
            # Result is a dask array
            result = coadd(client, coadder)
            # This is now a numpy array
            arr = result.compute()
            owave.maps[j] = enmap.enmap(arr.copy(),wcs)

    print("wave2map")
    coadd_map = wt.wave2map(owave)
    coadd_map[base_mask==0] = 0
    return coadd_map
    
