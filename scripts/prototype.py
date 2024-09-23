from pixell import enmap,curvedsky as cs, wavelets as wv,uharm,multimap,utils as u
import numpy as np
import utils
import os,sys
from orphics import io,maps
from coberus import Coadder, coadd
from dask.distributed import Client

"""

To do
- Exclude arrays depending on wavelet scale
- Covariance smoothing factors
- Data model
- Beam-bandpass factors in responses

Properties of a tag:
- scales to skip
- beam


"""



if __name__ == '__main__':
    out_root = utils.out_root
    outname = 'test'
    gal = '80'
    lpeaks = [0.,100.,500.,800.,1000.,2000.,3000.,4000.]
    cutbox = [[4.-2,4.-9],[-4.-2,-4.-9]]

    # Load arrays
    lmaps = []
    masks = []

    base_tag = 'night_pa5_f090' # We will extract on to this geometry and use its mask for the final mask
    tags = [base_tag,'143','daydeep_pa5_f150']

    # Wavelet scales to skip for each tag
    skips = []
    skips.append([0,1])
    skips.append([6,7])
    skips.append([0,1])
    
    fwhm = [2.2,7.0,1.4]

    # Load and extract maps on to base tag geometry
    for i,tag in enumerate(tags):
        imap = enmap.read_map(f'{out_root}/{outname}_{tag}_map.fits')
        mask = enmap.read_map(f'{out_root}/{outname}_{tag}_mask{gal}.fits')
        shape,wcs = imap.shape,imap.wcs
        if tag!=base_tag:
            imap = enmap.extract(imap,oshape,owcs)
            mask = enmap.extract(mask,oshape,owcs)
        else:
            base_mask = mask
            oshape,owcs = imap.shape, imap.wcs
        lmaps.append(imap.copy())
        masks.append(mask.copy())
        print(f"Loading {tag}.")

    shape,wcs = lmaps[0].shape, lmaps[0].wcs

    # Initialize Wavelets
    uht  = uharm.UHT(shape, wcs)
    basis = wv.CosineNeedlet(lpeaks = lpeaks)
    nwaves = basis.n
    wt = wv.WaveletTransform(uht, basis = basis)

    # Quick plots
    def plot(imap,tag,ind,mtype='map',**kwargs):
        io.hplot(imap,f'{out_root}/wavelet_{mtype}_{tag}_scale_{ind}',mask=0,**kwargs)

    # Function to smooth covariance maps with block downgrading and projection
    # back to original geometry
    def smooth(map,factor):
        downed = enmap.downgrade(map, factor, inclusive=True,op=np.nanmean)
        downed[np.isnan(downed)] = 0
        omap = enmap.upgrade(downed,factor,inclusive=True,oshape=map.shape)
        return omap

    # Helper for dictionaries
    def update(d,key,item):
        try:
            d[key]
        except KeyError:
            d[key] = []
        d[key].append(item)

    # These will hold file names for maps, masks and covariance maps
    # for use by the Coberus coadder
    fmasks = {}
    fmaps = {}
    fcovs = {}

    # Loop through arrays
    for i,(omap,mask) in enumerate(zip(lmaps,masks)):
        omap[mask==0] = 0
        print("Wavelet transform...")
        # Reconvolve to common beam
        ells = np.arange(6000)
        out_beam = maps.gauss_beam(ells, 1.6)
        in_beam = maps.gauss_beam(ells, fwhm[i])
        beam_ratio = out_beam / in_beam
        wavecs = wt.map2wave(omap,fl=beam_ratio,skip_coeffs=skips[i])
        plot(omap,tags[i],0,mtype='input_map',colorbar=True,downgrade=2,grid=True,ticks=10) # these are input maps
        smap = omap.submap(np.asarray(cutbox)*u.degree)
        plot(smap,tags[i],0,mtype='submap',colorbar=True,grid=True,ticks=0.5) # these are input maps
        
        if i==0:
            # Save multimap template for final coadded map
            owave = wavecs*0.

        # Loop through wavelet scales
        for j,wmap in enumerate(wavecs.maps):
            if (j in skips[i]): continue
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
            print("Map MB: ",wmap.nbytes/1024/1024.)
            enmap.write_map(wfname,wmap)


    print("Building covariance")
    for k in range(nwaves):
        fcovs[k] = [[''] * len(fmaps[k]) for h in range(len(fmaps[k]))]
        # Tags to be included in wavelet scale
        itags = []
        for i,tag in enumerate(tags):
            if k in skips[i]: continue
            itags.append(tag)

        for i in range(len(itags)):
            for j in range(i,len(itags)):
                print("Smoothing..")
                wmap1 = enmap.read_map(f'{out_root}/wavelet_map_{itags[i]}_scale_{k}.fits')
                wmap2 = enmap.read_map(f'{out_root}/wavelet_map_{itags[j]}_scale_{k}.fits')

                cov = smooth(wmap1*wmap2,8) # this factor needs to be adjusted
                if (k<2 or k>4)  and ('night' in itags[i]):
                    plot(cov,f'{itags[i]}_{itags[j]}',k,mtype='cov')
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
        responses = [1.0] * len(fmaps[j])

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

            shape,wcs = enmap.read_map_geometry(lmaps[0])
            owave.maps[j] = enmap.enmap(arr.copy(),wcs)

    print("wave2map")
    coadd_map = wt.wave2map(owave)
    coadd_map[base_mask==0] = 0
    print(coadd_map.shape, coadd_map.wcs)
    plot(coadd_map,"all",0,mtype='coadd',colorbar=True,downgrade=2,grid=True,ticks=10)
    smap = coadd_map.submap(np.asarray(cutbox)*u.degree)
    plot(smap,"all",0,mtype='coadd_submap',colorbar=True,grid=True,ticks=0.5) # these are input maps
