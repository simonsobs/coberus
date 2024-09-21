from pixell import enmap,curvedsky as cs, wavelets as wv,uharm,multimap
import numpy as np
import utils
from orphics import io,maps
from coberus import Coadder, coadd
from dask.distributed import Client

"""

To do
- Beam factors in responses
- Covariance smoothing factors
- Data model

"""

if __name__ == '__main__':
    out_root = utils.out_root
    outname = 'test'
    gal = '80'
    lpeaks = [0.,500.,800.,1000.,2000.,3000.,4000.]

    # Load arrays
    lmaps = []
    masks = []

    base_tag = 'night_pa5_f090' # We will extract on to this geometry
    tags = [base_tag,'143','daydeep_pa5_f150']
    fwhm = [2.2,7.0,1.4]
    # tags = [base_tag,'143']
    nmaps = len(tags)
    for i,tag in enumerate(tags):
        imap = enmap.read_map(f'{out_root}/{outname}_{tag}_map.fits')
        mask = enmap.read_map(f'{out_root}/{outname}_{tag}_mask{gal}.fits')
        shape,wcs = imap.shape,imap.wcs
        if tag!=base_tag:
            imap = enmap.extract(imap,shape,wcs)
            mask = enmap.extract(mask,shape,wcs)
        lmaps.append(imap.copy())
        masks.append(mask.copy())
        print(f"Loading {tag}.")

    shape,wcs = lmaps[0].shape, lmaps[0].wcs
    print(shape,wcs)

    uht  = uharm.UHT(shape, wcs)
    basis = wv.CosineNeedlet(lpeaks = lpeaks)
    nwaves = basis.n
    wt = wv.WaveletTransform(uht, basis = basis)

    def plot(imap,tag,ind,mtype='map',**kwargs):
        io.hplot(imap,f'{out_root}/wavelet_{mtype}_{tag}_scale_{ind}',mask=0,**kwargs)

    def smooth(map,factor):
        return enmap.upgrade(enmap.downgrade(map, factor, inclusive=True),factor,inclusive=True,oshape=map.shape)


    fmasks = []
    fmaps = []
    fcovs = []
    for j in range(nwaves):
        fmasks.append([''] * nmaps)
        fmaps.append([''] * nmaps)
        fcovs.append([[''] * nmaps for i in range(nmaps)])

    # Loop through ACT Array 1, 2, 3
    for i,(omap,mask) in enumerate(zip(lmaps,masks)):
        omap[mask==0] = 0
        print("Wavelet transform...")
        ells = np.arange(6000)
        out_beam = maps.gauss_beam(ells, 1.6)
        in_beam = maps.gauss_beam(ells, fwhm[i])
        beam_ratio = out_beam / in_beam
        wavecs = wt.map2wave(omap,fl=beam_ratio)
        print(len(wavecs.maps))
        if i==0:
            owave = wavecs*0.

        for j,wmap in enumerate(wavecs.maps):
            #plot(wmap,tags[i],j) # these are wavelet coefficient maps
            print("Projecting mask and writing wavelet map...")
            omask = enmap.project(mask,wmap.shape,wmap.wcs,order=0)
            # plot(omask,tags[i],j,mtype='mask',colorbar=True) # these are wavelet coefficient maps
            mfname = f'{out_root}/wavelet_mask_{tags[i]}_scale_{j}.fits'
            fmasks[j][i] = mfname
            enmap.write_map(mfname,omask)
            wfname = f'{out_root}/wavelet_map_{tags[i]}_scale_{j}.fits'
            fmaps[j][i] = wfname
            enmap.write_map(wfname,wmap)



    print("Building covariance")
    for i in range(len(tags)):
        for j in range(i,len(tags)):
            for k in range(nwaves):
                print("Smoothing..")
                wmap1 = enmap.read_map(f'{out_root}/wavelet_map_{tags[i]}_scale_{k}.fits')
                wmap2 = enmap.read_map(f'{out_root}/wavelet_map_{tags[j]}_scale_{k}.fits')

                cov = smooth(wmap1*wmap2,8) # this factor needs to be adjusted
                # if (k<2 or k>4)  and ('night' in tags[i]):
                #     plot(cov,f'{tags[i]}_{tags[j]}',k,mtype='cov')
                fcovname = f'{out_root}/wavelet_cov_scale_{k}_{tags[i]}_{tags[j]}.fits'
                fcovs[k][i][j] = fcovname
                fcovs[k][j][i] = fcovname
                enmap.write_map(fcovname,cov)



    for j in range(nwaves):
        print(f"Coadding scale {j}...")
        lmaps = fmaps[j]
        masks = fmasks[j]
        covs = fcovs[j]
        responses = [1.0] * nmaps

        coadder = Coadder(
            maps=lmaps,
            masks=masks,
            covariance_maps=covs,
            responses=responses
        )

        with Client() as client:

            # Result is a dask array
            result = coadd(client, coadder)

            # This is now a numpy array
            arr = result.compute()

            shape,wcs = enmap.read_map_geometry(lmaps[0])
            owave.maps[j] = enmap.enmap(arr.copy(),wcs)

    print("wave2map")
    coadd_map = wt.wave2map(owave)
    print(coadd_map.shape, coadd_map.wcs)
    plot(coadd_map,"all",0,mtype='coadd')
