from pixell import enmap,curvedsky as cs, wavelets as wv,uharm
import numpy as np
import utils
from orphics import io

# We load downgraded inpainted maps

out_root = utils.out_root
outname = 'test'


# Load arrays
maps = []
masks = []

base_tag = 'night_pa5_f090' # We will extract on to this geometry
tags = [base_tag,'143','daydeep_pa5_f150']
for tag in tags:
    imap = enmap.read_map(f'{out_root}/{outname}_{tag}_map.fits')
    mask = enmap.read_map(f'{out_root}/{outname}_{tag}_mask80.fits')
    shape,wcs = imap.shape,imap.wcs
    if tag!=base_tag:
        imap = enmap.extract(imap,shape,wcs)
        mask = enmap.extract(mask,shape,wcs)
    maps.append(imap.copy())
    masks.append(mask.copy())
    print(f"Loading {tag}.")

shape,wcs = maps[0].shape, maps[0].wcs
print(shape,wcs)
    
uht  = uharm.UHT(shape, wcs)
basis = wv.CosineNeedlet(lpeaks = [0., 100.,500.,800.,1000.])
wt = wv.WaveletTransform(uht, basis = basis)

def plot(imap,tag,ind):
    io.hplot(imap,f'{out_root}/wavelet_map_{tag}_scale_{ind}',mask=0)

# Loop through ACT Array 1, 2, 3
for i,(omap,mask) in enumerate(zip(maps,masks)):
    omap[mask==0] = 0
    print("Wavelet transform...")
    wavecs = wt.map2wave(omap)

    for j,wmap in enumerate(wavecs.maps):
        plot(wmap,tags[i],j) # these are wavelet coefficient maps


    # Build covariance matrix by squaring the maps? We can add smoothing of the squares later

    # Use coberus to coadd them as shown in the README
