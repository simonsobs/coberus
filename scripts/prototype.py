from pixell import enmap,curvedsky as cs, wavelets as wv,uharm

# Load an ACT mask
mask = enmap.read_map(...)
shape,wcs = mask.shape, mask.wcs

# Load arrays


uht  = uharm.UHT(shape, wcs)
basis = wv.CosineNeedlet(lpeaks = [100,500,800,1000])
wt = wv.WaveletTransform(uht, basis = basis)

# Loop through ACT Array 1, 2, 3
for omap in arrays:
    omap[mask==0] = 0
    wavecs = wt.map2wave(omap)

    for wmap in wavecs.maps:
        plot(wmap) # these are wavelet coefficient maps


    # Build covariance matrix by squaring the maps? We can add smoothing of the squares later

    # Use coberus to coadd them as shown in the README
