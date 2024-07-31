# Pseudo-code for the core needlet co-addition function

import numpy as np

# Core function to be written in C
def coadd_maps_pixel_domain(maps, cov_maps, masks, responses):
    """

    Arguments
    =========
    
    maps: float32 (nmaps, Ny, Nx)
    cov_maps: float32 (nmaps,nmaps, Ny, Nx)  # Could try passing compressed version with only nmaps*(nmaps+1)/2 maps
    masks: bool (nmaps, Ny, Nx)
    responses: float32 (nmaps,)

    Returns
    =======

    outmap: float32 (Ny,Nx)
    """

    Ny, Nx = maps.shape[-2:]
    outmap = np.zeros((Ny,Nx))
    # Loop over pixels
    for i in range(Ny):
        for j in range(Nx):
            # Get masks at this pixel
            mask = masks[:,i,j]

            # Select unmasked maps
            imaps = maps[mask,i,j]
            nout = imaps.size
            # There is probably a better way to do this in
            # numpy, but we're going to do this in C anyway bleh
            cov = cov_maps[:,:,i,j][:,mask][mask,:].reshape((nout,nout))
            a = responses[mask]
            
            # The linear algebra we want to do (right below Eq 3 in arxiv:1911.05717)
            Cinva = np.linalg.solve(cov,a)
            denom = np.dot(a,Cinva)

            Cinvd = np.linalg.solve(cov,imaps)
            numer = np.dot(a,Cinvd)

            outmap[i,j] = numer/denom
            
    return outmap

nmaps = 3
Ny = Nx = 32
maps = np.ones((nmaps,Ny,Nx))
masks = np.full((nmaps,Ny,Nx),True)
responses = np.ones((nmaps,))
# A trivial covmat that is identity at each pixel
cov_maps = np.eye(nmaps)[...,None,None] * np.ones((Ny,Nx))
omap = coadd_maps_pixel_domain(maps, cov_maps, masks, responses)
