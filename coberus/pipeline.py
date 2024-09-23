import numpy as np

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
