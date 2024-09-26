from pixell import enmap,curvedsky as cs, wavelets as wv,uharm,multimap,utils as u
from orphics import io
import numpy as np

lpeaks = [0.,100.,500.,800.,1000.,2000.,3000.,4000., 5000., 6000., 8000,10000.]


basis = wv.CosineNeedlet(lpeaks = lpeaks)

nwaves = basis.n

ls = np.arange(max(lpeaks))

for i in range(nwaves):
    pl = io.Plotter(xyscale='loglin')
    pl.add(ls,basis(i,ls))
    pl.vline(basis.lmins[i])
    pl.vline(basis.lmaxs[i])
    pl.done(f'cosine_basis_{i}.png')
