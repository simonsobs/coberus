import utils
from orphics import io
from pixell import enmap
import sys

out_root = utils.out_root
outname = sys.argv[1]

def plot(imap,tag,ind,mtype='map',**kwargs):
    io.hplot(imap,f'{out_root}/{outname}_{mtype}_{tag}_scale_{ind}',mask=0,**kwargs)
imap = enmap.read_map(f'{out_root}/{outname}_coadd_map.fits')
plot(imap,"all",0,mtype='coadd',colorbar=True,grid=True,ticks=10,downgrade=2)
