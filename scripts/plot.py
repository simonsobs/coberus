import utils
from orphics import io
from pixell import enmap
import sys

out_root = utils.out_root
outname = sys.argv[1]

imap = enmap.read_map(f'{out_root}/{outname}.fits')
io.hplot(imap,"coadd_map",colorbar=True,grid=True,ticks=10,downgrade=2)
