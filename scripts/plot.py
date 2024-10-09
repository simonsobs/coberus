import utils
from orphics import io
from pixell import enmap
import sys

outname = sys.argv[1]

imap = enmap.read_map(f'{outname}')
outpng = outname.replace('.fits','')
io.hplot(imap,f"{outpng}",colorbar=True,grid=True,ticks=10,downgrade=2)
