from pixell import enmap
from orphics import maps
import numpy as np
import utils
import os,sys

from concurrent import futures


# ARGPARSE

import argparse
# Parse command line
parser = argparse.ArgumentParser(description='Make a mask.')
parser.add_argument("outname", type=str,help='Name of outputs. Could include a path.')
# parser.add_argument("--rms-threshold",     type=float,  default=70.0,help="RMS threshold in uK-arcmin for ivar maps.")
parser.add_argument("--dfact",     type=int,  default=4,help="Downgrade factor.")
# parser.add_argument("--width-deg",     type=float,  default=0.2,help="Width in deg. to grow mask by.")
parser.add_argument("--nworkers",     type=int,  default=None,help="Maximum number of workers to parallelize over. Defaults to number of jobs.")
# parser.add_argument("--template-fname",     type=str,  default=None,help="Path to template ACT/SO map.",required=True)
# parser.add_argument("--planck-base-name",     type=str,  default=None,help="Root file name to Planck cut maps",required=True)
# parser.add_argument("--planck-cuts",     type=str,  default="60,70,80",help="Galactic cuts, comma spearated.")
# parser.add_argument("--ivar-search-string",     type=str,  default="cmb_night_?_3pass_4way_*_ivar.fits",help="Search string to find inverse variance maps in args.exp-path. The question mark will be replaced by the items in args.arrays.")
parser.add_argument("--tags",     type=str,  default='143,night_pa5_f090,daydeep_pa5_f150',help="Comma separated list of tags.")
# parser.add_argument("--exp-path",     type=str, help="Path to ACT/SO maps.",required=True)
args = parser.parse_args()


tags = args.tags.split(',')

out = os.path.join(utils.out_root,args.outname)

def do_job(i):
    desc = jobs[i]
    tag = desc[0]
    print(f"Doing {tag} {desc[1]}...")
    fname = utils.get_filename(desc[0],maptype=desc[1],splitnum=None,srcfree=True)
    if desc[1]=='map':
        sel = np.s_[0,...]
        op = np.mean
        post_op = lambda x : x
    elif desc[1]=='ivar':
        sel = None if tag in utils.act_tags else np.s_[0,...]
        op = np.sum
        post_op = lambda x : x
    elif desc[1][:4]=='mask':
        sel = None
        op = np.mean
        post_op = lambda x : maps.binary_mask(x)
    imap = enmap.read_map(fname,sel=sel)
    if imap.ndim!=2:
        print(imap.shape,tag,desc[1])
        raise ValueError
    enmap.write_map(f'{out}_{tag}_{desc[1]}.fits',post_op(utils.downgrade(imap,args.dfact,op=op)))

    return None

jobs = [(t,x) for t in tags for x in ['map','ivar','mask80']]
njobs = len(jobs)

with futures.ProcessPoolExecutor(max_workers=args.nworkers) as executor:
    omaps = list(executor.map(do_job, range(njobs)))
    executor.shutdown(wait=True)
    
