from pixell import enmap
from orphics import maps # github.com/msyriac/orphics
import numpy as np
import utils
import os,sys
from concurrent import futures

# ARGPARSE

import argparse
# Parse command line
parser = argparse.ArgumentParser(description='Make a mask.')
parser.add_argument("outname", type=str,help='Name of outputs. Could include a path.')
parser.add_argument("--dfact",     type=int,  default=2,help="Downgrade factor.")
parser.add_argument("--nworkers",     type=int,  default=None,help="Maximum number of workers to parallelize over. Defaults to number of jobs.")
parser.add_argument("--tags",     type=str,  default=None,help="Comma separated list of tags.")
parser.add_argument("--srcfull", action='store_true',help='Whether to use maps with sources in them.')
args = parser.parse_args()

if args.tags is None:
    tags = utils.act_tags + utils.planck_tags
    tags.remove('daywide_pa4_f220')
    tags.remove('daywide_pa6_f090')
    tags.remove('daywide_pa6_f150')
    tags.remove('030')
    tags.remove('044')
    tags.remove('070')
    tags.remove('545')
else:
    tags = args.tags.split(',')

out = os.path.join(utils.out_root,args.outname)

def do_job(i):
    desc = jobs[i]
    tag = desc[0]
    print(f"Doing {tag} {desc[1]}...")
    fname = utils.get_filename(desc[0],maptype=desc[1],splitnum=None,srcfree=not(args.srcfull))
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
    
