from pixell import enmap
import numpy as np

import argparse

# CHANGE THIS
DEFAULT_SIM_PATH = "/data7/jaejoonk/coberus_noiseonly_sims/" + \
                   "sim_#_lensmode_noiseonly_coadd_covsmooth_64.fits"

# need to subdivide sims into blocks
# to process a (potentially) large number
# of noise maps to infer inverse variance map.
def block_shape(nblocks, shape):
    blockdim = int(np.sqrt(nblocks))
    if len(shape) == 3:
        return (shape[0],
                shape[1] // blockdim,
                shape[2] // blockdim)
    else:
        return (shape[0] // blockdim, shape[1] // blockdim)

# return [(top left pixel coords), (bottom right pixel coords)]
def index_block(block_index, nblocks, block_shape):
    blockdim = int(np.sqrt(nblocks))
    row, col = block_index // blockdim, block_index % blockdim
    return [(row * block_shape[0], col * block_shape[1]),
            ((row+1) * block_shape[0], (col+1) * block_shape[1])]

# apply index_block to a map
def apply_index_block(imap, coords_obj):
    [top_left, bottom_right] = coords_obj
    return imap[top_left[0]:bottom_right[0],
                top_left[1]:bottom_right[1]] 

# return {idx: [(top left pixel coords, bottom right pixel coords)],
#         idx2: ...}
def all_block_indices(nblocks, block_shape):
    all_blocks = {}
    for i in range(nblocks):
        all_blocks[i] = index_block(i, nblocks, block_shape)

# return a map of all blocks stitched
def stitch(all_blocks):
    nblocks = len(all_blocks.keys())
    blockdim = int(np.sqrt(nblocks))
    block_shape = np.array(all_blocks[0]).shape
    print("Block shape: ", block_shape)
    # build full matrix at once
    imap = np.zeros((block_shape[0] * blockdim,
                     block_shape[1] * blockdim))
    
    # and fill in block by block
    for block_index in range(nblocks):
        row, col = block_index // blockdim, block_index % blockdim
        block = np.array(all_blocks[block_index])

        imap[row*block_shape[0]:(row+1)*block_shape[0],
             col*block_shape[1]:(col+1)*block_shape[1]] = block
    return imap

if __name__ == '__main__':
    # Parse command line
    parser = argparse.ArgumentParser(description="Build an ivar map, assuming pixel-to-pixel independence.")
    parser.add_argument("out_name", type=str, help="Name of outputs. Could include a path.")
    parser.add_argument("--sim-path", type=str, default=DEFAULT_SIM_PATH,
                        help="Path to noise sims.")
    parser.add_argument("--token", type=str, default='#',
                        help="Sim path token to retrieve specific sim")
    parser.add_argument("--sim-min", type=int, default=0, help="Minimum sim index")
    parser.add_argument("--sim-max", type=int, default=400, help="Maximum sim index")
    parser.add_argument("--chunks", type=int, default=64,
                        help="Number of chunks (preferably perfect squares)")
    parser.add_argument("--verbose", type=store_true, default=True, help="Verbose outputs")

    args = parser.parse_args()
    
    all_blocks = {}
    # save wcs somewhere
    wcs = None
    for chunk in range(args.chunks):
        print(f"- Chunk #{chunk+1} / {args.chunks}")
        all_chunks = []
        for sim_index in range(args.sim_min, args.sim_max+1):
            if args.verbose:
                print(f"-- Sim #{sim_index} / {args.sim_max-args.sim_min+1}: ", end="")
            sim_path = args.sim_path.replace(args.token, str(sim_index))
            try:
                imap_full = enmap.read_map(sim_path)
                wcs = imap_full.wcs
                if args.verbose: print(f"Loaded from {sim_path}.")
            except FileNotFoundError:
                    if args.verbose: print(f"Could not find {sim_path}. Skipping.")
                continue
            

            shape = block_shape(args.chunks, imap_full.shape)
            if args.verbose and sim_index == args.sim_min:
                print(f"Full shape: ({imap_full.shape[0]}, {imap_full.shape[1]})")
                print(f"Block shape: ({shape[0]}, {shape[1]})")
            imap_coords = index_block(chunk, args.chunks, shape)
            chunk_full = np.copy(apply_index_block(imap_full, imap_coords))
            del imap_full

            all_chunks.append(chunk_full)

        # build variance map
        chunk_var = np.var(np.array(all_chunks), axis=0)
        all_blocks[chunk] = chunk_var

    # stitch blocks, and then take inverse for ivar map
    omap = enmap.enmap(1. / stitch(all_blocks), wcs=wcs)
    # write to disk
    enmap.write_map(args.out_name + ".fits", omap)
    print(f"Wrote output to {args.out_name}.fits.") 
