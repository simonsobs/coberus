"""
Check the check_cmb_weights flag and the per-scale weight maps written by
coberus.pipeline.needlet_coadd, using the synthetic dataset from
tests/test_pipeline.py.
"""

import argparse
import glob

import numpy as np
from pixell import enmap
from test_pipeline import BASE_TAG, TAGS, make_dataset

from coberus.pipeline import gauss_beam, needlet_coadd


def main():
    """Run a block-smoothed CMB NILC with check_cmb_weights and check outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", default="/tmp/cob_chk_")
    parser.add_argument("--debug", action="store_true", help="Low-res quick run.")
    args = parser.parse_args()
    res, lmax = (8.0, 250) if args.debug else (4.0, 500)
    lpeaks = [lp for lp in [50, 100, 200, 350] if lp < lmax] + [lmax]

    tags = [t[0] for t in TAGS]
    info = make_dataset(f"{args.out_root}dataset", res, lmax, 1234)
    root = f"{args.out_root}run_"
    out = needlet_coadd(
        map_fname_func=lambda tag: info[tag]["map"],
        mask_fname_func=lambda tag: info[tag]["mask"],
        tags=tags,
        base_tag=BASE_TAG,
        lpeaks=lpeaks,
        lmins=[None] * len(tags),
        lmaxs=[None] * len(tags),
        response_func=lambda tag: 1.0,
        beam_func=lambda tag, ells: gauss_beam(np.asarray(ells), info[tag]["fwhm"]),
        out_beam_fwhm=8.0,
        out_root=root,
        cov_smooth_type="block",
        cov_smooth_factor=8,
        n_workers=2,
        check_cmb_weights=True,
    )
    print("keys:", sorted(out))
    for k in range(len(lpeaks)):
        files = sorted(glob.glob(f"{root}wavelet_weights_scale_{k}_*.fits"))
        ws = np.array([enmap.read_map(f) for f in files])
        tot = ws.sum(axis=0)
        nz = tot != 0
        print(
            f"scale {k}: {len(files)} weight maps, shape {ws.shape[1:]}, "
            f"sum over tags where nonzero: {tot[nz].min():.5f}..{tot[nz].max():.5f}"
        )


if __name__ == "__main__":
    main()
