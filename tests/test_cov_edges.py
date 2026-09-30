"""
Test of the footprint-normalized covariance estimate in coberus.pipeline
near overlapping mask edges.

Three maps share a white "CMB" signal (variance 1) plus independent white
noise, on masks A: dec < 4, B: dec > -4, C: |dec| < 30 (degrees), within
|dec| < 30. In -4 < dec < 4 all three maps contribute and the edges of both
A and B are within reach of the smoothing kernel. The covariance maps from
normalized_delta + coverage_covariances (Gaussian SHT smoothing, as in
needlet_coadd) are compared with the truth 1 + delta_ij * noise_i, and the
per-pixel covariance of the maps covering each pixel is checked to be
positive definite. The previous unnormalized estimator S[(w - S[w])^2] is
also computed, to check that the test is sensitive to the edge bias.

Example::

    python tests/test_cov_edges.py --debug
    python tests/test_cov_edges.py --output test_output
"""

import argparse
import os
import tempfile

import numpy as np
from pixell import enmap

from coberus.pipeline import (
    band_filter,
    cov_filter,
    coverage_covariances,
    normalized_delta,
)

NOISE_VAR = np.array([0.5, 1.0, 2.0])
DEC_LIM = 30.0


def make_masks(shape, wcs):
    """
    Build the binary masks A (dec < 4), B (dec > -4) and C, all restricted
    to |dec| < DEC_LIM degrees.

    Args:
        shape, wcs: Geometry of the maps.

    Returns:
        List of three binary enmaps.
    """
    dec = np.rad2deg(enmap.posmap(shape, wcs)[0])
    inside = np.abs(dec) < DEC_LIM
    return [
        enmap.enmap((inside & (dec < 4)).astype(np.float64), wcs),
        enmap.enmap((inside & (dec > -4)).astype(np.float64), wcs),
        enmap.enmap(inside.astype(np.float64), wcs),
    ]


def run_edge_test(workdir, res_arcmin=30.0, sigma_deg=3.0, nsim=8, seed=0):
    """
    Estimate covariance maps for nsim realizations and compare with truth.

    Args:
        workdir: Directory for the intermediate FITS files.
        res_arcmin: Pixel resolution in arcminutes.
        sigma_deg: Gaussian covariance smoothing scale in degrees.
        nsim: Number of realizations to average.
        seed: RNG seed.

    Returns:
        Dict with 'dec' (row declinations in degrees), 'ratio' and
        'naive_ratio' (dicts mapping (i, j) to the RA-averaged, sim-averaged
        ratio of the estimated to the true C_ij per row, NaN where either
        map is masked), and 'min_eig' (smallest per-pixel eigenvalue of the
        covariance of the covering maps, over all pixels and sims).
    """
    res = np.deg2rad(res_arcmin / 60.0)
    shape, wcs = enmap.band_geometry(np.deg2rad((-DEC_LIM - 10, DEC_LIM + 10)), res=res)
    masks = make_masks(shape, wcs)
    mfnames = []
    for i, mask in enumerate(masks):
        mfnames.append(os.path.join(workdir, f"mask_{i}.fits"))
        enmap.write_map(mfnames[-1], mask)

    sigma = np.deg2rad(sigma_deg)
    lmax = int(180 * 60 / res_arcmin)
    fl = cov_filter("gaussian", sigma, lmax, False, 0.5)

    def smooth(imap):
        return band_filter(imap, fl, lmax)

    n = len(masks)
    pairs = [(i, j) for i in range(n) for j in range(i, n)]
    truth = 1.0 + np.diag(NOISE_VAR)
    ratio = {p: 0.0 for p in pairs}
    naive_ratio = {p: 0.0 for p in pairs}
    min_eig = np.inf
    covered = np.array([m != 0 for m in masks])
    rng = np.random.default_rng(seed)
    for _ in range(nsim):
        sig = rng.standard_normal(shape)
        wmaps = [
            enmap.enmap(sig + np.sqrt(nv) * rng.standard_normal(shape), wcs) * m
            for nv, m in zip(NOISE_VAR, masks)
        ]
        dfnames = []
        for i, (w, m) in enumerate(zip(wmaps, masks)):
            dfnames.append(os.path.join(workdir, f"delta_{i}.fits"))
            enmap.write_map(dfnames[-1], normalized_delta(w, m, smooth))
        fcovs = coverage_covariances(
            dfnames,
            mfnames,
            lambda i, j: os.path.join(workdir, f"cov_{i}_{j}.fits"),
            smooth,
        )
        cov = np.zeros((n, n) + shape)
        for i, j in pairs:
            cov[i, j] = cov[j, i] = enmap.read_map(fcovs[i][j])
            both = covered[i] & covered[j]
            ratio[(i, j)] += row_mean(cov[i, j] / truth[i, j], both) / nsim
            dn = [w - smooth(w) for w in (wmaps[i], wmaps[j])]
            naive = smooth(dn[0] * dn[1])
            naive_ratio[(i, j)] += row_mean(naive / truth[i, j], both) / nsim
        min_eig = min(min_eig, min_covering_eig(cov, covered))
    dec = np.rad2deg(enmap.posmap(shape, wcs)[0][:, 0])
    return {"dec": dec, "ratio": ratio, "naive_ratio": naive_ratio, "min_eig": min_eig}


def row_mean(imap, sel):
    """
    Mean of imap over the selected pixels of each row.

    Args:
        imap: 2D array.
        sel: Boolean array of the same shape.

    Returns:
        1D array of row means, NaN for rows with no selected pixels.
    """
    cnt = sel.sum(axis=1)
    tot = np.where(sel, imap, 0).sum(axis=1)
    return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)


def min_covering_eig(cov, covered):
    """
    Smallest eigenvalue of the per-pixel covariance of the covering maps.

    Args:
        cov: Array (n, n, ny, nx) of covariance maps.
        covered: Boolean array (n, ny, nx) of mask coverage.

    Returns:
        The smallest eigenvalue over all pixels covered by any map.
    """
    code = sum(covered[i].astype(np.int64) << i for i in range(len(covered)))
    out = np.inf
    for t in np.unique(code):
        if t == 0:
            continue
        idx = [i for i in range(len(covered)) if (t >> i) & 1]
        sub = cov[np.ix_(idx, idx)][:, :, code == t]
        ev = np.linalg.eigvalsh(np.moveaxis(sub, -1, 0))
        out = min(out, ev[:, 0].min())
    return out


def check_results(res, tol=0.05):
    """
    Assert that the covariance is unbiased and positive definite, and that
    the naive estimator is visibly biased at the overlapping edges.

    Args:
        res: Output of run_edge_test.
        tol: Allowed absolute deviation of the row-averaged ratio from 1.
    """
    for p, r in res["ratio"].items():
        dev = np.nanmax(np.abs(r - 1))
        assert dev < tol, f"C_{p} biased: max |ratio-1| = {dev:.3f}"
    assert res["min_eig"] > 0, f"covariance not PD: min eig {res['min_eig']:.3e}"
    strip = np.abs(res["dec"]) < 3
    naive = res["naive_ratio"][(0, 1)][strip]
    assert np.nanmin(naive) < 1 - 3 * tol, "naive estimator unexpectedly unbiased"


def test_cov_edges(tmp_path):
    """Covariance near overlapping mask edges is unbiased and PD."""
    check_results(run_edge_test(str(tmp_path)))


def plot_results(res, fname):
    """
    Plot the row-averaged estimated/true covariance ratio against dec.

    Args:
        res: Output of run_edge_test.
        fname: Output PNG path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = "ABC"
    _, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, key, title in zip(
        axes,
        ["naive_ratio", "ratio"],
        ["previous: S[(w - S[w])^2]", "coverage-normalized"],
    ):
        for (i, j), r in res[key].items():
            ax.plot(res["dec"], r, label=f"C_{names[i]}{names[j]}")
        for edge in (-4, 4):
            ax.axvline(edge, color="k", ls=":", lw=1)
        ax.axhline(1, color="k", lw=1)
        ax.set_xlabel("dec [deg]")
        ax.set_title(title)
    axes[0].set_ylabel("estimated / true covariance")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fname, dpi=120)
    plt.close()


def main():
    """Run the edge test from the command line and optionally plot it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--res-arcmin", type=float, default=30.0)
    parser.add_argument("--sigma-deg", type=float, default=3.0)
    parser.add_argument("--nsim", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None, help="Directory for a plot.")
    parser.add_argument(
        "--debug", action="store_true", help="Coarser pixels and fewer sims."
    )
    args = parser.parse_args()
    if args.debug:
        args.res_arcmin, args.nsim = 60.0, 2
    with tempfile.TemporaryDirectory() as workdir:
        res = run_edge_test(
            workdir, args.res_arcmin, args.sigma_deg, args.nsim, args.seed
        )
    for p in res["ratio"]:
        print(
            f"C_{p}: max |ratio-1| = {np.nanmax(np.abs(res['ratio'][p] - 1)):.4f}, "
            f"naive min ratio = {np.nanmin(res['naive_ratio'][p]):.3f}"
        )
    print(f"min eigenvalue = {res['min_eig']:.3e}")
    if args.output is not None:
        os.makedirs(args.output, exist_ok=True)
        plot_results(res, os.path.join(args.output, "cov_edges.png"))
    if not args.debug:
        check_results(res)
        print("PASSED")


if __name__ == "__main__":
    main()
