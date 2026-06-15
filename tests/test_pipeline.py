"""
End-to-end test of coberus.pipeline.needlet_coadd on a small synthetic
ACT+Planck-like dataset.

Generates apodized maps on overlapping but unequal footprints (wide
Planck-like band, narrower ACT-like band with apodized source holes, and a
small day-time sub-band) with a realistic lensed CMB from a low-accuracy
CAMB call, per-tag Gaussian beams and white noise. The maps are written to
a RAMdisk, coadded with 'block', 'gaussian' (SHT) and 'tophat' (SHT)
covariance smoothing, and compared.

Example::

    python scripts/test_needlet_coadd.py --debug
    python scripts/test_needlet_coadd.py # full run
"""

import argparse
import os
import shutil
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pixell import curvedsky as cs, enmap, enplot
from scipy.stats import binned_statistic as binnedstat

from coberus.pipeline import gauss_beam, needlet_coadd

TAGS = [
    # tag, fwhm_arcmin, noise_uK_arcmin, footprint
    ("planck_143", 7.3, 33.0, "wide"),
    ("planck_217", 5.0, 47.0, "wide"),
    ("act_f090", 2.0, 18.0, "act"),
    ("act_f150", 1.4, 17.0, "act_narrow"),
    ("act_f150_day", 1.4, 25.0, "patch"),
]
BASE_TAG = "planck_143"
# Footprints with very different sky areas: (dec range, optional RA range)
FOOTPRINTS = {
    "wide": ((-75.0, 30.0), None),
    "act": ((-60.0, 20.0), None),
    "act_narrow": ((-40.0, 8.0), None),
    "patch": ((-25.0, 8.0), (20.0, 100.0)),
}
# (dec, ra) source cuts
HOLES_DEG = [(-45.0, 50.0), (-20.0, 150.0), (0.0, 280.0), (-10.0, 60.0)]
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache")


# Copied from msyriac/orphics
class bin1D:
    '''
    * Takes data defined on x0 and produces values binned on x.
    * Assumes x0 is linearly spaced and continuous in a domain?
    * Assumes x is continuous in a subdomain of x0.
    * Should handle NaNs correctly.
    '''
    

    def __init__(self, bin_edges):

        self.update_bin_edges(bin_edges)


    def update_bin_edges(self,bin_edges):
        
        self.bin_edges = bin_edges
        self.numbins = len(bin_edges)-1
        self.cents = (self.bin_edges[:-1]+self.bin_edges[1:])/2.

        self.bin_edges_min = self.bin_edges.min()
        self.bin_edges_max = self.bin_edges.max()

    def bin(self,ix,iy,stat=np.nanmean):
        x = ix.copy()
        y = iy.copy()
        # this just prevents an annoying warning (which is otherwise informative) everytime
        # all the values outside the bin_edges are nans
        y[x<self.bin_edges_min] = 0
        y[x>self.bin_edges_max] = 0

        bin_means = binnedstat(x,y,bins=self.bin_edges,statistic=stat)[0]
        
        return self.cents,bin_means

# Copied from msyriac/orphics
def white_noise(shape=None,wcs=None,noise_muK_arcmin=None,seed=None):
    """
    Generate a non-band-limited white noise map.
    """
    if seed is not None: np.random.seed(seed)
    ipsizemap = enmap.pixsizemap(shape,wcs)
    pmap = ipsizemap*((180.*60./np.pi)**2.)
    div = pmap/noise_muK_arcmin**2.
    return np.random.standard_normal(shape) / np.sqrt(div)


def get_camb_cl(lmax, cache_dir=CACHE_DIR):
    """
    Return the lensed CMB TT power spectrum C_ell in uK^2 up to lmax.

    Uses a low-accuracy CAMB call with fiducial LCDM parameters. The result
    has no free parameters besides lmax, so it is cached to disk.

    Args:
        lmax: Maximum multipole.
        cache_dir: Directory for the cached spectrum.

    Returns:
        1D array of C_ell from ell=0 to lmax.
    """
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, f"camb_cltt_lmax_{lmax}.npy")
    if os.path.exists(fname):
        return np.load(fname)
    import camb

    pars = camb.set_params(
        H0=67.5,
        ombh2=0.022,
        omch2=0.122,
        As=2.1e-9,
        ns=0.965,
        lmax=lmax + 500,
        lens_potential_accuracy=0,
    )
    pars.set_accuracy(AccuracyBoost=0.5, lAccuracyBoost=0.5)
    results = camb.get_results(pars)
    cl = results.get_cmb_power_spectra(pars, CMB_unit="muK", raw_cl=True)["total"][
        : lmax + 1, 0
    ]
    np.save(fname, cl)
    return cl


def cos_taper(x):
    """Cosine taper rising from 0 at x<=0 to 1 at x>=1 for array x."""
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(x, 0, 1))


def make_geometry(dec_range_deg, ra_range_deg, res):
    """
    Build a CAR geometry for a dec band, optionally cropped in RA.

    The band is sliced from the global full-sky pixelization (and the RA
    crop is pixel-snapped from it), so all footprints are pixel-compatible
    with each other.

    Args:
        dec_range_deg: (dec_lo, dec_hi) of the footprint in degrees.
        ra_range_deg: Optional (ra_lo, ra_hi) in degrees; None keeps the
            full RA circle.
        res: Pixel resolution in radians.

    Returns:
        (shape, wcs) tuple.
    """
    shape, wcs = enmap.band_geometry(np.deg2rad(dec_range_deg), res=res)
    if ra_range_deg is not None:
        # RA ordered high -> low to match the native decreasing-RA axis
        box = np.deg2rad(
            [
                [dec_range_deg[0], ra_range_deg[1]],
                [dec_range_deg[1], ra_range_deg[0]],
            ]
        )
        shape, wcs = enmap.subgeo(shape, wcs, box=box)
    return shape, wcs


def make_apod(
    shape,
    wcs,
    dec_range_deg,
    taper_deg,
    ra_range_deg=None,
    holes_deg=None,
    hole_deg=2.0,
):
    """
    Build an apodization map for a footprint with optional holes.

    Args:
        shape, wcs: Geometry of the map.
        dec_range_deg: (dec_lo, dec_hi) of the footprint in degrees.
        taper_deg: Width of the cosine taper at the footprint edges in
            degrees.
        ra_range_deg: Optional (ra_lo, ra_hi) in degrees for RA-cropped
            patches; adds a taper at the RA edges.
        holes_deg: Optional list of (dec, ra) hole centers in degrees.
        hole_deg: Hole radius in degrees; the taper extends over another
            hole_deg beyond the radius.

    Returns:
        Apodization enmap in [0, 1].
    """
    dec, ra = enmap.posmap(shape, wcs)
    lo, hi = np.deg2rad(dec_range_deg)
    w = np.deg2rad(taper_deg)
    apod = cos_taper(np.minimum(dec - lo, hi - dec) / w)
    if ra_range_deg is not None:
        rlo, rhi = np.deg2rad(ra_range_deg)
        apod = apod * cos_taper(np.minimum(ra - rlo, rhi - ra) / w)
    if holes_deg:
        pts = np.deg2rad(np.asarray(holes_deg)).T
        r = enmap.distance_from(shape, wcs, pts)
        apod = apod * cos_taper((r - np.deg2rad(hole_deg)) / np.deg2rad(hole_deg))
    return enmap.enmap(apod, wcs)


def make_dataset(dset_dir, res_arcmin, lmax, seed):
    """
    Generate the synthetic dataset and write maps and binary masks to disk.

    Each tag's map is a common CMB realization convolved with the tag's
    beam, plus independent white noise, multiplied by an apodization over
    the tag's footprint. The binary mask keeps apod > 0.99.

    Args:
        dset_dir: Output directory (ideally on a RAMdisk).
        res_arcmin: Pixel resolution in arcminutes.
        lmax: Maximum multipole for the CMB realization.
        seed: Master RNG seed.

    Returns:
        Dict mapping tag to {'map': path, 'mask': path, 'fwhm': fwhm}.
    """
    os.makedirs(dset_dir, exist_ok=True)
    cl = get_camb_cl(lmax)
    alm = cs.rand_alm(cl, lmax=lmax, seed=seed)
    res = np.deg2rad(res_arcmin / 60.0)
    ells = np.arange(lmax + 1)
    info = {}
    for i, (tag, fwhm, noise, foot) in enumerate(TAGS):
        dec_range, ra_range = FOOTPRINTS[foot]
        shape, wcs = make_geometry(dec_range, ra_range, res)
        omap = cs.alm2map(
            cs.almxfl(alm.copy(), gauss_beam(ells, fwhm)), enmap.empty(shape, wcs)
        )
        rng = np.random.default_rng(seed + 100 + i)
        omap = omap + white_noise(shape,wcs,noise)
        holes = HOLES_DEG if foot != "wide" else None
        apod = make_apod(
            shape,
            wcs,
            dec_range,
            taper_deg=3.0,
            ra_range_deg=ra_range,
            holes_deg=holes,
        )
        mfile = os.path.join(dset_dir, f"map_{tag}.fits")
        kfile = os.path.join(dset_dir, f"mask_{tag}.fits")
        enmap.write_map(mfile, omap * apod)
        enmap.write_map(kfile, (apod > 0.99).astype(np.float64))
        info[tag] = {"map": mfile, "mask": kfile, "fwhm": fwhm}
        print(f"Wrote {tag}: shape={shape}, fwhm={fwhm}', noise={noise} uK-arcmin")
    return info


def eplot(imap, fname, downgrade, **kwargs):
    """Render an enmap with pixell.enplot and write it to fname (PNG)."""
    p = enplot.plot(imap, downgrade=downgrade, colorbar=True, ticks=20, **kwargs)
    enplot.write(fname, p)


def binned_coadd_cl(imap, mask, lmax, bin_edges, out_beam_fwhm):
    """
    Compute the binned, beam-deconvolved power spectrum of a masked coadd.

    Uses w2 = int(mask^2)dA / 4pi normalization for the binary
    footprint mask.

    Args:
        imap: Coadded enmap (already zeroed outside mask).
        mask: Binary footprint enmap.
        lmax: Maximum multipole.
        bin_edges: Multipole bin edges.
        out_beam_fwhm: Output beam FWHM in arcminutes to deconvolve.

    Returns:
        (bin centers, binned C_ell) tuple.
    """
    w2 = np.sum(mask**2 * mask.pixsizemap()) / (4 * np.pi)
    cl = cs.alm2cl(cs.map2alm(imap, lmax=lmax)) / w2
    cl /= gauss_beam(np.arange(cl.size), out_beam_fwhm) ** 2
    return bin1D(bin_edges).bin(np.arange(cl.size), cl)


def plot_spectra(outmaps, cl_theory, lmax, out_beam_fwhm, fname):
    """
    Plot theory vs measured coadd D_ell for each smoothing run.

    The top panel shows the theory curve, binned theory and binned coadd
    bandpowers; the bottom panel shows the ratio of each coadd to the
    binned theory, where the deficit from unity reflects the empirical-ILC
    bias of each covariance smoothing scheme.

    Args:
        outmaps: Dict mapping run name to needlet_coadd output dict.
        cl_theory: Theory C_ell used to generate the CMB.
        lmax: Maximum multipole.
        out_beam_fwhm: Output beam FWHM in arcminutes.
        fname: Output PNG path.
    """
    ells = np.arange(lmax + 1)
    dfact = ells * (ells + 1) / (2 * np.pi)
    bin_edges = np.arange(40, lmax - 10, 40)
    cents, clt = bin1D(bin_edges).bin(ells, cl_theory)
    bfact = cents * (cents + 1) / (2 * np.pi)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True, height_ratios=[2, 1]
    )
    ax1.plot(ells, dfact * cl_theory, "k-", lw=1, label="CAMB theory")
    ax1.plot(cents, bfact * clt, "k_", ms=10, label="binned theory")
    for run, marker in zip(outmaps, ["o", "s", "^"]):
        _, cl = binned_coadd_cl(
            outmaps[run]["coadd"],
            outmaps[run]["mask"],
            lmax,
            bin_edges,
            out_beam_fwhm,
        )
        ax1.plot(cents, bfact * cl, marker, ms=4, label=f"coadd ({run})")
        ax2.plot(cents, cl / clt, marker, ms=4)
    ax1.set_ylabel(r"$D_\ell$ [$\mu$K$^2$]")
    ax1.legend()
    ax2.axhline(1, color="k", lw=1)
    ax2.set_xlabel(r"$\ell$")
    ax2.set_ylabel(r"$C_\ell^{\rm coadd} / C_\ell^{\rm theory}$")
    plt.tight_layout()
    plt.savefig(fname, dpi=120)
    plt.close()



def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="test_output",
        help="Directory for plots.",
    )
    parser.add_argument(
        "--out-root",
        default=None,
        help="Prefix for the dataset and intermediates. Defaults to a "
        "RAMdisk (/dev/shm/) when it has enough free space, else /tmp/.",
    )
    parser.add_argument(
        "--res-arcmin", type=float, default=4.0, help="Pixel resolution in arcminutes."
    )
    parser.add_argument("--lmax", type=int, default=500, help="Maximum multipole.")
    parser.add_argument(
        "--lpeaks",
        type=int,
        nargs="+",
        default=None,
        help="Cosine needlet lpeaks (default: a few scales).",
    )
    parser.add_argument(
        "--out-beam-fwhm",
        type=float,
        default=8.0,
        help="Output beam FWHM in arcminutes.",
    )
    parser.add_argument(
        "--ilc-bias-tol",
        type=float,
        default=0.01,
        help="ILC bias tolerance for gaussian smoothing.",
    )
    parser.add_argument(
        "--cov-smooth-factor",
        type=int,
        default=32,
        help="Block downgrade factor for block smoothing.",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of dask workers."
    )
    parser.add_argument("--seed", type=int, default=1234, help="RNG seed.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Coarser resolution and lower lmax for quick tests.",
    )
    args = parser.parse_args(argv) 

    if args.out_root is None:
        # Prefer a RAMdisk, but fall back to /tmp on small /dev/shm mounts
        free_gib = shutil.disk_usage("/dev/shm").free / 1024**3
        args.out_root = "/dev/shm/cob_test_" if free_gib > 4 else "/tmp/cob_test_"
        print(f"out_root = {args.out_root} (/dev/shm has {free_gib:.1f} GiB free)")

    if args.debug:
        args.res_arcmin = max(args.res_arcmin, 8.0)
        args.lmax = min(args.lmax, 250)
        args.cov_smooth_factor = min(args.cov_smooth_factor, 8)
        args.workers = min(args.workers, 2)
    if args.lpeaks is None:
        args.lpeaks = [lp for lp in [50, 100, 200, 350] if lp < args.lmax]
        args.lpeaks.append(args.lmax)
    print(f"lpeaks = {args.lpeaks}")

    os.makedirs(args.output, exist_ok=True)
    tags = [t[0] for t in TAGS]
    info = make_dataset(
        f"{args.out_root}dataset", args.res_arcmin, args.lmax, args.seed
    )

    figs = []
    for tag in [BASE_TAG, "act_f090", "act_f150_day"]:
        png = f"input_{tag}"
        eplot(
            enmap.read_map(info[tag]["map"]),
            os.path.join(args.output, png),
            downgrade=2,
            range=400,
        )
        figs.append((f"Input map: {tag}", f"{png}.png"))

    timings, outmaps = {}, {}
    for run in ["block", "gaussian", "tophat"]:
        print(f"=== Running needlet_coadd with {run} covariance smoothing ===")
        t0 = time.time()
        outmaps[run] = needlet_coadd(
            map_fname_func=lambda tag: info[tag]["map"],
            mask_fname_func=lambda tag: info[tag]["mask"],
            tags=tags,
            base_tag=BASE_TAG,
            lpeaks=args.lpeaks,
            lmins=[None] * len(tags),
            lmaxs=[None] * len(tags),
            response_func=lambda tag: 1.0,
            beam_func=lambda tag, ells: gauss_beam(np.asarray(ells), info[tag]["fwhm"]),
            out_beam_fwhm=args.out_beam_fwhm,
            out_root=f"{args.out_root}{run}_",
            cov_smooth_type=run,
            cov_smooth_factor=args.cov_smooth_factor,
            ilc_bias_tol=args.ilc_bias_tol,
            n_workers=args.workers,
            delete_intermediate=True,
        )
        timings[run] = time.time() - t0
        print(f"{run} run took {timings[run] / 60.0:.2f} min")
        png = f"coadd_{run}"
        eplot(
            outmaps[run]["coadd"],
            os.path.join(args.output, png),
            downgrade=2,
            range=400,
        )
        figs.append((f"Coadd: {run} smoothing", f"{png}.png"))

    eplot(
        outmaps["block"]["mask"],
        os.path.join(args.output, "mask"),
        downgrade=2,
        min=0,
        max=1,
    )
    figs.append(("Final footprint mask", "mask.png"))
    for run in ["gaussian", "tophat"]:
        png = f"diff_{run}_block"
        eplot(
            outmaps[run]["coadd"] - outmaps["block"]["coadd"],
            os.path.join(args.output, png),
            downgrade=2,
            range=50,
        )
        figs.append((f"Difference: {run} - block", f"{png}.png"))

    plot_spectra(
        outmaps,
        get_camb_cl(args.lmax),
        args.lmax,
        args.out_beam_fwhm,
        os.path.join(args.output, "spectra.png"),
    )
    figs.append(("Coadd power spectra vs theory", "spectra.png"))
    return outmaps[run]["coadd"]



def test_main():

    x = main([
        "--debug"
    ])
    assert x is not None
    assert(x.wcs is not None)

if __name__ == "__main__":
    main()
