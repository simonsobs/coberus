"""
Two-map test of coberus.pipeline.needlet_coadd.

Map 'wide' covers a large dec band and is noisy; map 'deep' covers an
RA/dec patch fully inside the wide footprint and has lower noise. Both
share the same CMB realization and the same Gaussian beam (equal to the
output beam), with independent white noise.

For a common signal plus white noise with equal beams, the exact ILC
weights at every needlet scale are inverse-noise-variance weights,

    w_deep = s_wide^2 / (s_wide^2 + s_deep^2),   w_wide = 1 - w_deep,

inside the deep footprint, and w_wide = 1 outside it. The script checks:

1. Weight maps per scale against these values, including a profile of
   w_deep against distance to the deep-map mask edge.
2. Signal preservation: the signal-only maps coadded with the data
   weights (via nmap_labels) should reproduce the input CMB.
3. Noise: the noise-only coadd should have variance s_wide^2 outside
   the deep footprint and s_wide^2 s_deep^2 / (s_wide^2 + s_deep^2)
   inside it.

Example::

    python tests/check_two_map_subset.py --debug
"""

import argparse
import glob
import itertools
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from coberus.pipeline import gauss_beam, needlet_coadd
from pixell import curvedsky as cs
from pixell import enmap

from test_pipeline import get_camb_cl, make_apod, make_geometry, white_noise

WIDE = ("wide", (-50.0, 10.0), None)
DEEP = ("deep", (-35.0, -5.0), (40.0, 140.0))


def make_two_maps(dset_dir, res_arcmin, lmax, fwhm, noises, taper_deg, seed):
    """
    Write data, signal-only and noise-only maps and binary masks for the
    wide and deep tags.

    Args:
        dset_dir: Output directory.
        res_arcmin: Pixel resolution in arcminutes.
        lmax: Maximum multipole of the CMB realization.
        fwhm: Beam FWHM in arcminutes for both maps.
        noises: Dict tag -> white noise level in uK-arcmin.
        taper_deg: Width of the apodization taper in degrees.
        seed: RNG seed.

    Returns:
        (info, alm) where info maps tag -> dict of file paths for 'map',
        'signal', 'noise' and 'mask', and alm is the CMB realization.
    """
    os.makedirs(dset_dir, exist_ok=True)
    alm = cs.rand_alm(get_camb_cl(lmax), lmax=lmax, seed=seed)
    balm = cs.almxfl(alm, gauss_beam(np.arange(lmax + 1), fwhm))
    res = np.deg2rad(res_arcmin / 60.0)
    info = {}
    for i, (tag, dec_range, ra_range) in enumerate([WIDE, DEEP]):
        shape, wcs = make_geometry(dec_range, ra_range, res)
        apod = make_apod(shape, wcs, dec_range, taper_deg, ra_range_deg=ra_range)
        sig = cs.alm2map(balm, enmap.empty(shape, wcs))
        np.random.seed(seed + 1 + i)
        noise = white_noise(shape, wcs, noises[tag])
        info[tag] = {}
        for kind, m in [
            ("map", (sig + noise) * apod),
            ("signal", sig * apod),
            ("noise", noise * apod),
            ("mask", (apod > 0.99).astype(np.float64)),
        ]:
            fname = os.path.join(dset_dir, f"{kind}_{tag}.fits")
            enmap.write_map(fname, enmap.enmap(m, wcs))
            info[tag][kind] = fname
        print(f"Wrote {tag}: shape={shape}, noise={noises[tag]} uK-arcmin")
    return info, alm


def signed_edge_distance(mask):
    """
    Signed distance in degrees to the edge of a binary mask: positive
    inside the mask, negative outside.

    Args:
        mask: Binary enmap.

    Returns:
        enmap of signed distances in degrees.
    """
    inside = enmap.distance_transform(mask > 0)
    outside = enmap.distance_transform(mask == 0)
    return enmap.enmap(np.rad2deg(np.where(mask > 0, inside, -outside)), mask.wcs)


def analyze_run(out, root, info, alm, args, expected, run):
    """
    Compare one needlet_coadd run against the analytic expectations.

    Args:
        out: needlet_coadd output dict.
        root: out_root used for the run (to find the weight maps).
        info: Dataset info from make_two_maps.
        alm: Input CMB alm.
        args: Parsed CLI arguments.
        expected: Dict of expected values.
        run: Name of the run, used in figure names.

    Returns:
        (summary dict, list of (caption, png) figures).
    """
    mask_wide = enmap.read_map(info["wide"]["mask"])
    shape, wcs = mask_wide.shape, mask_wide.wcs
    mask_deep = enmap.extract(enmap.read_map(info["deep"]["mask"]), shape, wcs)
    d_wide = signed_edge_distance(mask_wide)
    d_deep = signed_edge_distance(mask_deep)
    buf = args.edge_buffer
    overlap = (d_deep > buf) & (d_wide > buf)
    wide_only = (d_deep < -buf) & (d_wide > buf)

    summary = {"scales": []}
    figs = []
    nscale = len(args.lpeaks)
    _, axes = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.arange(-15, 15.5, 0.5)
    cents = 0.5 * (bins[1:] + bins[:-1])
    for k in range(nscale):
        fw = glob.glob(f"{root}wavelet_weights_scale_{k}_wide.fits")
        fd = glob.glob(f"{root}wavelet_weights_scale_{k}_deep.fits")
        ww = enmap.read_map(fw[0])
        wd = enmap.read_map(fd[0])
        # Weight maps live on the (downgraded) per-scale wavelet geometry
        dd = enmap.project(d_deep, ww.shape, ww.wcs, order=0)
        dw = enmap.project(d_wide, ww.shape, ww.wcs, order=0)
        ov = (dd > buf) & (dw > buf)
        wo = (dd < -buf) & (dw > buf)
        s = {
            "scale": k,
            "w_deep_overlap_mean": float(wd[ov].mean()),
            "w_deep_overlap_std": float(wd[ov].std()),
            "w_wide_wide_only_mean": float(ww[wo].mean()),
            "w_deep_wide_only_absmax": float(np.abs(wd[wo]).max()),
            "w_deep_near_edge_mean": float(wd[(dd > 0) & (dd < 2) & (dw > buf)].mean()),
            "wsum_absdev_max": float(np.abs((ww + wd)[(dw > 0)] - 1).max()),
        }
        summary["scales"].append(s)
        sel = dw > buf
        prof = [
            wd[sel & (dd >= lo) & (dd < hi)].mean()
            if np.any(sel & (dd >= lo) & (dd < hi))
            else np.nan
            for lo, hi in itertools.pairwise(bins)
        ]
        axes[0].plot(cents, prof, label=f"scale {k} (lpeak {args.lpeaks[k]})")
        axes[1].errorbar(
            k, s["w_deep_overlap_mean"], s["w_deep_overlap_std"], fmt="o", color="C0"
        )
    axes[0].axhline(expected["w_deep"], color="k", ls="--", lw=1)
    axes[0].axvline(0, color="gray", lw=0.5)
    axes[0].set_xlabel("signed distance to deep-map mask edge [deg]")
    axes[0].set_ylabel(r"mean $w_{\rm deep}$")
    axes[0].legend(fontsize=7)
    axes[0].set_title(f"{run}: weight profile across deep edge")
    axes[1].axhline(expected["w_deep"], color="k", ls="--", lw=1, label="expected")
    axes[1].set_xlabel("needlet scale")
    axes[1].set_ylabel(r"$w_{\rm deep}$ in overlap interior (mean $\pm$ std)")
    axes[1].legend()
    plt.tight_layout()
    png = f"two_map_weights_{run}.png"
    plt.savefig(os.path.join(args.output, png), dpi=110)
    plt.close()
    figs.append((f"{run}: deep-map weight vs distance to its mask edge", png))

    # Signal preservation: truth is the beam-convolved CMB band-limited to
    # the needlet support (ell < lmax)
    lmax = max(args.lpeaks)
    fl = gauss_beam(np.arange(lmax + 1), args.fwhm)
    fl[lmax:] = 0
    truth = cs.alm2map(cs.almxfl(alm, fl), enmap.empty(shape, wcs))
    sig = out["signal_coadd"]
    resid = sig - truth
    rms_t = truth[mask_wide > 0].std()
    summary["signal_resid_rel_overlap"] = float(resid[overlap].std() / rms_t)
    summary["signal_resid_rel_wide_only"] = float(resid[wide_only].std() / rms_t)
    summary["signal_resid_rel_all"] = float(resid[mask_wide > 0].std() / rms_t)

    # Noise: measured variance ratios vs expectation. Normalize by the
    # wide-only region so the band-limit of the white noise cancels.
    noi = out["noise_coadd"]
    pix = noi.pixsizemap() * (180 * 60 / np.pi) ** 2  # arcmin^2
    var_ov = np.mean(noi[overlap] ** 2 * pix[overlap])
    var_wo = np.mean(noi[wide_only] ** 2 * pix[wide_only])
    summary["noise_var_ratio_wide_only_over_overlap"] = float(var_wo / var_ov)
    summary["noise_eff_overlap_rel"] = float(np.sqrt(var_ov / var_wo))

    _, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ext = None
    for ax, m, title, vr in [
        (axes[0], out["coadd"], "data coadd", 300),
        (axes[1], resid, "signal coadd - input CMB", 5),
        (
            axes[2],
            noi * np.sqrt(pix),
            r"noise coadd $\times\sqrt{\Omega_{\rm pix}}$ [uK-arcmin]",
            3 * args.noise_wide,
        ),
    ]:
        im = ax.imshow(
            m,
            origin="lower",
            cmap="RdBu_r",
            vmin=-vr,
            vmax=vr,
            extent=ext,
            aspect="auto",
        )
        ax.contour(mask_deep, levels=[0.5], colors="k", linewidths=0.6)
        ax.set_title(f"{run}: {title}")
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    png = f"two_map_maps_{run}.png"
    plt.savefig(os.path.join(args.output, png), dpi=110)
    plt.close()
    figs.append(
        (f"{run}: coadd, signal residual and noise coadd (black: deep mask edge)", png)
    )
    return summary, figs


def write_report(fname, configs, expected, summaries, figs):
    """
    Write an HTML report with the summary tables and figures.

    Args:
        fname: Output HTML path.
        configs: Dict config name -> argparse.Namespace of its parameters.
        expected: Dict of expected values.
        summaries: Dict run -> summary from analyze_run.
        figs: List of (caption, png) tuples.
    """
    cfg_rows = "".join(
        f"<tr><td>{name}</td><td>{c.noise_wide:g}</td><td>{c.noise_deep:g}</td>"
        f"<td>{c.fwhm:.1f}</td><td>{c.lpeaks}</td><td>{c.note}</td></tr>"
        for name, c in configs.items()
    )
    rows = []
    for run, s in summaries.items():
        for sc in s["scales"]:
            rows.append(
                f"<tr><td>{run}</td><td>{sc['scale']}</td>"
                f"<td>{sc['w_deep_overlap_mean']:.4f} &plusmn; "
                f"{sc['w_deep_overlap_std']:.4f}</td>"
                f"<td>{sc['w_deep_near_edge_mean']:.4f}</td>"
                f"<td>{sc['w_wide_wide_only_mean']:.4f}</td>"
                f"<td>{sc['wsum_absdev_max']:.1e}</td></tr>"
            )
    glob_rows = [
        f"<tr><td>{run}</td><td>{s['signal_resid_rel_overlap']:.2e}</td>"
        f"<td>{s['signal_resid_rel_wide_only']:.2e}</td>"
        f"<td>{s['signal_resid_rel_all']:.2e}</td>"
        f"<td>{s['noise_var_ratio_wide_only_over_overlap']:.3f}</td>"
        f"<td>{s['noise_eff_overlap_rel']:.3f}</td>"
        f"<td>{s['time_min']:.2f}</td></tr>"
        for run, s in summaries.items()
    ]
    imgs = "".join(
        f"<figure><img src='{p}'><figcaption>{c}</figcaption></figure>" for c, p in figs
    )
    c0 = next(iter(configs.values()))
    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>Two-map NILC test</title>
<style>body{{font-family:sans-serif;max-width:1100px;margin:auto;padding:16px}}
table{{border-collapse:collapse;margin-bottom:1em}}
td,th{{border:1px solid #999;padding:3px 8px}}
img{{max-width:100%}}</style></head><body>
<h1>coberus needlet_coadd: two-map subset-footprint test</h1>
<p>wide (base tag): dec {WIDE[1]}, full RA. deep: dec {DEEP[1]}, RA {DEEP[2]}.
Same CMB and same beam (= output beam) in both maps, independent white noise,
{c0.taper_deg} deg cosine apodization, binary masks at apod &gt; 0.99,
res {c0.res_arcmin}'. Interior regions exclude {c0.edge_buffer} deg from mask
edges.</p>
<table><tr><th>config</th><th>noise wide [uK']</th><th>noise deep [uK']</th>
<th>fwhm [']</th><th>lpeaks</th><th>purpose</th></tr>{cfg_rows}</table>
<p>Expected in every config: w_deep = {expected["w_deep"]:.4f} in the overlap,
w_wide = 1 outside; noise variance ratio (wide-only / overlap) =
{expected["noise_var_ratio"]:.3f}; effective overlap noise / wide noise =
{expected["noise_eff_rel"]:.3f}.</p>
<h2>Per-scale weights</h2>
<table><tr><th>run</th><th>scale</th><th>w_deep overlap interior</th>
<th>w_deep within 2 deg inside deep edge</th><th>w_wide wide-only</th>
<th>max |sum w - 1|</th></tr>{"".join(rows)}</table>
<h2>Signal and noise</h2>
<table><tr><th>run</th><th>signal resid / CMB rms (overlap)</th>
<th>(wide-only)</th><th>(whole base mask)</th><th>noise var ratio</th>
<th>overlap noise / wide noise</th><th>time [min]</th></tr>
{"".join(glob_rows)}</table>
{imgs}</body></html>"""
    with open(fname, "w") as f:
        f.write(html)


def auto_fwhm(lmax, bl_min=0.01):
    """
    FWHM in arcminutes of the Gaussian beam with B_ell = bl_min at lmax.

    Args:
        lmax: Multipole at which the beam reaches bl_min.
        bl_min: Beam value at lmax.

    Returns:
        FWHM in arcminutes.
    """
    return np.rad2deg(np.sqrt(-16 * np.log(2) * np.log(bl_min)) / lmax) * 60


def main():
    """Generate the two-map datasets, run needlet_coadd and check outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="test_output", help="Plot/report dir.")
    parser.add_argument(
        "--out-root", default="/tmp/cob_two_", help="Intermediates prefix."
    )
    parser.add_argument("--res-arcmin", type=float, default=4.0)
    parser.add_argument("--lmax", type=int, default=500)
    parser.add_argument("--noise-wide", type=float, default=40.0, help="[uK-arcmin]")
    parser.add_argument("--noise-deep", type=float, default=20.0, help="[uK-arcmin]")
    parser.add_argument(
        "--high-noise-factor",
        type=float,
        default=50.0,
        help="Noise multiplier for the noise-dominated config.",
    )
    parser.add_argument("--taper-deg", type=float, default=3.0)
    parser.add_argument("--edge-buffer", type=float, default=5.0, help="[deg]")
    parser.add_argument("--cov-smooth-factor", type=int, default=16)
    parser.add_argument("--runs", nargs="+", default=["block", "gaussian", "tophat"])
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["baseline", "highnoise", "sharpcut"],
        help="baseline: realistic noise, beam with B(lmax)=0.01; highnoise: "
        "noise x high-noise-factor; sharpcut: 10' beam, signal at lmax.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true", help="Low-res quick run.")
    args = parser.parse_args()
    if args.debug:
        args.res_arcmin, args.lmax = 8.0, 250
        args.cov_smooth_factor = min(args.cov_smooth_factor, 8)
        args.workers = min(args.workers, 2)
    args.lpeaks = [lp for lp in [0, 50, 100, 200, 350] if lp < args.lmax] + [args.lmax]
    os.makedirs(args.output, exist_ok=True)

    nw2, nd2 = args.noise_wide**2, args.noise_deep**2
    expected = {
        "w_deep": nw2 / (nw2 + nd2),
        "noise_var_ratio": (nw2 + nd2) / nd2,
        "noise_eff_rel": np.sqrt(nd2 / (nw2 + nd2)),
    }
    print("expected:", expected)
    hn = args.high_noise_factor
    all_configs = {
        "baseline": {
            "noise_wide": args.noise_wide,
            "noise_deep": args.noise_deep,
            "fwhm": auto_fwhm(args.lmax),
            "note": "realistic noise; beam suppresses signal near lmax",
        },
        "highnoise": {
            "noise_wide": hn * args.noise_wide,
            "noise_deep": hn * args.noise_deep,
            "fwhm": auto_fwhm(args.lmax),
            "note": "noise-dominated at all scales: clean check of the ILC weights",
        },
        "sharpcut": {
            "noise_wide": args.noise_wide,
            "noise_deep": args.noise_deep,
            "fwhm": 10.0,
            "note": "beam ~1 at lmax: signal truncated sharply by the top needlet",
        },
    }
    configs = {
        name: argparse.Namespace(**{**vars(args), **all_configs[name]})
        for name in args.configs
    }
    summaries, figs = {}, []
    for name, cfg in configs.items():
        noises = {"wide": cfg.noise_wide, "deep": cfg.noise_deep}
        info, alm = make_two_maps(
            f"{args.out_root}{name}_dataset",
            cfg.res_arcmin,
            cfg.lmax,
            cfg.fwhm,
            noises,
            cfg.taper_deg,
            cfg.seed,
        )
        for run in args.runs:
            label = f"{name}_{run}"
            print(f"=== {label} ===")
            root = f"{args.out_root}{label}_"
            t0 = time.time()
            out = needlet_coadd(
                map_fname_func=lambda tag, info=info: info[tag]["map"],
                mask_fname_func=lambda tag, info=info: info[tag]["mask"],
                tags=["wide", "deep"],
                base_tag="wide",
                lpeaks=cfg.lpeaks,
                lmins=[None, None],
                lmaxs=[None, None],
                response_func=lambda tag: 1.0,
                beam_func=lambda tag, ells, fwhm=cfg.fwhm: gauss_beam(
                    np.asarray(ells), fwhm
                ),
                out_beam_fwhm=cfg.fwhm,
                out_root=root,
                cov_smooth_type=run,
                cov_smooth_factor=cfg.cov_smooth_factor,
                n_workers=cfg.workers,
                nmap_labels=["signal", "noise"],
                nmap_label_fname_func=lambda lab, tag, info=info: info[tag][lab],
                check_cmb_weights=True,
            )
            dt = (time.time() - t0) / 60.0
            s, f = analyze_run(out, root, info, alm, cfg, expected, label)
            s["time_min"] = dt
            summaries[label] = s
            figs += f
            print(json.dumps(s, indent=1))
            write_report(
                os.path.join(args.output, "two_map_report.html"),
                configs,
                expected,
                summaries,
                figs,
            )
    with open(os.path.join(args.output, "two_map_summary.json"), "w") as fh:
        json.dump({"expected": expected, "runs": summaries}, fh, indent=1)


if __name__ == "__main__":
    main()
