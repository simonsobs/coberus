from pixell import enmap, curvedsky as cs, wavelets as wv, uharm, wcsutils
import numpy as np
import os
from coberus import Coadder, coadd
from dask.distributed import Client
from contextlib import nullcontext
import healpy as hp
from collections import defaultdict
import time
import psutil


def gauss_beam(ell, fwhm):
    r"""
    Compute Gaussian beam window function $B_\ell$ for FWHM in arcminutes.

    Args:
        ell: Multipole moment(s) (float or ndarray).
        fwhm: Beam Full Width at Half Maximum in arcminutes.

    Returns:
        The beam attenuation factor $\exp(-\ell^2 \theta_{fwhm}^2 / (16 \ln 2))$.
    """
    tht_fwhm = np.deg2rad(fwhm / 60.0)
    return np.exp(-(tht_fwhm**2.0) * (ell**2.0) / (16.0 * np.log(2.0)))


def block_smooth(imap, factor, slow=False):
    """
    Smooth an enmap by block-averaging and resampling to the original geometry.

    Args:
        imap: Input enmap.
        factor: Integer block-downscaling factor.
        slow: If True, uses enmap.project; otherwise uses enmap.upgrade.

    Returns:
        The resampled enmap with original shape and WCS.
    """
    downed = enmap.downgrade(imap, factor, inclusive=True, op=np.nanmean)
    downed[np.isnan(downed)] = 0
    if slow:
        omap = enmap.project(downed, imap.shape, imap.wcs, order=0)
    else:
        omap = enmap.upgrade(downed, factor, inclusive=True, oshape=imap.shape)
    return omap


def band_filter(imap, fl, lmax, tol=1e-8):
    """
    Apply an isotropic harmonic filter with SHTs truncated to the filter's
    effective band-limit.

    Equivalent to pixell.curvedsky.filter(imap, fl, lmax=lmax) up to ~tol,
    but much faster for low-pass filters: both SHTs run at the largest
    multipole where the filter amplitude still exceeds tol times its peak,
    rather than at lmax, and modes where the filter is negligible are never
    analyzed.

    Args:
        imap: Input enmap.
        fl: 1D harmonic filter array.
        lmax: Maximum multipole of the untruncated filter operation.
        tol: Relative filter amplitude below which multipoles are dropped.

    Returns:
        The filtered enmap on the input geometry.
    """
    fl = np.asarray(fl[: lmax + 1])
    nz = np.where(np.abs(fl) > tol * np.abs(fl).max())[0]
    lcut = max(int(nz[-1]), 1) if nz.size else 1
    if lcut >= lmax:
        return cs.filter(imap, fl, lmax=lmax)
    alm = cs.almxfl(cs.map2alm(imap, lmax=lcut), fl[: lcut + 1])
    return cs.alm2map(alm, enmap.empty(imap.shape, imap.wcs, imap.dtype))


def free_mem():
    return f"{psutil.virtual_memory()[1] / 1024 / 1024 / 1024:.1f} GiB"


def get_scales(basis, tags, ellmins, ellmaxs):
    """
    Given a pixell.wavelets.basis wavelet basis
    object and minimum and maximum multipoles
    for datasets tagged by tags, return a dictionary
    mapping the tag names to a list of wavelet
    coefficient indices that we should calculate
    wavelet maps for.
    """
    ntags = len(ellmins)
    scales = {}
    if len(ellmaxs) != ntags:
        raise ValueError
    for tag, ellmin, ellmax in zip(tags, ellmins, ellmaxs):
        scales[tag] = []
        for i in range(basis.n):
            wlmin = basis.lmins[i]
            wlmax = basis.lmaxs[i]
            if ellmin is None:
                ellmin = 0
            if ellmax is None:
                ellmax = np.inf
            if ellmin > wlmin:
                continue
            if ellmax < wlmax:
                continue
            scales[tag].append(i)
    return scales


def compute_tophat_beam(w_rad, lmax, w_rad_in=None, n_theta=20000):
    """
    Computes harmonic space beam for top hat filter used in 2307.01258.

    Parameters
    ----------

    w_rad: float
        Smoothing scale in radians

    lmax: int
        Maximum multipole

    w_rad_in: float, optional
        Optional float that sets a second filter scale and excludes modes below
        this filter scale. Used for choosing an annulus in angular space

    n_theta: float, optional
        Number of theta bins to use when computing the angular smoothing
        function.

    Returns
    -------

    filt_harm: float array
        Harmonic-space filter beam for multipoles up to lmax.

    """

    theta_max = 10 * w_rad
    theta = np.linspace(0, theta_max, n_theta)
    filt_ang = 1 / ((1 + (theta / w_rad)) ** 6)

    if w_rad_in is not None:
        filt_ang -= 1 / ((1 + (theta / w_rad_in)) ** 6)

    filt_harm = hp.sphtfunc.beam2bl(filt_ang, theta, int(lmax))
    filt_harm /= filt_harm[0]

    return filt_harm


def cov_filter(cov_smooth_type, sigma_rad, lmax, use_annulus, annulus_fwhm_ratio):
    """
    Build the harmonic filter used for SHT-based covariance smoothing.

    Args:
        cov_smooth_type: 'gaussian' (procedure from 2307.01043) or 'tophat'
            (procedure from 2307.01258).
        sigma_rad: Smoothing scale in radians.
        lmax: Maximum multipole.
        use_annulus: For 'tophat', whether to exclude modes below a second
            inner filter scale.
        annulus_fwhm_ratio: Ratio of the inner annulus scale to sigma_rad.

    Returns:
        1D harmonic filter array up to lmax.
    """
    if cov_smooth_type == "gaussian":
        ells = np.arange(lmax + 1)
        return np.exp(-0.5 * ells * (ells + 1) * sigma_rad**2)
    w_rad_in = annulus_fwhm_ratio * sigma_rad if use_annulus else None
    return compute_tophat_beam(sigma_rad, lmax, w_rad_in=w_rad_in)


def cov_smooth(
    wmap1, wmap2, cov_smooth_type, cov_smooth_factor, sigma_rad, fft_smooth, lmax, fl
):
    """
    Smooth the product of two wavelet maps into an empirical covariance map.

    Mean subtraction (smooth_mean_cov) is handled by the caller, which passes
    mean-subtracted delta maps in place of the raw wavelet maps; this function
    only smooths the product.

    Args:
        wmap1: First wavelet enmap (or its mean-subtracted delta).
        wmap2: Second wavelet enmap (or its mean-subtracted delta).
        cov_smooth_type: 'block', 'gaussian' or 'tophat'.
        cov_smooth_factor: Block downgrade factor for 'block' smoothing.
        sigma_rad: Smoothing scale in radians (unused for 'block').
        fft_smooth: For 'gaussian', smooth with FFTs instead of SHTs.
        lmax: Maximum multipole for SHT-based smoothing.
        fl: Harmonic filter from cov_filter; None for the block/FFT paths.

    Returns:
        The smoothed covariance enmap.
    """
    prod = wmap1 * wmap2
    if cov_smooth_type == "block":
        return block_smooth(prod, cov_smooth_factor, slow=False)
    if cov_smooth_type == "gaussian" and fft_smooth:
        # Gaussian smoothing procedure from 2307.01043 using FFTs.
        return enmap.smooth_gauss(prod, sigma_rad)
    # SHT-based gaussian (2307.01043) or tophat (2307.01258, Eq. 11)
    # smoothing, with the SHTs truncated to the filter's band-limit.
    return band_filter(prod, fl, lmax)


def project_mask(imask, oshape, owcs, threshold=0.99):
    """Safely reproject a binary mask, keeping the result 0/1.
    If the geometries are pixel-compatible, use enmap.extract, which copies
    values exactly with no interpolation. Otherwise bilinear-project and
    re-binarize.
    """
    if wcsutils.is_compatible(imask.wcs, owcs):
        return enmap.extract(imask, oshape, owcs)
    return 1.0 * (enmap.project(imask, oshape, owcs, order=1) > threshold)


def get_noise_realization(
    map_fname_func,
    ivar_fname_func,
    tags,
    base_tag,
    map_coadd=None,
    ivar_coadd=None,
    seed=None,
):
    """
    Create a noise map realization from a list of split maps by pair
    differencing.

    Args:
        map_fname_func : func
            Accepts the tag name and returns a path to the input split map.

        ivar_fname_func : func
            Accepts the tag name and returns a path to the input inverse
            variance split map.

        tags : list
            Ordered list of tags for split maps.

        base_tag : str
            Name of the tag whose geometry all other tags are extracted
            to.

        seed : optional,int
            Random seed for coeffcients of linear sum of difference maps.

        map_coadd : optional,enmap
            Inverse variance weighted coadded map. If not specified, will be
            calculated from the map splits.

        ivar_coadd : optional,enmap
            Coadded inverse variance. If not specified, will be calculated
            from the map splits.

    Returns:
        noise : enmap
            The noise map realization.
    """

    shape, wcs = enmap.read_map_geometry(map_fname_func(base_tag))

    assert len(tags) >= 2, "Need at least two tags"

    if map_coadd is None or ivar_coadd is None:
        # Build the inverse variance weighted coadded map from the splits if
        # not provided
        map_coadd = enmap.zeros(shape, wcs)
        ivar_coadd = enmap.zeros(shape, wcs)

        for i, tag in enumerate(tags):
            map_i = enmap.read_map(map_fname_func(tag))
            ivar_i = enmap.read_map(ivar_fname_func(tag))

            if tag != base_tag:
                map_i = enmap.extract(map_i, shape, wcs)
                ivar_i = enmap.extract(ivar_i, shape, wcs)

            map_coadd += ivar_i * map_i
            ivar_coadd += ivar_i

        mask = ivar_coadd > 0
        map_coadd[mask] = map_coadd[mask] / ivar_coadd[mask]

    rng = np.random.default_rng(seed)

    noise = enmap.zeros(shape, wcs)

    for i, tag in enumerate(tags):
        map_i = enmap.read_map(map_fname_func(tag))
        ivar_i = enmap.read_map(ivar_fname_func(tag))

        if tag != base_tag:
            map_i = enmap.extract(map_i, shape, wcs)
            ivar_i = enmap.extract(ivar_i, shape, wcs)

        diff_i = (map_i - map_coadd)

        # Sign flip
        coeff = rng.choice([-1, 1])
        noise += coeff * ivar_i * diff_i

    noise[mask] *= 1. / ivar_coadd[mask]

    return noise


def needlet_coadd(
    map_fname_func,
    mask_fname_func,
    tags,
    base_tag,
    lpeaks,
    lmins,
    lmaxs,
    response_func,
    beam_func,
    out_beam_fwhm,
    out_root,
    oshape=None,
    owcs=None,
    deproj_response_funcs=None,
    noise_map_fname_func=None,
    cov_smooth_type="block",
    cov_smooth_factor=64,
    ilc_bias_tol=0.01,
    fft_smooth=False,
    smooth_mean_cov=True,
    cov_smooth_scales=None,
    use_annulus=False,
    annulus_fwhm_ratio=0.5,
    map_postprocess_func=None,
    mask_postprocess_func=None,
    client=None,
    n_workers=None,
    io_suffix="",
    delete_intermediate=False,
    nmap_labels=None,
    nmap_label_fname_func=None,
    apply_mask=False,
):
    """
    Generic function for coadding maps using an empirical
    covariance determined from smoothed products of maps in a
    needlet basis. Each input map is identified
    by a string called 'tag'. The properties of these maps are specified
    through functions of the tag name.

    However, if nmap_label_fname_func are provided, covariances +
    weights estimated from the data map are then used to coadd the
    "nmap"s and they are returned as well, formatted as:
    output[f'{nmap_label}_coadd'] for [nmap_label,...] in nmap_labels.

    Parameters
    ----------

    map_fname_func : func
        Accepts the tag name and returns a path to the input map.
        We recommend you mask and apodize these maps.

    mask_fname_func : func
        Accepts the tag name and returns a path to *binary* masks
        for each tag. These binary masks denote which regions of
        each map to include in the final coadd. We recommend these
        masks exclude regions where the apodization of the input
        maps drops below some threshold, say 0.99.

    tags : list
        Ordered list of tags to coadd

    base_tag : str
        Name of the tag whose geometry all other tags are extracted
        to and whose mask is used as the final mask.

    lpeaks : list
        List of peak multipoles that define a cosine needlet basis

    lmins : list
        List of minimum multipoles beyond which a tag is not used
        in a wavelet scale

    lmaxs : list
        List of maximum multipoles beyond which a tag is not used
        in a wavelet scale

    response_func : func
        Accepts the tag name and returns the map response value.
        Use lambda x: 1 for the CMB solution.

    beam_func : func
        Accepts the tag name and the multipole as arguments and
        returns the value of the beam.

    out_beam_fwhm : float
        FWHM in arcminutes for the beam of the final map

    out_root : str
        Root path for intermediate outputs. We highly recommend
        using a RAMdisk (/dev/shm/) if you have enough memory.
        This will significantly speed up the code. Remember to
        append a unique code for each job if you have multiple
        jobs sharing memory on a node.
        e.g. out_root = '/dev/shm/sim_1_'

    oshape : optional, tuple
        Shape of the output map geometry. If both oshape and owcs are
        provided (e.g. a downgraded version of the base_tag geometry), the
        final coadded map(s) are reconstructed onto this geometry. If either
        is None, the output geometry defaults to the base_tag geometry.

    owcs : optional, astropy.wcs.WCS
        WCS of the output map geometry. See oshape.

    deproj_response_funcs : list of funcs
        List of response functions to deproject. Each function should accepts
        the tag name and return the map response value.

    noise_map_fname_func : optional,func
        Accepts the tag name and returns a path to the input noise map for
        covariance calculation. If not specified, input maps will be used to
        estimate per-pixel noise.

    cov_smooth_type : optional,str
        Type of smoothing to use when constructing the covariance. Current
        options are 'block', 'gaussian', and 'tophat'. Default value is 'block'.

    cov_smooth_factor : optional,int
        Factor by which to block downgrade the covariance maps if using block
        smoothing.

    ilc_bias_tol : optional, float
        Bias tolerance used to determine FWHM for Gaussian smoothing
        of covariance if using Gaussian smoothing. Based on Eq. 43 of
        2307.01043.

    fft_smooth : optional, boolean
        If true, use FFT's instead of SHT's to smooth the map. Default
        option is to use SHT's.

    smooth_mean_cov: optional, boolean
        Only used for gaussian or top-hat smoothing.
        If True, computes the covariance from <(A-A_smooth)(B-B_smooth)>, as
        in pyilc. Otherwise, computes the covariance from <AB>. Default is True.

    map_postprocess_func : optional,func
        A function to apply to each loaded map

    mask_postprocess_func : optional,func
        A function to apply to each loaded mask

    n_workers : optional,int
        Number of workers for distributed Dask tasks

    delete_intermediate : optional,bool
        Whether to delete intermediate outputs

    nmap_labels : optional,list[str]
        List of possible optional maps' labels

    nmap_label_fname_func : optional,func | (nmap_label, fname) -> nmap
        Optional maps not used for covariance, but coadded with
        the same weights. Accepts the optional map's label and filename
        and returns a path

    apply_mask : optional, boolean
        If true, zero out regions of the input maps based on their masks.
        This will make the maps masked sharply before wavelet transforms,
        and is hence not recommended.



    Returns
    -------

    outmaps : dict
        Dictionary of output ndmaps. Always contains:
          'coadd' : final coadded map on the output geometry.
          'mask'  : the final footprint mask (the base_tag mask, projected
                    onto the output geometry when one is provided).
        For each label in nmap_labels, also contains '{label}_coadd' with the
        coadd of those maps using the same weights.

    """
    if nmap_labels is None:
        nmap_labels = []

    # Sanity check that the output dictionary keys we will produce are unique
    _output_keys = ["coadd"] + [f"{label}_coadd" for label in nmap_labels] + ["mask"]
    if len(_output_keys) != len(set(_output_keys)):
        raise ValueError(
            f"Duplicate keys would be produced in output dictionary: {_output_keys}. "
            "Check nmap_labels for collisions with 'coadd' or 'mask'."
        )
    # These would collide with internal wavelet_{map,mask,delta}_* file names
    _reserved = {"map", "mask", "delta", "noise_map"} & set(nmap_labels)
    if _reserved:
        raise ValueError(f"nmap_labels uses reserved label(s): {sorted(_reserved)}")

    start_time = time.time()
    lmax = max(lpeaks)  # Cosine needlets have zero support beyond lpeak
    ells = np.arange(lmax)
    shape, wcs = enmap.read_map_geometry(map_fname_func(base_tag))
    n_deproj = len(deproj_response_funcs) if (deproj_response_funcs is not None) else 0

    # Compute fsky from base mask. This is used to determine covariance smoothing scales
    base_mask = enmap.read_map(mask_fname_func(base_tag))
    fsky = (
        (np.sum(base_mask**2) / np.prod(base_mask.shape))
        * base_mask.area()
        / (4 * np.pi)
    )
    print("fsky={:.2f}".format(fsky))

    # Initialize Wavelets
    uht = uharm.UHT(shape, wcs, mode="curved")
    basis = wv.CosineNeedlet(lpeaks=lpeaks)
    scales = get_scales(basis, tags, lmins, lmaxs)
    nwaves = basis.n
    wt = wv.WaveletTransform(uht, basis=basis)

    # Optional separate output geometry (e.g. a downgrade of base_tag). Only
    # the final wave2map reconstruction uses this; per-scale wavelet
    # geometries and the wavelet-domain coadd are unchanged.
    if (oshape is None) or (owcs is None):
        wt_out = wt
        oshape, owcs = shape, wcs
    else:
        uht_out = uharm.UHT(oshape, owcs, mode="curved")
        wt_out = wv.WaveletTransform(uht_out, basis=basis)

    # Compute number of tags used at each needlet scale
    n_tag_per_scale = np.sum(
        [np.bincount(scales[tag], minlength=basis.n) for tag in tags], axis=0
    )

    # if using optional additional maps to coadd
    do_nmaps = len(nmap_labels) > 0 and (nmap_label_fname_func is not None)

    def _get_wave(fname_func, itag, imask):
        gmap = enmap.read_map(fname_func(itag))

        if map_postprocess_func is not None:
            gmap = map_postprocess_func(gmap)
        if itag != base_tag:
            gmap = enmap.extract(gmap, shape, wcs)

        if apply_mask:
            gmap[imask == 0] = 0
        # Reconvolve to common beam
        out_beam = gauss_beam(ells, out_beam_fwhm)
        in_beam = beam_func(itag, ells)
        if in_beam.ndim != 1:
            raise ValueError
        beam_ratio = out_beam / in_beam
        wavecs = wt.map2wave(
            gmap, fl=beam_ratio, scales=scales[itag], fill_value=np.nan
        )
        return wavecs

    # These will hold file names for maps, masks and covariance maps
    # for use by the Coberus coadder
    fmasks = defaultdict(list)
    fmaps = defaultdict(list)
    fcovs = {}
    filenames = []
    # store additional maps if desired
    if do_nmaps:
        nfmaps = {label: defaultdict(list) for label in nmap_labels}

    print(f"Free memory: {free_mem()}")
    totgibytes = 0.0

    # Covariance smoothing
    cov_smooth_types = ["block", "gaussian", "tophat"]
    assert cov_smooth_type in cov_smooth_types, (
        f"Unrecognized covariance smoothing type.  Available options are: {', '.join(cov_smooth_types)}"
    )

    ############################################################################
    # Determine smoothing scales
    ############################################################################
    # If the user does not input a set of smoothing scales for the covariance
    # calculation, then we comptue them here. For gaussian smoothing, we compute
    # the scales from the ILC bias tolerance (see 2307.01043). For tophat
    # filters, we use Eq. 13 of 2307.01258.
    #
    # In principle, these calculations should account for the fact that the
    # number of maps that contribute to each pixel can vary over the sky. This
    # would require introducing an anisotropic smoothing. Instead, we use the
    # total number of maps considered in the ILC at a given needlet scale.
    # This is conservative -- it overestimates the smoothing scale needed for a
    # given ILC bias tolerance in regions that are only covered by a subset of
    # the input maps.

    if cov_smooth_scales is None:
        if cov_smooth_type == "gaussian":
            print(
                f"Covariance smoothing scales not specified. Determining scales with ILC bias tolerance of {ilc_bias_tol}"
            )
            n_modes_eff = (
                np.asarray(
                    [
                        np.sum((2 * ells + 1) * basis(i, ells) ** 2)
                        for i in range(basis.n)
                    ]
                )
                * fsky
            )
            n_freq_eff = n_tag_per_scale

            cov_smooth_scales = np.sqrt(
                2 * abs(1 + n_deproj - n_freq_eff) / (ilc_bias_tol * n_modes_eff)
            )  # Radians

            assert all(cov_smooth_scales < np.pi), (
                "Not enough modes to satisfy ILC bias tolerance."
            )

        elif cov_smooth_type == "tophat":
            # Note that this differs from Nmodes used in Gaussian smoothing. Here, we approximate the
            # needlets as top-hats in harmonic space. The number of modes used in the Gaussian smoothing

            # Note that this implementation is slightly different than the one
            # in 2307.01258 (https://github.com/ACTCollaboration/NILC/blob/main/NILC/ilc.py)
            # which uses:
            # n_modes_eff = np.asarray([np.sum((2*ells+1)*np.where(basis(i, ells) !=0 , 1, 0))
            #                         for i in range(basis.n)])

            n_modes_eff = (
                np.asarray(
                    [
                        np.sum((2 * ells + 1) * basis(i, ells) ** 2)
                        for i in range(basis.n)
                    ]
                )
                * fsky
            )
            n_freq_eff = n_tag_per_scale

            # Eq 13 of 2307.01258
            arccos_arg = 1.0 - 2.0 * (20 * n_freq_eff / n_modes_eff)
            arccos_arg[arccos_arg < -1] = -1
            cov_smooth_scales = 2 * np.arccos(arccos_arg)

    else:
        assert len(cov_smooth_scales) == nwaves, (
            "Number of covariance smoothing scales does not match number of wavelets"
        )

    start_time_wavelets = time.time()

    # Loop through arrays
    for i, tag in enumerate(tags):
        mask = enmap.read_map(mask_fname_func(tag))
        if mask_postprocess_func is not None:
            mask = mask_postprocess_func(mask)
        if tag != base_tag:
            mask = enmap.extract(mask, shape, wcs)
        else:
            base_mask = mask

        wavecs = _get_wave(map_fname_func, tag, mask)
        nwavecs = {
            label: _get_wave(
                lambda fname: nmap_label_fname_func(label, fname), tag, mask
            )
            for label in nmap_labels
        }

        if noise_map_fname_func is not None:
             noise_wavecs = _get_wave(noise_map_fname_func, tag, mask)

        if i == 0:
            # Save multimap template for final coadded map
            owave = wavecs * 0.0

        # Loop through wavelet scales
        for j, wmap in enumerate(wavecs.maps):
            if j not in scales[tags[i]]:
                continue
            print("Projecting mask and writing wavelet map...")
            # Project masks on to wavelet map geometries
            omask = project_mask(mask, wmap.shape, wmap.wcs)

            mfname = f"{out_root}wavelet_mask_{tags[i]}_scale_{j}.fits"
            filenames.append(mfname)
            fmasks[j].append(mfname)
            enmap.write_map(mfname, omask)

            wfname = f"{out_root}wavelet_map_{tags[i]}_scale_{j}.fits"
            filenames.append(wfname)
            fmaps[j].append(wfname)
            enmap.write_map(wfname, wmap)

            totgibytes = totgibytes + (wmap.nbytes / 1024 / 1024.0 / 1024.0 * 2.0)

            # optional maps
            for label in nmap_labels:
                nwfname = f"{out_root}wavelet_{label}_{tags[i]}_scale_{j}.fits"
                filenames.append(nwfname)
                nfmaps[label][j].append(nwfname)
                enmap.write_map(nwfname, nwavecs[label].maps[j])
                totgibytes = totgibytes + (wmap.nbytes / 1024 / 1024.0 / 1024.0)

        if noise_map_fname_func is not None:
            # Loop through wavelet scales for noise maps
            for j, wmap in enumerate(noise_wavecs.maps):
                if j not in scales[tags[i]]:
                    continue

                wfname = f"{out_root}wavelet_noise_map_{tags[i]}_scale_{j}.fits"
                filenames.append(wfname)
                enmap.write_map(wfname, wmap)

                totgibytes = totgibytes + (wmap.nbytes / 1024 / 1024.0 / 1024.0)


    elapsed_time = time.time() - start_time_wavelets
    print(f"Wavelets finished in {elapsed_time / 60.0:.2f} minutes.")
    print(f"Total disk: {totgibytes:.1f} GiB")
    print(f"Free memory: {free_mem()}")
    print("Building covariance...")
    start_time_covariance = time.time()
    included_tags = {}
    for k in range(nwaves):
        fcovs[k] = [[""] * len(fmaps[k]) for h in range(len(fmaps[k]))]
        # Tags to be included in wavelet scale
        itags = []
        for i, tag in enumerate(tags):
            if k not in scales[tag]:
                continue
            itags.append(tag)
        included_tags[k] = list(itags)

        if cov_smooth_type == "block":
            sigma_rad = None
        else:
            sigma_rad = cov_smooth_scales[k]

        # Build the harmonic smoothing filter once per scale (the tophat
        # beam in particular is expensive to compute). fl is None for the
        # paths that do not use SHT filtering.
        if cov_smooth_type == "block" or (cov_smooth_type == "gaussian" and fft_smooth):
            fl = None
        else:
            fl = cov_filter(
                cov_smooth_type, sigma_rad, lmax, use_annulus, annulus_fwhm_ratio
            )

        # For mean-subtracted covariances, smooth each tag's wavelet map once
        # here rather than once per pair inside cov_smooth; smoothing the pair
        # products of the deltas then gives the same covariance.
        hoist = cov_smooth_type != "block" and smooth_mean_cov
        dfnames = []
        if hoist:
            for tag in itags:
                if noise_map_fname_func is None:
                    wmap = enmap.read_map(f"{out_root}wavelet_map_{tag}_scale_{k}.fits")
                else:
                    wmap = enmap.read_map(f"{out_root}wavelet_noise_map_{tag}_scale_{k}.fits")
                if fl is None:
                    smap = enmap.smooth_gauss(wmap, sigma_rad)
                else:
                    smap = band_filter(wmap, fl, lmax)
                dfname = f"{out_root}wavelet_delta_{tag}_scale_{k}.fits"
                enmap.write_map(dfname, wmap - smap)
                dfnames.append(dfname)
        prefix = "delta" if hoist else "map" if noise_map_fname_func is None else "noise_map"

        for i in range(len(itags)):
            wmap1 = enmap.read_map(
                f"{out_root}wavelet_{prefix}_{itags[i]}_scale_{k}.fits"
            )

            for j in range(i, len(itags)):
                if i == j:  # Potentially minor speedup?
                    wmap2 = wmap1
                else:
                    wmap2 = enmap.read_map(
                        f"{out_root}wavelet_{prefix}_{itags[j]}_scale_{k}.fits"
                    )

                cov = cov_smooth(
                    wmap1,
                    wmap2,
                    cov_smooth_type,
                    cov_smooth_factor,
                    sigma_rad,
                    fft_smooth,
                    lmax,
                    fl,
                )

                fcovname = f"{out_root}wavelet_cov_scale_{k}_{itags[i]}_{itags[j]}.fits"
                fcovs[k][i][j] = fcovname
                fcovs[k][j][i] = fcovname
                enmap.write_map(fcovname, cov)
                filenames.append(fcovname)
                totgibytes = totgibytes + (cov.nbytes / 1024 / 1024.0 / 1024.0)

        # The delta maps are only needed within this scale, so free the
        # (RAM)disk space immediately rather than at the end of the run.
        for dfname in dfnames:
            os.remove(dfname)

    elapsed_time = time.time() - start_time_covariance
    print(f"Covariance finished in {elapsed_time / 60.0:.2f} minutes.")

    print(f"Total disk: {totgibytes:.1f} GiB")
    print(f"Free memory: {free_mem()}")

    # Project the final footprint mask onto the output geometry (no-op when
    # output geometry equals the base_tag geometry).
    if wt_out is wt:
        out_base_mask = base_mask
    else:
        out_base_mask = project_mask(base_mask, oshape, owcs)

    outmaptypes = ["coadd"] + [label + "_coadd" for label in nmap_labels]
    outmaps = {}

    client_ctx = Client(n_workers=n_workers) if client is None else nullcontext(client)
    with client_ctx as client:
        for outmaptype in outmaptypes:
            # This part uses Coberus to do distributed Dask
            # pixel-space coadding of the maps for each
            # wavelet scale
            for j in range(nwaves):
                print(f"Coadding {outmaptype} scale {j}...")
                if outmaptype == "coadd":
                    lmaps = fmaps[j]
                else:
                    lmaps = nfmaps[outmaptype[:-6]][j]

                masks = fmasks[j]
                covs = fcovs[j]
                responses = [response_func(tag) for tag in included_tags[j]]

                if n_deproj == 0:
                    deproj_responses = []
                else:
                    deproj_responses = [
                        [deproj_func_i(tag) for tag in included_tags[j]]
                        for deproj_func_i in deproj_response_funcs
                    ]  # N_deproj x N_freq

                coadder = Coadder(
                    maps=lmaps,
                    masks=masks,
                    covariance_maps=covs,
                    responses=responses,
                    deproj_responses=deproj_responses,
                )

                print("Number of workers: ", len(client.scheduler_info()["workers"]))
                # Result is a dask array
                result = coadd(client, coadder)
                # This is now a numpy array
                arr = result.compute()
                owave.maps[j] = enmap.enmap(arr.copy(), owave.maps[j].wcs)

            coadd_map = wt_out.wave2map(owave)
            coadd_map[out_base_mask == 0] = 0
            outmaps[outmaptype] = coadd_map.copy()

    print(f"Free memory: {free_mem()}")
    outmaps["mask"] = out_base_mask
    if delete_intermediate:
        for filename in filenames:
            os.remove(filename)
    elapsed_time = time.time() - start_time
    print(f"Done in {elapsed_time / 60.0:.2f} minutes.")
    return outmaps
