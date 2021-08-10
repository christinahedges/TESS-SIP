import lightkurve as lk
import numpy as np
from scipy import sparse
from tqdm import tqdm
import warnings

from astropy.timeseries import lombscargle


def vstack(dms):
    """Custom vertical stack script to stack lightkurve design matrices"""
    npoints = np.sum([dm.shape[0] for dm in dms])
    ncomps = np.sum([dm.shape[1] for dm in dms])
    if sparse.issparse(dms[0].X):
        X = sparse.lil_matrix((npoints, ncomps))
    else:
        X = np.zeros((npoints, ncomps))
    idx = 0
    jdx = 0
    for dm in dms:
        X[idx : idx + dm.shape[0], jdx : jdx + dm.shape[1]] += dm.X
        idx = idx + dm.shape[0]
        jdx = jdx + dm.shape[1]
    prior_mu = np.hstack([dm.prior_mu for dm in dms])
    prior_sigma = np.hstack([dm.prior_sigma for dm in dms])
    name = dms[0].name
    if sparse.issparse(dms[0].X):
        return lk.correctors.SparseDesignMatrix(
            X.tocsr(), name=name, prior_mu=prior_mu, prior_sigma=prior_sigma
        )
    else:
        return lk.correctors.DesignMatrix(
            X, name=name, prior_mu=prior_mu, prior_sigma=prior_sigma
        )


def prepare_tpfs(data, npca_components=3, aperture_threshold=3):
    """Creates a dataset for TESS-SIP

    Parameters:
    -----------
    data: lk.TargetPixelFileCollection
        Collection of target pixel files create a SIP
    npca_components: int
        Number of principle components to use for background

    Returns:
    -----------
    lc : lk.LightCurve
        The light curve to create a TESS SIP of
    lc_bkg: : lk.LightCurve
        The light curve of the background pixels
    data_uncorr: list
        List of the input TPFs, with the scattered light
        re added. (TESS Pipeline removes it)
    bkgs : list
        List of design matrices containing the background information.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Get the un-background subtracted data
        if hasattr(data[0], "flux_bkg"):
            data_uncorr = [
                (tpf + np.nan_to_num(tpf.flux_bkg.value))[
                    np.isfinite(np.nansum(tpf.flux_bkg.value, axis=(1, 2)))
                ]
                for tpf in data
            ]
        else:
            data_uncorr = [tpf for tpf in data]

        apers = [
            tpf.pipeline_mask
            if tpf.pipeline_mask.any()
            else tpf.create_threshold_mask(aperture_threshold)
            for tpf in data_uncorr
        ]
        bkg_apers = [
            (~aper) & (np.nansum(tpf.flux, axis=0) != 0)
            for aper, tpf in zip(apers, data_uncorr)
        ]
        lc = (
            lk.LightCurveCollection(
                [
                    tpf.to_lightcurve(aperture_mask=aper)
                    for tpf, aper in zip(data_uncorr, apers)
                ]
            )
            .stitch(lambda x: x)
            .normalize()
        )
        lc.flux_err.value[~np.isfinite(lc.flux_err.value)] = np.nanmedian(
            lc.flux_err.value
        )

        # Run the same routines on the background pixels
        lc_bkg = (
            lk.LightCurveCollection(
                [
                    tpf.to_lightcurve(aperture_mask=bkg_aper)
                    for tpf, bkg_aper in zip(data_uncorr, bkg_apers)
                ]
            )
            .stitch(lambda x: x)
            .normalize()
        )
        lc_bkg.flux_err.value[~np.isfinite(lc_bkg.flux_err.value)] = np.nanmedian(
            lc_bkg.flux_err.value
        )

        bkgs = [
            lk.correctors.DesignMatrix(
                np.nan_to_num(tpf.flux.value[:, bkg_aper]), name="bkg"
            )
            .pca(npca_components)
            .append_constant()
            .to_sparse()
            for tpf, bkg_aper in zip(data_uncorr, bkg_apers)
        ]
        for bkg in bkgs:
            bkg.prior_mu[-1] = 1
            bkg.prior_sigma[-1] = 0.1
            bkg.prior_mu[:-1] = 0
            bkg.prior_sigma[:-1] = 0.1

        # Split at the datadownlink
        bkgs = [
            bkg.split(list((np.where(np.diff(tpf.time.jd) > 0.3)[0] + 1)))
            for bkg, tpf in zip(bkgs, data_uncorr)
        ]
    return lc, lc_bkg, data_uncorr, bkgs


def prepare_lcfs(data):
    """Creates a dataset for TESS-SIP

    Parameters:
    -----------
    data: lk.LightCurveCollection
        Collection of light curve files create a SIP

    Returns:
    -----------
    lc : lk.LightCurve
        The light curve to create a TESS SIP of
    data_uncorr: list
        List of the input LCFs, with the scattered light
        re added. (TESS Pipeline removes it)
    bkgs : list
        List of design matrices containing the background information.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for lcf in data:
            lcf.remove_nans(column="sap_flux")
            lcf.remove_nans(column="sap_bkg")
            lcf.flux = lcf.sap_flux
            lcf.flux_err = lcf.sap_flux_err
        data_uncorr = [
            (lcf + np.nan_to_num(lcf.sap_bkg))[np.isfinite(lcf.sap_bkg)] for lcf in data
        ]

        lc = lk.LightCurveCollection(data_uncorr).stitch(lambda x: x).normalize()

        # lc_bkg = lk.LightCurveCollection(data_uncorr).stitch(lambda x: x)
        # lc_bkg.flux = lc_bkg.sap_bkg
        # lc_bkg.flux_err = lc_bkg.sap_bkg_err
        # lc_bkg = lc_bkg.normalize()

        bkgs = [
            lk.correctors.DesignMatrix(np.nan_to_num(lcf.sap_bkg.value), name="bkg")
            .append_constant()
            .to_sparse()
            for lcf in data_uncorr
        ]
        for bkg in bkgs:
            bkg.prior_mu[-1] = 1
            bkg.prior_sigma[-1] = 0.1
            bkg.prior_mu[:-1] = 0
            bkg.prior_sigma[:-1] = 0.1

        # Split at the datadownlink
        bkgs = [
            bkg.split(list((np.where(np.diff(tpf.time.jd) > 0.3)[0] + 1)))
            for bkg, tpf in zip(bkgs, data_uncorr)
        ]
    return lc, data_uncorr, bkgs


def fit_model(lc, dm, sigma_f_inv, mask=None, return_model=False):
    if mask is None:
        mask = np.ones(len(lc.flux.value), bool)
    sigma_w_inv = dm.X[mask].T.dot(dm.X[mask].multiply(sigma_f_inv[mask])).toarray()
    sigma_w_inv += np.diag(1.0 / dm.prior_sigma ** 2)

    B = dm.X[mask].T.dot((lc.flux.value[mask] / lc.flux_err.value[mask] ** 2))
    B += dm.prior_mu / dm.prior_sigma ** 2
    w = np.linalg.solve(sigma_w_inv, B)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        werr = ((np.linalg.inv(sigma_w_inv)) ** 0.5).diagonal()
    if return_model:
        return dm.X.dot(w)
    return w, werr


def SIP(
    data,
    sigma=5,
    min_period=10,
    max_period=100,
    nperiods=300,
    npca_components=2,
    aperture_threshold=3,
    sff=False,
    sff_kwargs={},
    periods=None,
):
    """
    Systematics-insensitive periodogram for finding periods in long period NASA's TESS data.

    SIP can be used to find the best fitting sinusoid period in long period TESS data, while
    mitigating the instrument and scattered light background systematics.

    A description of the concepts behind a SIP is given here in the context of K2 data:
    https://ui.adsabs.harvard.edu/abs/2016ApJ...818..109A/abstract

    Parameters
    ----------
    data : lightkurve.TargetPixelFileCollection or lk.collections.LightCurveCollection
        A collection of target pixel files or light curve files from the TESS mission.
        This can be generated using lightkurve's search functions, for example:
            tpfs = lk.search_targetpixelfile('TIC 288735205', mission='tess').download_all()
            OR
            lcfs = lk.search_lightcurve('TIC 288735205', mission='tess').download_all()
    sigma : int or float
        SIP will run a single first iteration at a period of 27 days to remove significant
        outliers. Set sigma to a value, above which outliers will be clipped
    min_period : float
        The minimum period for the periodogram
    max_period : float
        The maximum period for the periodogram
    nperiods : int
        The number of periods to fit
    npca_components : int
        Number of pca components to detrend with. Default is 2.
    aperture_threshold : float
        If there is no aperture mask from the pipeline, will create one. Set
        aperture_threshold to set the thresholding for the aperture creation.
        (See lightkurve's create_threshold_mask function.)
    sff : boolean
        Whether to run SFF detrending simultaneously. This is most useful for K2 data.
        When True, will run SFF detrending.
    sff_kwargs : dict
        Dictionary of SFF key words to pass. See lightkurve's SFFCorrector.
    periods : None or numpy.ndarray
        A list of specific periods to use when evaluating the periodogram. If this
        parameter is not None, then the parameters min_period, max_period, and
        nperiods will be ignored.

    Returns
    -------
    r : dict
        Dictionary containing the following entries:
            periods: the periods evaluated
            power: the power at each period (definite as the amplitude of the sinusoid)
            raw_lc: the original light curve from the input target pixel files
            corr_lc: the light curve with the best fitting systematics removed
            period_at_max_power: the best fit period of the sinusoid.
            power_bkg: the power at each period for the pixels -outside- the aperture
            raw_lc_bkg: the background light curve (pixels outside aperture)
            model: the systematics model used to correct the light curve
    """

    # Checking if input data is correct type
    if isinstance(data, lk.TargetPixelFileCollection):
        if not all([isinstance(tpf, lk.TessTargetPixelFile) for tpf in data]):
            raise TypeError(
                """The list of objects within the input data type
                lightkurve.TargetPixelFileCollection must be all be type:
                lightkurve.TessTargetPixelFile"""
            )
    elif isinstance(data, lk.LightCurveCollection):
        if not all([isinstance(lcf, lk.TessLightCurve) for lcf in data]):
            raise TypeError(
                """The list of objects within the input data type
                lightkurve.LightCurveCollection must be all be type:
                lightkurve.TessLightCurve"""
            )
    else:
        raise TypeError(
            """The data input must be a collection of target pixel files
            or light curve files of type:
            lightkurve.TargetPixelFileCollection
            OR
            lightkurve.LightCurveCollection"""
        )

    # Setup data
    # Setup for when input data is a collection of target pixel files
    if isinstance(data, lk.TargetPixelFileCollection):
        lc, lc_bkg, data_uncorr, bkgs = prepare_tpfs(
            data, npca_components=npca_components, aperture_threshold=aperture_threshold
        )
        fit_bkg = True

    # Setup for when input data is a collection of light curve files
    elif isinstance(data, lk.LightCurveCollection):
        lc, data_uncorr, bkgs = prepare_lcfs(data)
        fit_bkg = False

    systematics_dm = vstack(bkgs)
    sigma_f_inv = sparse.csr_matrix(1 / lc.flux_err.value[:, None] ** 2)

    # Make a dummy design matrix
    period = 27
    ls_dm = lk.correctors.DesignMatrix(
        lombscargle.implementations.mle.design_matrix(
            lc.time.jd, frequency=1 / period, bias=False, nterms=1
        ),
        name="LS",
    ).to_sparse()
    ls_dm.prior_sigma = np.ones(ls_dm.shape[1]) * 1000
    dm = lk.correctors.SparseDesignMatrixCollection(
        [systematics_dm, ls_dm]
    ).to_designmatrix(name="design_matrix")

    if sff:
        sff_dm = []
        for lc in data_uncorr:
            if not isinstance(lc, lk.LightCurve):
                lc = lc.to_lightcurve()
            s = lk.correctors.SFFCorrector(lc)
            _ = s.correct(**sff_kwargs)
            sff_dm.append(s.dmc["sff"].to_sparse())
        sff_dm = vstack(sff_dm)
        dm = lk.correctors.SparseDesignMatrixCollection([dm, sff_dm]).to_designmatrix(
            name="design_matrix"
        )

    # Do a first pass at 27 days, just to find ridiculous outliers
    mask = np.isfinite(lc.flux.value)
    mask &= np.isfinite(lc.flux_err.value)
    mod = fit_model(lc, dm, sigma_f_inv, mask=mask, return_model=True)
    mask = ~(lc - mod * lc.flux.unit).remove_outliers(return_mask=True, sigma=sigma)[1]
    if periods is not None:
        periods = np.copy(periods)
    else:
        # Loop over some periods we care about
        periods = 1 / np.linspace(1 / min_period, 1 / max_period, nperiods)

    ws = np.zeros((len(periods), dm.X.shape[1]))
    ws_err = np.zeros((len(periods), dm.X.shape[1]))
    if fit_bkg:
        ws_bkg = np.zeros((len(periods), dm.X.shape[1]))
        ws_err_bkg = np.zeros((len(periods), dm.X.shape[1]))

    for idx, period in enumerate(tqdm(periods, desc="Running pixels in aperture")):
        dm.X[:, -ls_dm.shape[1] :] = lombscargle.implementations.mle.design_matrix(
            lc.time.jd, frequency=1 / period, bias=False, nterms=1
        )
        ws[idx], ws_err[idx] = fit_model(lc, dm, sigma_f_inv, mask=mask)
        if fit_bkg:
            ws_bkg[idx], ws_err_bkg[idx] = fit_model(lc_bkg, dm, sigma_f_inv, mask=mask)
    power = (ws[:, -2] ** 2 + ws[:, -1] ** 2) ** 0.5
    am = np.argmax(power)
    dm.X[:, -ls_dm.shape[1] :] = lombscargle.implementations.mle.design_matrix(
        lc.time.jd, frequency=1 / periods[am], bias=False, nterms=1
    )
    mod = dm.X[:, :-2].dot(ws[am][:-2])
    if fit_bkg:
        power_bkg = (ws_bkg[:, -2] ** 2 + ws_bkg[:, -1] ** 2) ** 0.5

    r = {
        "periods": periods,
        "power": power,
        "raw_lc": lc,
        "corr_lc": lc - mod * lc.flux.unit + 1 * lc.flux.unit,
        "period_at_max_power": periods[am],
        "model": mod * lc.flux.unit,
    }
    if fit_bkg:
        r["power_bkg"] = power_bkg
        r["raw_lc_bkg"] = lc_bkg
    return r
