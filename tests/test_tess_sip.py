import tess_sip as ts
import lightkurve as lk
import os
import numpy as np

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
TESTDIR = "/".join(PACKAGEDIR.split("/")[:-1])

tpf_filenames = [
    TESTDIR + "/tests/data/tpf1.fits",
    TESTDIR + "/tests/data/tpf2.fits",
]

lcf_filenames = [
    TESTDIR + "/tests/data/lcf1.fits",
    TESTDIR + "/tests/data/lcf2.fits",
]


def test_sip():
    keys = [
        "periods",
        "power",
        "raw_lc",
        "power_bkg",
        "raw_lc_bkg",
        "corr_lc",
        "period_at_max_power",
        "model",
    ]

    tpfs = lk.TargetPixelFileCollection([lk.read(f) for f in tpf_filenames])
    lc, lc_bkg, data_uncorr, bkgs = ts.tess_sip.prepare_tpfs(tpfs)
    r1 = ts.SIP(tpfs)
    assert np.all([key in r1 for key in keys])
    lcfs = lk.LightCurveCollection([lk.read(f) for f in lcf_filenames])
    lc, lc_bkg, data_uncorr, bkgs = ts.tess_sip.prepare_lcfs(lcfs)
    r2 = ts.SIP(lcfs)
    assert np.all([key in r2 for key in keys])
