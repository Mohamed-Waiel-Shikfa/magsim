import math

from magsim import constants as c


def test_mu0_codata2018():
    assert c.MU0 == 1.25663706212e-6
    # CODATA 2018 MU0 differs from the pre-2019 exact 4e-7*pi by ~5.5e-10 relative.
    assert abs(c.MU0 / (4e-7 * math.pi) - 1) < 1e-9


def test_copper_data():
    assert c.RHO_CU_20 == 1.7241e-8
    assert c.ALPHA_CU == 3.93e-3
    assert c.DENSITY_CU == 8960
    assert c.CP_CU == 385
