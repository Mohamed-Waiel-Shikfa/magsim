"""Sourced permanent-magnet grade table, Preisach identification and temperature scaling (task 2).

Public API
----------
list_grades() -> list[str]
grade_info(name) -> dict          all row numbers, sources, assumptions, fitted parameters, fit errors
get_material(name, **overrides)   -> PreisachMaterial (overrides: Br, mu_rec, mu_perp, Hcj, alpha_Br,
                                     beta_Hcj, T_ref, depth, name; the fitted distribution shape is kept)
fit_datasheet(...)                identification (a): least squares on datasheet Br, Hcj, Hcb, (BH)max, Hk
fit_curve(H, B, ...)              identification (b): least squares on a tabulated second-quadrant curve
fit_library(write=True)           re-identifies every grade, writes res_exp/materials/preisach_fits.json

Units: SI (T, A/m, J/m^3, 1/K, Ohm m, kg/m^3, J/(kg K), K). Temperature coefficients are stored per
kelvin (datasheet %/degC divided by 100). Unit conversions: 1 Oe = 1000/(4 pi) A/m, 1 MGOe = 7957.747 J/m^3,
1 cal = 4.184 J (thermochemical).

Model semantics (see materials/preisach.py): `mu_rec` of a row is the datasheet recoil permeability,
which MMPA 0100-00 defines as the average slope of the recoil loop. In the classical Preisach model a
recoil from any second-quadrant point with |H| < 2 Hc_min of the switching-field support is purely
reversible, so the model's chord equals its reversible permeability `mu_rec`. NdFeB rows with a
grade-specific source (N42) fix it; Alnico rows only have a class value from another manufacturer
(Arnold), used as a prior (sigma 25 %); rows without a value identify it from (BH)max and Hcb.
The identified value is `grade_info(name)["mu_rec_model"]`.
"""

import hashlib
import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy import optimize

from magsim.constants import MU0
from magsim.materials import preisach
from magsim.materials.preisach import PreisachDistribution, PreisachMaterial

DATA_DIR = Path(__file__).resolve().parents[3] / "res_exp" / "materials"
FIT_FILE = DATA_DIR / "preisach_fits.json"
FIT_VERSION = 1

OE = 1000.0 / (4.0 * np.pi)  # A/m per oersted
MGOE = 1e6 * 1e-4 * OE  # J/m^3 per MGOe
CAL = 4.184  # J per (thermochemical) calorie
R_GAS = 8.314462618  # J/(mol K), CODATA 2018 (exact)
T20 = 293.15  # K

# Gate tolerances (docs/SPEC.md P1): relative.
TOLERANCE = {"Br": 0.005, "Hcj": 0.005, "Hcb": 0.02, "BHmax": 0.02, "Hk": 0.02}

# ----------------------------------------------------------------------------------------------- sources
SRC = {
    "mmpa": "MMPA Standard No. 0100-00, Standard Specifications for Permanent Magnet Materials (Magnetic "
            "Materials Producers Association): Table II-1 typical magnetic properties and composition of Alnico, "
            "II-4 physical properties (density, resistivity at 20 degC), II-5 thermal properties (reversible "
            "coefficients near Br and near Hc), definition of recoil permeability in the glossary; copy at "
            "https://www.duramag.com/pdf/MMPA_0100-00.pdf",
    "arnold_alnico": "Arnold Magnetic Technologies, Cast Alnico Permanent Magnets brochure (Rev. C, Feb 2003), "
                     "'Magnetic and Physical Properties (Typical Values)', recoil permeability column (G/Oe); "
                     "https://www.arnoldmagnetics.com/wp-content/uploads/2017/10/Cast-Alnico-Permanent-Magnet-Brochure-101117.pdf",
    "arnold_neo": "Arnold Magnetic Technologies, Neodymium Iron Boron Magnet Catalog (Rev. 210607/210802), grade data "
                  "sheet (min/nominal/max Br, HcB, HcJ min, (BH)max; alpha(Br), alpha(HcJ) measured 20-80 degC; "
                  "specific heat 0.11 cal/(g degC); density 7.6 g/cm^3; resistivity 150 (par) / 130 (perp) "
                  "uOhm cm); https://www.arnoldmagnetics.com/wp-content/uploads/2019/06/Arnold-Neo-Catalog.pdf",
    "eclipse": "Eclipse Magnetics, 'NdFeB Magnets / Neodymium Iron Boron Magnets' datasheet pp. 1-4 (N52 grade "
               "minima Br 1.430 T, Hcb 796 kA/m, Hcj 875 kA/m, (BH)max 398 kJ/m^3; typical resistivity "
               "150 uOhm cm, density 7500 kg/m^3, heat capacity 0.12 kcal/(kg K)), as transcribed in "
               "$PARENT/simulations_old/Switch_sim/magnet-simulation/magnet_pulse_sim.py:54-59,175-180 "
               "(reference case R1); https://www.eclipsemagnetics.com/",
    "r2": "$PARENT/simulations_old/Switch_sim/magnet-flip-sim/src/params.py:60-78 (reference case R2, N42); its "
          "docs/REFERENCES.md cites SM Magnetics / Magnet Expert (Br, Hcb, Hcj), Osenc Magnets / K&J Magnetics "
          "(mu_rec 1.05), K&J Magnetics (resistivity 1.4 uOhm m)",
    "vendor": "Supplier Alnico property sheet $PARENT/simulations_old/Force_compute/Alnico性能表.png (grade, MMPA "
              "equivalent, Br, Hcb, Hcj, (BH)max); the MMPA-equivalent column maps each row to its class",
    "katter": "M. Katter, 'Angular dependence of the demagnetization stability of sintered Nd-Fe-B magnets', "
              "IEEE Trans. Magn. 41(10), 3853-3855 (2005), doi:10.1109/TMAG.2005.854872 (DOI from the "
              "bibliographic record; abstract: reversible permeability 1.03-1.05 parallel and 1.12-1.17 "
              "perpendicular to the easy axis for fields well below the coercivity)",
    "mq1": "Magnequench, 'Magnet Selection for Cost-Optimized Motor Designs', SAE Automotive Engineering, Oct 2023, "
           "Table 1 'Representative MQ1 magnet grades', grade MQ1-10 (Br 7.04 kG, Hc 5.73 kOe, Hci 9.26 kOe, "
           "(BH)max 10.17 MGOe); https://www.mobilityengineeringtech.com/component/content/article/49076-sae-ma-07236",
    "femm_lng37": "res_exp/materials/femm_lng37.csv (FEMM material 'Cast Alnico 5 (LNG37)', "
                  "$PARENT/simulations_old/Force_compute/FEMM/accurate_alnico.FEM), identification (b)",
    "dulong_petit": "Dulong-Petit estimate c_p = 3R sum_i(w_i/M_i) from the MMPA Table II-1 composition "
                    "(balance Fe); classical high-temperature limit, room-temperature c_p of Fe, Co, Ni is within "
                    "a few percent of it",
    "ruoho2010": "Ruoho et al., IEEE Trans. Magn. 46(1), 15-20 (2010), doi:10.1109/TMAG.2009.2027815: sintered "
                 "rare-earth magnet resistivity is anisotropic, the anisotropy exceeding the -40..150 degC variation",
}

# MMPA Table II-1 composition, weight percent (Al, Ni, Co, Cu, Ti); balance Fe.
_ALNICO_COMPOSITION = {
    "Alnico 2": (10, 19, 13, 3, 0), "Alnico 3": (12, 25, 0, 3, 0), "Alnico 5": (8, 14, 24, 3, 0),
    "Alnico 5DG": (8, 14, 24, 3, 0), "Alnico 5-7": (8, 14, 24, 3, 0), "Alnico 6": (8, 16, 24, 3, 1),
    "Alnico 8": (7, 15, 35, 4, 5), "Alnico 8HC": (8, 14, 38, 3, 8), "Alnico 9": (7, 15, 35, 4, 5),
}
_MOLAR_MASS = {"Al": 26.9815, "Ni": 58.6934, "Co": 58.9332, "Cu": 63.546, "Ti": 47.867, "Fe": 55.845}  # g/mol


def dulong_petit_cp(composition_wt: tuple) -> float:
    """c_p = 3R sum(w_i/M_i) [J/(kg K)] for (Al, Ni, Co, Cu, Ti) weight percent, balance Fe."""
    names = ("Al", "Ni", "Co", "Cu", "Ti")
    w = dict(zip(names, composition_wt))
    w["Fe"] = 100.0 - sum(composition_wt)
    mol_per_g = sum(w[k] / 100.0 / _MOLAR_MASS[k] for k in w)
    return 3.0 * R_GAS * mol_per_g * 1000.0


# Alnico class data (MMPA 0100-00 Tables II-4, II-5; Arnold brochure recoil permeability).
# rho in uOhm cm (MMPA II-4), density g/cm^3 (II-4), TC %/degC near Br and near Hc (II-5), mu_rec (Arnold).
_ALNICO_CLASS = {
    "Alnico 2": dict(rho=65, density=7.1, tc_br=-0.03, tc_hc=-0.02, mu_rec=6.2, mu_src="Alnico 2"),
    "Alnico 3": dict(rho=60, density=6.9, tc_br=None, tc_hc=None, mu_rec=5.1, mu_src="Alnico 3"),
    "Alnico 5": dict(rho=47, density=7.3, tc_br=-0.02, tc_hc=0.01, mu_rec=3.7, mu_src="Alnico 5"),
    "Alnico 5DG": dict(rho=47, density=7.3, tc_br=None, tc_hc=None, mu_rec=2.4, mu_src="Alnico 5cc"),
    "Alnico 5-7": dict(rho=47, density=7.3, tc_br=None, tc_hc=None, mu_rec=2.0, mu_src="ArKomax 800"),
    "Alnico 6": dict(rho=50, density=7.3, tc_br=-0.02, tc_hc=0.03, mu_rec=5.6, mu_src="Alnico 6"),
    "Alnico 8": dict(rho=53, density=7.3, tc_br=-0.025, tc_hc=0.01, mu_rec=2.0, mu_src="Alnico 8B"),
    "Alnico 8HC": dict(rho=54, density=7.3, tc_br=-0.025, tc_hc=0.01, mu_rec=2.0, mu_src="Alnico 8H"),
    "Alnico 9": dict(rho=53, density=7.3, tc_br=-0.025, tc_hc=0.01, mu_rec=1.3, mu_src="Alnico 9"),
}


@dataclass(frozen=True)
class Grade:
    """One row of the grade table (SI units). None = not given by the source (not a fit target)."""

    name: str
    family: str  # "sintered NdFeB" | "bonded NdFeB" | "cast Alnico"
    Br: float | None  # T (None for curve-identified rows: comes from the fit)
    Hcj: float | None  # A/m
    Hcb: float | None  # A/m
    BHmax: float | None  # J/m^3
    Hk: float | None  # A/m (J = 0.9 Br)
    mu_rec: float | None  # datasheet recoil permeability; None -> identified
    mu_perp: float | None  # None -> equal to the model mu_rec (assumption recorded in `assumptions`)
    alpha_Br: float  # 1/K
    beta_Hcj: float  # 1/K
    rho_e: float  # Ohm m, perpendicular to the easy axis (eddy loops around the axis)
    rho_e_par: float  # Ohm m, along the easy axis
    density: float  # kg/m^3
    cp: float  # J/(kg K)
    T_ref: float = T20
    mmpa_class: str | None = None
    curve: str | None = None  # CSV for identification (b)
    sources: dict = field(default_factory=dict)
    assumptions: tuple = ()


def _ndfeb(name, Br, Hcb, Hcj, BHmax, src_mag, *, mu_rec=None, alpha=-0.12, beta=-0.62, rho=(1.30e-6, 1.50e-6),
           density=7600.0, cp=0.11 * CAL * 1000, src_phys="arnold_neo", assumptions=(), family="sintered NdFeB",
           mu_perp=1.15, src_mu_rec=None, src_mu_perp="katter", src_thermal=None):
    src = {"magnetic": SRC[src_mag], "thermal": SRC[src_thermal or src_phys],
           "resistivity": SRC[src_phys], "density": SRC[src_phys], "cp": SRC[src_phys],
           "mu_perp": SRC[src_mu_perp] if src_mu_perp else "assumption",
           "mu_rec": SRC[src_mu_rec] if src_mu_rec else "identified from Br, Hcb, (BH)max (fit_datasheet)"}
    return Grade(name, family, Br, Hcj, Hcb, BHmax, None, mu_rec, mu_perp, alpha / 100, beta / 100, rho[0], rho[1],
                 density, cp, sources=src, assumptions=tuple(assumptions))


def _alnico(name, cls, Br, Hcb, Hcj, BHmax, src_mag, *, extra_assumptions=(), curve=None):
    c = _ALNICO_CLASS[cls]
    assumptions = list(extra_assumptions)
    tc_cls = cls
    if c["tc_br"] is None:
        tc_cls = {"Alnico 3": "Alnico 2"}.get(cls, "Alnico 5")
        assumptions.append(f"# ASSUMPTION: MMPA II-5 lists no thermal coefficients for {cls}; {tc_cls} values used "
                           f"({'isotropic cast grade' if tc_cls == 'Alnico 2' else 'same Al-Ni-Co-Cu composition'}).")
    tc = _ALNICO_CLASS[tc_cls]
    if c["mu_src"] != cls:
        assumptions.append(f"# ASSUMPTION: recoil permeability of {cls} taken from Arnold '{c['mu_src']}', the Arnold "
                           f"grade whose Br/Hc/(BH)max match the MMPA {cls} row.")
    assumptions.append("# ASSUMPTION: mu_perp = mu_rec (no transverse permeability data found for Alnico).")
    assumptions.append("# ASSUMPTION: Hcj temperature coefficient = MMPA II-5 coefficient 'near Hc' (Hcj ~ Hc for Alnico).")
    src = {"magnetic": SRC[src_mag],
           "mu_rec": "prior (sigma 25 %) for identification (a): " + SRC["arnold_alnico"] + f" (grade {c['mu_src']})",
           "mu_perp": "assumption", "thermal": SRC["mmpa"] + f" (Table II-5, {tc_cls})",
           "resistivity": SRC["mmpa"] + f" (Table II-4, {cls})", "density": SRC["mmpa"] + f" (Table II-4, {cls})",
           "cp": SRC["dulong_petit"] + f" ({cls})", "class": f"MMPA {cls}"}
    rho = c["rho"] * 1e-8
    mu_rec = c["mu_rec"]
    if curve:  # identified together with the curve, the class value is not imposed
        mu_rec = None
        src["mu_rec"] = "identified from the curve (fit_curve); Arnold class value for comparison: " + src["mu_rec"]
    return Grade(name, "cast Alnico", Br, Hcj, Hcb, BHmax, None, mu_rec, None, tc["tc_br"] / 100,
                 tc["tc_hc"] / 100, rho, rho, c["density"] * 1000.0, dulong_petit_cp(_ALNICO_COMPOSITION[cls]),
                 mmpa_class=cls, curve=curve, sources=src, assumptions=tuple(assumptions))


_KA = 1e3
GRADES: dict[str, Grade] = {}


def _add(g: Grade):
    GRADES[g.name] = g


# --- sintered NdFeB -------------------------------------------------------------------------------------
_MIN_NOTE = "Arnold catalogue minimum values (Br, HcB, HcJ, (BH)max) used as one consistent set; the nominal Br/HcB " \
            "cannot be combined with the min-only HcJ (Arnold N52 nominal HcB 979 > min HcJ 875 kA/m)."
_add(_ndfeb("N35", 1.170, 860 * _KA, 955 * _KA, 263e3, "arnold_neo", assumptions=(_MIN_NOTE,)))
_add(Grade("N42", "sintered NdFeB", 1.32, 955e3, 915e3, None, None, 1.05, 1.15, -0.0012, -0.006, 1.4e-6, 1.4e-6, 7500.0,
           460.0, sources={"magnetic": SRC["r2"], "mu_rec": SRC["r2"], "mu_perp": SRC["katter"], "thermal": SRC["r2"],
                           "resistivity": SRC["r2"], "density": SRC["r2"], "cp": SRC["r2"]},
           assumptions=("R2 reference values; params.py gives no (BH)max, so none is fitted.",
                        "# ASSUMPTION: params.py gives one resistivity; used for both directions.")))
_add(_ndfeb("N48", 1.370, 836 * _KA, 875 * _KA, 358e3, "arnold_neo", assumptions=(_MIN_NOTE,)))
_add(_ndfeb("N52", 1.430, 796 * _KA, 875 * _KA, 398e3, "eclipse", rho=(1.50e-6, 1.50e-6), density=7500.0,
            cp=0.12 * CAL * 1000, src_phys="eclipse", src_thermal="arnold_neo",
            assumptions=("R1 reference values (Eclipse minima).",
                         "Eclipse gives no temperature coefficients: Arnold N52 sheet values used.",
                         "mu_rec identified from (BH)max (the old code's 1.05 was itself an assumption).")))
_add(_ndfeb("N42SH", 1.280, 955 * _KA, 1592 * _KA, 310e3, "arnold_neo", beta=-0.55, assumptions=(
    _MIN_NOTE, "Hcb and (BH)max lie on the straight part of the curve: the knee shape (switching-field width) is "
               "set by the family prior (# ASSUMPTION: s = 0.04, w/(h0 s) = 1).")))
# --- bonded NdFeB ---------------------------------------------------------------------------------------
_add(Grade("NdFeB-bonded", "bonded NdFeB", 7.04e3 * 1e-4, 9.26e3 * OE, 5.73e3 * OE, 10.17 * MGOE, None, None, None,
           -0.0012, -0.004, 1.0e-4, 1.0e-4, 6000.0, 460.0,
           sources={"magnetic": SRC["mq1"], "mu_rec": "identified from Br, Hcb, (BH)max (fit_datasheet)",
                    "mu_perp": "assumption", "thermal": "assumption", "resistivity": "assumption",
                    "density": "assumption", "cp": "assumption"},
           assumptions=(
               "# ASSUMPTION: mu_perp = identified mu_rec (isotropic melt-spun powder: no preferred axis before magnetising).",
               "# ASSUMPTION: alpha_Br = -0.12 %/K (Nd2Fe14B phase, as sintered) and beta_Hcj = -0.40 %/K; no MQ1 datasheet "
               "coefficients found. Bonded NdFeB is not used by the reference cases.",
               "# ASSUMPTION: resistivity 1e-4 Ohm m, an order-of-magnitude placeholder for powder in an epoxy matrix "
               "(>= 50x sintered NdFeB); no source found.",
               "# ASSUMPTION: density 6000 kg/m^3 (~80 % packing of 7.6 g/cm^3 powder) and c_p 460 J/(kg K) (sintered NdFeB).",
               "Knee shape partly set by the family prior (# ASSUMPTION: s = 0.08, w/(h0 s) = 1).")))
# --- cast Alnico, MMPA 0100-00 Table II-1 (typical values) ----------------------------------------------
_add(_alnico("Alnico 2", "Alnico 2", 0.750, 45 * _KA, 46 * _KA, 13.5e3, "mmpa"))
_add(_alnico("Alnico 3", "Alnico 3", 0.700, 38 * _KA, 40 * _KA, 10.7e3, "mmpa"))
_add(_alnico("Alnico 5", "Alnico 5", 1.280, 51 * _KA, 51 * _KA, 43.8e3, "mmpa"))
_add(_alnico("Alnico 5-7", "Alnico 5-7", 1.350, 59 * _KA, 59 * _KA, 59.7e3, "mmpa"))
_add(_alnico("Alnico 8", "Alnico 8", 0.820, 131 * _KA, 148 * _KA, 42.2e3, "mmpa"))
_add(_alnico("Alnico 9", "Alnico 9", 1.060, 119 * _KA, 119 * _KA, 71.6e3, "mmpa"))
# --- vendor LNG / LNGT rows (supplier sheet), class data from the MMPA-equivalent column -----------------
_VENDOR = [  # name, MMPA equivalent, Br [mT], Hcb, Hcj [kA/m], (BH)max [kJ/m^3]
    ("LN10", "Alnico 3", 600, 40, 43, 10), ("LNG13", "Alnico 2", 680, 48, 51, 13),
    ("LNGT18", "Alnico 8", 580, 90, 92, 18), ("LNG37", "Alnico 5", 1200, 48, 49, 37),
    ("LNG40", "Alnico 5", 1250, 48, 49, 40), ("LNG44", "Alnico 5", 1250, 52, 53, 44),
    ("LNG52", "Alnico 5DG", 1300, 56, 57, 52), ("LNG60", "Alnico 5-7", 1350, 59, 60, 60),
    ("LNGT28", "Alnico 6", 1000, 58, 59, 28), ("LNGT36J", "Alnico 8HC", 700, 140, 148, 36),
    ("LNGT38", "Alnico 8", 800, 110, 112, 38), ("LNGT44", "Alnico 8", 880, 120, 122, 44),
    ("LNGT60", "Alnico 9", 900, 110, 112, 60), ("LNGT72", "Alnico 9", 1050, 112, 114, 72),
]
for _n, _cls, _br, _hcb, _hcj, _bh in _VENDOR:
    _extra = ("# ASSUMPTION: the sheet lists LNGT18 as isotropic; Alnico 8 class data (anisotropic) used.",) \
        if _n == "LNGT18" else ()
    _add(_alnico(_n, _cls, _br * 1e-3, _hcb * _KA, _hcj * _KA, _bh * 1e3, "vendor", extra_assumptions=_extra))
_add(_alnico("LNG37-femm", "Alnico 5", None, None, None, None, "femm_lng37", curve="femm_lng37.csv",
             extra_assumptions=("Magnetic curve identified from the FEMM table (identification (b)); mu_rec identified "
                                "with it (the Alnico 5 class value 3.7 is not imposed).",)))


def list_grades() -> list[str]:
    """Names of all library grades, in table order."""
    return list(GRADES)


# ----------------------------------------------------------------------------------------------- identification
FAMILY_PRIOR = {
    # g kind; prior support of the Hc density, U0 = ln(hc_max/h0), L0 = ln(h0/hc_min) (log-normal with log-width
    # s0 <=> U0 = L0 = 4 s0), prior ratio r0 = w/(h0 s); mu_rec: prior value mu0 (absolute sigma 0.1) when the row has
    # none, relative prior sigma mu_sigma around the row value (0 = fixed), upper bound mu_max.
    "sintered NdFeB": dict(kind="gauss", U0=0.16, L0=0.16, r0=1.0, mu0=1.05, mu_sigma=0.0, mu_max=1.3),
    "bonded NdFeB": dict(kind="gauss", U0=0.32, L0=0.32, r0=1.0, mu0=1.15, mu_sigma=0.0, mu_max=1.5),
    # Alnico: U0, L0, r0 from the LNG37-femm curve fit (data-derived); the Arnold recoil permeability is a class
    # value from another manufacturer's product, so it is a prior (sigma 25 %), not a constraint.
    "cast Alnico": dict(kind="lorentz", U0=None, L0=None, r0=None, mu0=2.0, mu_sigma=0.25, mu_max=10.0),
}
PRIOR_SIGMA_LOG = 1.0  # weak: one e-fold in U, L and r is one standard deviation
R_MIN, R_MAX = 0.25, 4.0  # bounds on r = w/(h0 s)
# Support of the switching-field density kept within [h0/10, 3 h0]: magnetising fields of 3-5 Hc saturate cast
# Alnico (Arnold brochure: 2000-6000 Oe "required magnetizing field"), so no relay needs more than ~3 h0; the
# bound also keeps the z -> Hc map of the quadrature well conditioned (tests/test_library.py checks accuracy).
SUPPORT_MIN, U_MAX, L_MAX = 0.004, np.log(3.0), np.log(10.0)


def shape_from_support(U: float, L: float) -> tuple[float, float]:
    """(s, skew) of the truncated Tukey-g log-normal with support [h0 exp(-L), h0 exp(U)] at |z| = Z_MAX."""
    k = np.log(U / L) / preisach.Z_MAX
    if abs(k) < 1e-12:
        return U / preisach.Z_MAX, 0.0
    return U * k / np.expm1(k * preisach.Z_MAX), k


def _material(Br, h0, U, L, r, kind, mu_rec, H_scale, **kw) -> PreisachMaterial:
    s, k = shape_from_support(U, L)
    return PreisachMaterial(Br, PreisachDistribution(h0, s, r * h0 * s, kind, k), H_scale, mu_rec=mu_rec, **kw)


def _params(mat: PreisachMaterial, Br, U, L, r) -> dict:
    d = mat.dist
    return {"Br": float(Br), "h0": d.h0, "s": d.s, "w": d.w, "r": float(r), "skew": d.skew, "U": float(U),
            "L": float(L), "kind": d.kind, "mu_rec": mat.mu_rec, "H_scale": mat.H_scale, "Js_irr": mat.Js_irr,
            "P_r": mat.P_r}


def fit_datasheet(Br: float, Hcj: float, Hcb: float | None = None, BHmax: float | None = None,
                  Hk: float | None = None, squareness: float | None = None, mu_rec: float | None = None, *,
                  mu_sigma: float = 0.0, kind: str = "gauss", U0: float = 0.16, L0: float = 0.16, r0: float = 1.0,
                  mu0: float = 1.05, mu_max: float = 1.3) -> dict:
    """Identification (a): Preisach parameters from datasheet numbers by least squares.

    Fitted: h0 (median switching field / Hcj), the support of the Hc density [h0 e^-L, h0 e^U] (which sets
    its width s and skew, `shape_from_support`), r = w/(h0 s), and the reversible permeability mu_rec in
    [1, mu_max] unless `mu_rec` is given with mu_sigma = 0 (fixed). Js_irr follows from J(0) = Br on the
    descending major branch (exact). Residuals: (model/target - 1)/(tol/4) for Hcj (tol 0.5 %), Hcb, (BH)max
    and Hk (tol 2 %; Hk = squareness*Hcj if only the squareness Hk/Hcj is given), plus weak priors that only
    decide what the datasheet leaves undetermined: (ln U - ln U0), (ln L - ln L0), (ln r - ln r0) with sigma 1,
    and for mu_rec either (mu_rec/mu_rec_row - 1)/mu_sigma or, without a row value, (mu_rec - mu0)/0.1.
    Field unit H_scale = Hcj. scipy.optimize.least_squares (trust-region reflective, bounds).
    Returns dict: params, characteristics, targets, rel_error per target, cost, success, message.
    """
    if Hk is None and squareness is not None:
        Hk = squareness * Hcj
    targets = {k: v for k, v in (("Hcj", Hcj), ("Hcb", Hcb), ("BHmax", BHmax), ("Hk", Hk)) if v is not None}
    free_mu = mu_rec is None or mu_sigma > 0
    mu_start = mu0 if mu_rec is None else min(mu_rec, mu_max)
    lb = [0.6, np.log(SUPPORT_MIN), np.log(SUPPORT_MIN), np.log(R_MIN)] + ([1.0] if free_mu else [])
    ub = [1.6, np.log(U_MAX), np.log(L_MAX), np.log(R_MAX)] + ([mu_max] if free_mu else [])
    x0 = np.clip([1.0, np.log(U0), np.log(L0), np.log(r0)] + ([mu_start] if free_mu else []),
                 np.array(lb) + 1e-9, np.array(ub) - 1e-9)

    def build(x):
        return _material(Br, x[0], np.exp(x[1]), np.exp(x[2]), np.exp(x[3]), kind, x[4] if free_mu else mu_rec, Hcj)

    def resid(x):
        ch = build(x).characteristics()
        res = [(ch[k] / v - 1.0) / (TOLERANCE[k] / 4.0) for k, v in targets.items()]
        res += [(x[1] - np.log(U0)) / PRIOR_SIGMA_LOG, (x[2] - np.log(L0)) / PRIOR_SIGMA_LOG,
                (x[3] - np.log(r0)) / PRIOR_SIGMA_LOG]
        if free_mu:
            res.append((x[4] / mu_rec - 1.0) / mu_sigma if mu_rec is not None else (x[4] - mu0) / 0.1)
        return np.array(res)

    sol = optimize.least_squares(resid, x0, bounds=(lb, ub), x_scale="jac", xtol=1e-12, ftol=1e-12, gtol=1e-12,
                                 diff_step=1e-6)
    mat = build(sol.x)
    ch = mat.characteristics()
    targets["Br"] = Br
    rel = {k: ch[k] / v - 1.0 for k, v in targets.items()}
    return {"params": _params(mat, Br, *np.exp(sol.x[1:4])), "characteristics": ch, "targets": targets,
            "rel_error": rel, "cost": float(sol.cost), "success": bool(sol.success), "message": sol.message,
            "method": "datasheet least squares (a)"}


def read_curve(path) -> tuple[np.ndarray, np.ndarray]:
    """Read a CSV with '#' comments and columns H_A_per_m and B_T (or J_T); returns (H, B) [A/m, T]."""
    path = Path(path)
    if not path.is_absolute():
        path = DATA_DIR / path
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    head = [h.strip() for h in lines[0].split(",")]
    data = np.array([[float(v) for v in ln.split(",")] for ln in lines[1:]])
    H = data[:, head.index("H_A_per_m")]
    if "B_T" in head:
        B = data[:, head.index("B_T")]
    elif "J_T" in head:
        B = data[:, head.index("J_T")] + MU0 * H
    else:
        raise ValueError(f"{path}: need a B_T or J_T column, got {head}")
    return H, B


def fit_curve(H, B, *, kind: str = "lorentz", mu_rec: float | None = None) -> dict:
    """Identification (b): least squares of the descending major branch B(H) to tabulated points.

    Fitted: Br, h0, support (U, L) of the Hc density, r = w/(h0 s) and mu_rec (unless given), same bounds as
    `fit_datasheet`, no priors. Field unit H_scale = -H at the data point with the smallest |B| (about Hcb).
    Unweighted residuals in tesla. Standard errors from the Jacobian, cov = (J^T J)^-1 RSS/(n - p), delta
    method for U, L, r (fitted as logarithms). Returns params, stderr, rms, max_abs, residuals, characteristics.
    """
    H = np.asarray(H, dtype=float)
    B = np.asarray(B, dtype=float)
    H_scale = float(-H[np.argmin(np.abs(B))])
    Bmax = float(B.max())
    free_mu = mu_rec is None
    x0 = [Bmax, 1.0, np.log(0.3), np.log(0.3), np.log(0.6)] + ([2.0] if free_mu else [])
    lb = [0.5 * Bmax, 0.5, np.log(SUPPORT_MIN), np.log(SUPPORT_MIN), np.log(R_MIN)] + ([1.0] if free_mu else [])
    ub = [1.5 * Bmax, 2.0, np.log(U_MAX), np.log(L_MAX), np.log(R_MAX)] + ([10.0] if free_mu else [])

    def build(x):
        return _material(x[0], x[1], np.exp(x[2]), np.exp(x[3]), np.exp(x[4]), kind, x[5] if free_mu else mu_rec,
                         H_scale)

    def model(x):
        return MU0 * H + build(x).major_loop(H)[0]

    sol = optimize.least_squares(lambda x: model(x) - B, x0, bounds=(lb, ub), x_scale="jac", xtol=1e-12,
                                 ftol=1e-12, gtol=1e-12, diff_step=1e-6)
    res = model(sol.x) - B
    n, k = H.size, sol.x.size
    dof = max(n - k, 1)
    cov = np.linalg.pinv(sol.jac.T @ sol.jac) * float(res @ res) / dof
    se = np.sqrt(np.diag(cov))
    names = ["Br", "h0", "U", "L", "r"] + (["mu_rec"] if free_mu else [])
    stderr = {nm: float(v) for nm, v in zip(names, se)}
    for i, nm in ((2, "U"), (3, "L"), (4, "r")):  # delta method: d q = q d ln q
        stderr[nm] = float(np.exp(sol.x[i]) * se[i])
    mat = build(sol.x)
    return {"params": _params(mat, sol.x[0], *np.exp(sol.x[2:5])), "stderr": stderr,
            "rms": float(np.sqrt(np.mean(res ** 2))), "max_abs": float(np.abs(res).max()),
            "residuals": res.tolist(), "H": H.tolist(), "B": B.tolist(), "dof": dof,
            "characteristics": mat.characteristics(), "cost": float(sol.cost), "success": bool(sol.success),
            "message": sol.message, "method": "curve least squares (b)"}


# ----------------------------------------------------------------------------------------------- fitted cache
_FIT_CACHE: dict[str, dict] = {}
_FILE_CACHE: dict | None = None


def _fit_key(g: Grade, prior: dict) -> str:
    """Hash of everything that determines a grade's fit (stale-cache detection)."""
    payload = {"v": FIT_VERSION, "z": preisach.Z_MAX, "n": preisach.N_PANEL, "Br": g.Br, "Hcj": g.Hcj, "Hcb": g.Hcb,
               "BHmax": g.BHmax, "Hk": g.Hk, "mu_rec": g.mu_rec, "family": g.family, "curve": g.curve,
               "prior": prior}
    if g.curve:
        payload["curve_data"] = hashlib.sha256((DATA_DIR / g.curve).read_bytes()).hexdigest()
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=float).encode()).hexdigest()[:16]


def _prior(g: Grade) -> dict:
    pr = dict(FAMILY_PRIOR[g.family])
    if g.family == "cast Alnico" and not g.curve:
        ref = _fit_for("LNG37-femm")["params"]  # data-derived Alnico shape prior
        pr["U0"], pr["L0"], pr["r0"] = float(ref["U"]), float(ref["L"]), float(ref["r"])
    return pr


def fit_grade(name: str) -> dict:
    """Identify one library grade (datasheet fit (a), or curve fit (b) for rows with a curve)."""
    g = GRADES[name]
    pr = _prior(g)
    if g.curve:
        H, B = read_curve(g.curve)
        out = fit_curve(H, B, kind=pr["kind"], mu_rec=None)
    else:
        out = fit_datasheet(g.Br, g.Hcj, g.Hcb, g.BHmax, g.Hk, mu_rec=g.mu_rec, mu_sigma=pr["mu_sigma"],
                            kind=pr["kind"], U0=pr["U0"], L0=pr["L0"], r0=pr["r0"], mu0=pr["mu0"],
                            mu_max=pr["mu_max"])
    out["key"] = _fit_key(g, pr)
    out["grade"] = name
    return out


def _load_file() -> dict:
    global _FILE_CACHE
    if _FILE_CACHE is None:
        _FILE_CACHE = json.loads(FIT_FILE.read_text()) if FIT_FILE.exists() else {}
    return _FILE_CACHE


def _fit_for(name: str) -> dict:
    if name in _FIT_CACHE:
        return _FIT_CACHE[name]
    g = GRADES[name]
    stored = _load_file().get("grades", {}).get(name)
    if stored is not None and stored.get("key") == _fit_key(g, _prior(g)):
        _FIT_CACHE[name] = stored
    else:
        warnings.warn(f"{name}: no up-to-date fit in {FIT_FILE.name}; identifying now "
                      f"(regenerate with `python -m magsim.materials.library`)", stacklevel=3)
        _FIT_CACHE[name] = fit_grade(name)
    return _FIT_CACHE[name]


def fit_library(write: bool = True, path: Path | None = None) -> dict:
    """Re-identify every grade (LNG37-femm first: it sets the Alnico shape prior) and optionally write JSON."""
    _FIT_CACHE.clear()
    out = {}
    for name in ["LNG37-femm"] + [n for n in GRADES if n != "LNG37-femm"]:
        _FIT_CACHE[name] = out[name] = _jsonable(fit_grade(name))
    if write:
        payload = {"description": "Preisach parameters identified by magsim.materials.library.fit_library(); "
                                  "regenerate with `uv run python -m magsim.materials.library`.",
                   "version": FIT_VERSION, "grades": out}
        (path or FIT_FILE).write_text(json.dumps(payload, indent=1, sort_keys=True))
        global _FILE_CACHE
        _FILE_CACHE = payload
    return out


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ----------------------------------------------------------------------------------------------- public API
def grade_info(name: str) -> dict:
    """All numbers of a grade (SI), its sources and assumptions, and the identified model with fit errors."""
    if name not in GRADES:
        raise KeyError(f"unknown grade {name!r}; see list_grades()")
    g = GRADES[name]
    info = asdict(g)
    fit = _fit_for(name)
    info["fit"] = fit
    if g.curve:  # curve-identified row: datasheet-style numbers come from the model
        ch = fit["characteristics"]
        info.update(Br=ch["Br"], Hcj=ch["Hcj"], Hcb=ch["Hcb"], BHmax=ch["BHmax"])
    info["mu_rec_model"] = fit["params"]["mu_rec"]
    info["mu_perp_model"] = g.mu_perp if g.mu_perp is not None else fit["params"]["mu_rec"]
    return info


_OVERRIDES = {"Br", "mu_rec", "mu_perp", "Hcj", "alpha_Br", "beta_Hcj", "T_ref", "depth", "name"}


def get_material(name: str, /, **overrides) -> PreisachMaterial:
    """PreisachMaterial for a library grade.

    overrides (no re-identification; the fitted distribution shape in reduced units is kept):
      Br [T]       - remanence at T_ref (calibration), Js_irr rescaled so J(0) = Br;
      Hcj [A/m]    - rescales the field axis of the distribution (H_scale * Hcj/Hcj_row);
      mu_rec, mu_perp, alpha_Br [1/K], beta_Hcj [1/K], T_ref [K], depth, name.
    """
    unknown = set(overrides) - _OVERRIDES
    if unknown:
        raise ValueError(f"unknown override(s) {sorted(unknown)}; allowed: {sorted(_OVERRIDES)}")
    info = grade_info(name)
    p = info["fit"]["params"]
    H_scale = p["H_scale"]
    if "Hcj" in overrides:
        H_scale *= float(overrides["Hcj"]) / info["Hcj"]
    kw = dict(mu_rec=p["mu_rec"], mu_perp=info["mu_perp_model"], alpha_Br=info["alpha_Br"],
              beta_Hcj=info["beta_Hcj"], T_ref=info["T_ref"], name=name)
    kw.update({k: v for k, v in overrides.items() if k in ("mu_rec", "mu_perp", "alpha_Br", "beta_Hcj", "T_ref",
                                                            "depth", "name")})
    Br = float(overrides.get("Br", p["Br"]))
    return PreisachMaterial(Br, PreisachDistribution(p["h0"], p["s"], p["w"], p["kind"], p["skew"]), H_scale, **kw)


if __name__ == "__main__":  # regenerate res_exp/materials/preisach_fits.json
    res = fit_library(write=True)
    for nm, f in res.items():
        err = f.get("rel_error") or {"rms_T": f.get("rms")}
        print(f"{nm:14s} " + " ".join(f"{k}={v:+.2e}" for k, v in err.items()))
