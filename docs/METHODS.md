# magsim: methods

Methods description for the force and switching simulators, written from the implemented code
(`src/magsim/`). Bib keys refer to `docs/references.bib`. Sections 7 and parts of 8 describe the
coupled switching solver and the force application of tasks 5 and 6; their numbers are filled in from
those reports.

## 1. Problem statement and conventions

Given permanent magnets (cuboids, cylinders, rings; NdFeB and Alnico grades) at arbitrary poses, we
compute (i) the self-consistent, history-dependent polarisation of every magnet and the force and
torque on a target body, and (ii) the coupled transient of a pulse-driver circuit, its coil, eddy
currents in the magnet and the magnet's hysteretic switching.

All quantities are SI (m, A, T, A/m, s, V, Ω, K), including the scenario YAML; $\mu_0 =
1.25663706212\times10^{-6}$ H/m (CODATA 2018, `magsim.constants.MU0`). Polarisation $\mathbf J =
\mu_0\mathbf M$ in tesla, so $\mathbf B = \mu_0\mathbf H + \mathbf J$. The global frame is right-handed
Cartesian; a body's pose is its centroid position and a rotation (body → global). In the body frame
every shape is centred on the origin, the cylinder/ring axis is body $z$ and the easy axis
$\hat{\mathbf e}$ defaults to $+z$. Material laws take the field in the material, split as
$H_\parallel = \mathbf H\cdot\hat{\mathbf e}$ and the two orthogonal components $\mathbf H_\perp$.
Forces and torques are reported on a named target body, from all other bodies, about a pivot (default:
target centroid). Temperatures are kelvin in code.

## 2. Material model

### 2.1 Scalar Preisach plus reversible part

Each cell carries an independent scalar Preisach model along its easy axis [mayergoyz1991]:

$$J_\parallel = J_{s,\mathrm{irr}}(T)\,P(H_\parallel;\text{history}) + \mu_0(\mu_\mathrm{rec}-1)H_\parallel,\qquad
\mathbf J_\perp = \mu_0(\mu_\perp-1)\mathbf H_\perp .$$

$P\in[-1,1]$ is the signed measure of the "up" relay hysterons $\gamma_{\alpha\beta}$ (switch up at
$\alpha$, down at $\beta$, $\alpha\ge\beta$). It is evaluated through the Everett function
$E(\alpha,\beta)$ (mass of hysterons with $\beta\le\beta'\le\alpha'\le\alpha$) and the
alternating-extrema memory: for the input history base $\to t_1\to\dots\to t_k\to h$ (base
$s_0=\pm1$ for saturation at $\pm\infty$),

$$P = s_0 + 2\sum_j \operatorname{sign}(y_j-y_{j-1})\,E\big(\max(y_j,y_{j-1}),\min(y_j,y_{j-1})\big).$$

The recoil slope at any reversal point is exactly $\mu_0\mu_\mathrm{rec}$, because the Everett term
has zero slope there. $\mu_\perp$ is linear and history-free.

**Distribution.** In the coordinates $H_c=(\alpha-\beta)/2$, $H_u=(\alpha+\beta)/2$ (reduced fields
$h = H/H_\mathrm{scale}$) the density factorises, $\mu\,d\alpha\,d\beta = f(H_c)\,g(H_u)\,dH_c\,dH_u$,
normalised to $E(+\infty,-\infty)=1$:

- $f$: a switching-field density with median $h_0$ and log-width $s$, written as
  $H_c = h_0\exp\!\big(s(e^{\kappa z}-1)/\kappa\big)$, $z\sim\mathcal N(0,1)$ truncated at
  $|z|\le 4$ and renormalised (Tukey-g skew $\kappa$; $\kappa=0$ is the log-normal). The truncation is
  part of the model definition, not an approximation.
- $g$: interaction-field density of width $w$: Lorentzian (Cauchy) for cast Alnico, Gaussian for
  sintered and bonded NdFeB. On the FEMM LNG37 curve the Lorentzian gives 1.6 mT RMS against 12.7 mT for
  a Gaussian product (before skew), and the skew lowers the fit further.

With $G$ the closed-form CDF of $g$,

$$E(\alpha,\beta)=\int_0^{(\alpha-\beta)/2} f(H_c)\,\big[G(\alpha-H_c)-G(\beta+H_c)\big]\,dH_c ,$$

a single 1-D quadrature (Gauss–Legendre, 24 nodes on each of three panels in $z$).

**History storage, wiping-out.** Each cell stores up to `depth` (default 32) alternating, nested
turning points in reduced units. A new extremum that passes a stored one wipes it out
[mayergoyz1991]. If the stack is full, the innermost reversal is not stored and that excursion is
treated as reversible. $J$ stays continuous, and the error in $P$ is at most the Everett weight of that
innermost minor loop. `MagnetSystem.solve` is a trial evaluation. Only `commit` advances the history,
along the monotone segment from the last committed $H_\parallel$.

**Initial states.** `+sat`/`-sat` (base $\pm\infty$); `thermal` (demagnetised staircase along
$\alpha=-\beta$ with 16 decaying turning points, the last tuned so that $P=0$ at $h=0$); or a float
$P_0$ (a recoil state at $H=0$ if $|P_0|\le P_r$, else a point on the major branch).

**Temperature.** $J_{s,\mathrm{irr}}(T)=J_{s,\mathrm{irr}}\,[1+\alpha_{B_r}(T-T_\mathrm{ref})]$ and
$H_\mathrm{scale}(T)=H_{cj}\,[1+\beta_{H_{cj}}(T-T_\mathrm{ref})]$. History is stored in reduced
units, so a temperature change rescales the field axis without altering any hysteron state. A later
commit at a different $T$ can then switch hysterons (coercivity-loss irreversible change). Thermally
activated (viscous) loss is not modelled.

### 2.2 Identification

$J_{s,\mathrm{irr}} = B_r/P_r$ with $P_r = 1-2E(+\infty,0)$, so $J(H{=}0)=B_r$ holds exactly on the
descending major branch. Two routes (`materials/library.py`):

(a) **Datasheet** (`fit_datasheet`): $H_\mathrm{scale}=H_{cj}$. Bounded least squares
(trust-region reflective) on $h_0$, the support $[h_0e^{-L},h_0e^{U}]$ of $f$ (which fixes $s$ and
$\kappa$), $r=w/(h_0 s)$ and $\mu_\mathrm{rec}\in[1,\mu_\mathrm{max}]$. Residuals are the relative errors
of $H_{cj}$, $H_{cb}$, $(BH)_\mathrm{max}$, $H_k$, each scaled by a quarter of its gate tolerance, plus
weak log-priors ($\sigma=1$) on $U,L,r$ that only decide what the datasheet leaves undetermined. A
prior on $\mu_\mathrm{rec}$ is fixed (N42, sourced), has $\sigma=25$ % (Alnico, Arnold class value),
or is absolute ($\sigma=0.1$) when the row has none. The support is bounded to $[h_0/10, 3h_0]$.
(b) **Curve** (`fit_curve`): unweighted least squares of the descending branch $B(H)$ to a tabulated
second-quadrant curve. $B_r$, $h_0$, $U$, $L$, $r$ and $\mu_\mathrm{rec}$ are free and there are no
priors. Standard errors come from $(J^\top J)^{-1}\,\mathrm{RSS}/(n-p)$ with the delta method for
log-parameters.

Alnico datasheet fits use the shape ($U,L,r$) identified on the FEMM LNG37 curve as their prior.
The datasheet $\mu_\mathrm{rec}$ is MMPA's average recoil slope. In this model a recoil from
$|H|<2H_{c,\min}$ is purely reversible, so that chord equals the model's $\mu_\mathrm{rec}$.

### 2.3 Grade library and sources

| family | grades | magnetic data | other data |
|---|---|---|---|
| sintered NdFeB | N35, N48, N42SH | Arnold Neo catalogue minima [arnold_neo] | Arnold thermal, $\rho_e$, $c_p$ |
| | N42 | reference case R2 parameters (old `params.py`) | $\mu_\perp$ from [katter2005] |
| | N52 | Eclipse minima (reference case R1) | Arnold temperature coefficients |
| bonded NdFeB | NdFeB-bonded | MQ1-10 [magnequench2023] | thermal, $\rho_e$, density are assumptions |
| cast Alnico | Alnico 2, 3, 5, 5-7, 8, 9 | MMPA 0100-00 Table II-1 [mmpa0100] | MMPA II-4/II-5; $\mu_\mathrm{rec}$ prior [arnold_alnico]; $c_p$ Dulong–Petit from composition |
| | LN10 … LNGT72 (14 rows) | supplier sheet, mapped to MMPA class | class data as above |
| | LNG37-femm | FEMM "Cast Alnico 5 (LNG37)" table, route (b) | Alnico 5 class |

Anisotropic sintered-magnet resistivity follows [ruoho2010]. Every assumption is recorded as
`# ASSUMPTION:` in the grade's `grade_info()`.

**Known identification deficiencies.**
- LN10 misses the $H_{cb}$ gate: +2.6 % against the 2 % tolerance. Br, $H_{cj}$ and $(BH)_\mathrm{max}$
  are within 0.15 %.
- Alnico 8 ends with $\mu_\mathrm{rec}$ on its lower bound ($\mu_\mathrm{rec}=1.000$, against the
  Arnold class prior 2.0). The datasheet targets are met (≤ 0.94 %), but the reversible permeability is
  not identified for that grade.
- The classical Preisach model has zero-width minor loops for small excursions: any recoil with
  $|H|<2H_{c,\min}$ of the switching-field support is purely reversible, so no small-loop hysteresis
  loss is represented.

## 3. Magnetostatics: nonlinear volume-integral method of moments

Each magnet is meshed into cells of uniform polarisation (`geometry.py`): cuboids into $n_x n_y n_z$
cuboids (optionally graded toward the faces); cylinders and rings into $n_r n_\phi n_z$ cells of exact
geometry. With $n_\phi=1$ the cells are full annuli (field = cylinder($r_2$) − cylinder($r_1$)
[caciagli2018]). With $n_\phi>1$ they are cylinder segments [slanovc2022]. A full cylinder keeps a
central full-cylinder core, so no cell has $r_\mathrm{in}=0$. Neighbouring cells share whole faces.

Following the Radia/MacMMems family [chubar1998, chadebec2006, carpentier2013], the cell-averaged
field is

$$\mathbf H_i = \mathbf H_{\mathrm{app},i} + \sum_k N_{ik}\mathbf J_k,\qquad
N_{ik} = \sum_g w_{ig}\,\mathbf H_k(\mathbf x_{ig})/|\mathbf J_k| .$$

$\mathbf H_k$ is the exact field of a uniformly polarised cuboid, cylinder or segment (magpylib
[ortner2020, ortner2023]). The target average uses $q^3$ tensor Gauss–Legendre points per cell
($q=2$ default; full annuli use $q$ radial × $q$ axial × $\max(2q,4)$ azimuthal points). magpylib's
$\mu_0$ (CODATA 2022) is rescaled to the project constant. Self-blocks are computed once in the body
frame and rotated ($R N R^\top$ per 3×3 block). Cross-blocks are rebuilt when a pose changes. The dense
matrix plus Jacobian is capped at 1 GB.

**Nonlinear solve.** The residual is $\mathbf R(\mathbf J)=\mathbf J-[J_\parallel(H_\parallel)\hat{\mathbf e}
+\mu_0\chi_\perp\mathbf H_\perp]$ in each cell's easy-axis frame. We use Newton with Jacobian
$I-DN$, $D=\partial_H J_\parallel\,\hat{\mathbf e}\hat{\mathbf e}^\top+\mu_0\chi_\perp(\hat{\mathbf e}_1\hat{\mathbf e}_1^\top+\hat{\mathbf e}_2\hat{\mathbf e}_2^\top)$,
solved by dense LU. A backtracking line search on $\max|\mathbf R|$ tries steps 1 down to $10^{-3}$. If
Newton fails, a continuation fallback scales $\mathbf H\to\lambda(\mathbf H_\mathrm{app}+N\mathbf J)$ for
$\lambda: 0\to1$, with adaptive steps of 0.1 (doubled up to 0.25 on success, halved on failure,
abandoned below $10^{-4}$). Convergence requires $\max|\mathbf R|<10^{-9}$ T and a last update
$\max|\Delta\mathbf J|<10^{-9}$ T. The default initial guess is the last solved $\mathbf J$.

**History commit.** `solve` never changes material state. `commit(sol)` passes each cell's
$H_\parallel$ to `material.commit`, so load protocols (saturate → open circuit → place at gap → pull
off) follow the true path, including Alnico recoil and irreversible loss.

## 4. Force and torque

With uniform $\mathbf J$ per cell, the only magnetic charges are surface charges on cell faces,
$\sigma = (\mathbf J_- - \mathbf J_+)\cdot\mathbf n$ (external faces $\mathbf J\cdot\mathbf n_\mathrm{out}$).
The force and torque on the target from all other bodies are

$$\mathbf F=\sum_\text{faces}\int\sigma\,\mathbf H_\mathrm{other}\,dA,\qquad
\mathbf T=\sum_\text{faces}\int(\mathbf r-\mathbf p)\times\sigma\,\mathbf H_\mathrm{other}\,dA .$$

$\mathbf H_\mathrm{other}$ is evaluated with the exact kernels. On external faces, evaluation points are
moved $10^{-10}\ell$ into the target, which gives the one-sided limit at contact.

*The torque already contains the couple.* The Kelvin torque density is
$\mathbf r\times(\mathbf J\cdot\nabla)\mathbf H+\mathbf J\times\mathbf H$. Integrating
$\mathbf r\times(\mathbf J\cdot\nabla)\mathbf H$ by parts turns it into the moment of the surface and
volume charges minus $\int\mathbf J\times\mathbf H\,dV$. The charge moment above therefore equals the
full Kelvin torque, and adding $\int\mathbf J\times\mathbf H\,dV$ again double counts. For a uniform
body in a uniform field, $\oint\mathbf r\times\sigma\mathbf H\,dA=V\mathbf J\times\mathbf H$ exactly.
Finite-difference virtual work confirms this (gate F5).

**Quadrature.** Each (target panel, source cell) pair is dispatched on its distance $d$ and the panel
size $L$:
- far ($d\ge L/5$): tensor Gauss–Legendre, order $2\le q\le30$ per direction from the Bernstein-ellipse
  bound $\rho^{-2q}\le10^{-8}$;
- near: parameter intervals are split at the projections of the source cell's edges, with tanh–sinh
  (37 nodes) on each sub-interval. This captures the endpoint log/jump singularities. For non-aligned
  near pairs, the pieces are capped at about $2d$, which is less accurate (an error estimate is
  available);
- near aligned cuboids (frames differing by a signed permutation): the source faces parallel to the
  target face interact through the exact Akoun–Yonnet rectangle–rectangle corner sums [akoun1984,
  allag2009], with the correct zero-gap one-sided limits. Only the perpendicular source charges are
  integrated numerically. For the torque, the lever-arm part $(\mathbf c_\mathrm{face}-\mathbf p)\times
  \mathbf F_\mathrm{face}$ uses the exact face force.

Independent references (`analytic.py`): full-vector Akoun–Yonnet cuboid force, coaxial-cylinder force
from Lipschitz–Hankel kernels [vokoun2009], a ported 1-D coaxial-rod slab solver, Aharoni prism
[aharoni1998] and Chen–Brug–Goldfarb cylinder [chen1991] magnetometric demagnetising factors, and the
dipole–dipole force.

## 5. Coil and eddy-current models

**Coil filaments** (`coil.py`) [grover1946, ruehli1972]. Every turn is a closed planar loop at
$z=\text{const}$ (circular, or a rounded rectangle approximated by `n_arc` chords per quarter arc).
Helical pitch, layer cross-overs and the foil spiral transition are ignored. `geometry_report()` bounds
the resulting error: path length $\approx (p/P)^2/2$, and the net axial current cancels in alternate
layers. Round wire uses one filament per turn, which is exact up to $O((r/R)^2)$ for uniform current.
Foil is split into near-square sub-filaments with equal current shares (DC). Mutual inductances use
Maxwell's elliptic-integral formula for coaxial circles and the closed-form Neumann integral for straight
segments (parallel and oblique). Self terms are the loop's mutual with a copy displaced by the
geometric mean distance of its cross-section: $re^{-1/4}$ for round wire, Grover's rectangle GMD for
foil. **Resistance**: $R(T)=\rho_\mathrm{Cu}(T)\,\ell/A$ with
$\rho_\mathrm{Cu}=1.7241\times10^{-8}\,[1+3.93\times10^{-3}(T-293.15)]$ Ω m (IACS, within 0.5 % of
[matula1979] over 250–400 K). Skin depth and the Dowell factor [dowell1966] are reported at the pulse's
dominant frequency, with a warning above 1.05. **Field per amp** comes from magpylib `Circle`/`Polyline`
filaments for round wire. Foil uses Gauss–Legendre current sheets across its thickness: exact cylinder
side walls for circular foil, analytic strip sheets otherwise.

**Eddy loops** (`eddy.py`). The magnet is cut into $n_z$ axial slabs, each holding $n_r$ nested bands
with uniform current density:
- cylinders and rings: annuli, $R_k=\rho_e 2\pi r_\mathrm{mid}/(\Delta r\Delta z)$;
- cuboids: bands between similar rectangles (stream function $\psi=\max(2|x|/A,2|y|/B)$),
  $R_k=4\rho_e s_\mathrm{mid}(A/B+B/A)/(\Delta s\,\Delta z)$, derived from $P=\int\rho J^2dV$. This
  assumes the currents follow similar rectangles.

Inductances reuse the coil filament machinery. Field per amp integrates exact uniformly polarised
bodies (a side-wall current sheet) across the band thickness. The network is linear, $(R+j\omega
M)\mathbf i = -j\omega\mathbf m\,i_s$. It is validated against the exact Bessel solution for a long
cylinder in a uniform axial AC field (gate C4).

## 6. Circuit

The KiCad schematic is exported with `kicad-cli sch export netlist --format spice` (or a given `.cir`
is used). `build_deck` keeps every base element and inserts between the coil nodes $a,b$:

$$a\;-\;L\;-\;V_\mathrm{sense}(0\,\mathrm V)\;-\;R(T)\;-\;V_\mathrm{EMF}\;-\;b .$$

- $R(T)$ is a behavioural source $V=i_cR_0[1+\alpha(T-T_\mathrm{ref})]$, with $T$ the voltage of an
  adiabatic thermal node. The node is fed by $i_c^2R(T)$ into $C_\mathrm{th}=mc_p$ and leaks
  through $10^{12}$ Ω to $T_0$.
- Eddy loop $k$ is a floating $L$–$R$–$V_k$ loop referenced to ground through 1 GΩ. All coil/loop
  couplings are `K` elements from the inductance matrix, after checks that $|k|<1$ and that the matrix
  is positive definite.
- $V_\mathrm{EMF}$ and $V_k$ are PWL sources carrying the magnet's flux-linkage rates.

Parts the schematic does not draw (capacitor bank, ESR, stimulus, `.ic`, `.options`, default
`method=gear, reltol=1e-4`) come from the scenario's `extra` lines. Model files are `.include`d with
absolute paths resolved at run time. ngspice 47 runs in batch mode with `wrdata` output [ngspice]. The
board snapshot is `res_exp/circuits/magnetiser/`.

## 7. Coupled switching transient

<!-- Implementation in src/magsim/switching.py (task 6); check this section against the final code. -->

**Magnet drive.** The magnet is the MoM + Preisach model of §§2–3, driven quasi-statically by
$\mathbf H_\mathrm{app}(\mathbf x,t)=\mathbf h_c(\mathbf x)i_c(t)+\sum_k\mathbf h_k(\mathbf x)i_k(t)$,
sampled at the MoM field points. The Preisach model is rate-independent, so the magnet is marched on
the field path: it is re-solved and committed wherever the applied field has changed by more than a
tolerance since the last step, not at every SPICE time point.

**Back-EMF by reciprocity.** The flux linkage of circuit $c$ (coil or eddy loop) due to the magnet is
$\lambda_c(t)=\sum_\text{cells}V\,\mathbf J\cdot\bar{\mathbf h}_c$, with $\bar{\mathbf h}_c$ the
cell-averaged field per amp of that circuit. Its rate $d\lambda_c/dt$ is written into the circuit's PWL
EMF source.

**Waveform relaxation.** Gauss–Jacobi waveform relaxation [lelarasmee1982]: run ngspice with the current
EMF waveforms, march the magnet on the resulting currents, update every EMF PWL, and repeat until
$\max|\Delta i|/\max|i|<10^{-4}$ (default, at most 15 iterations). The residual history is reported.

**Field definitions** ($\hat{\mathbf s}$ = target switching direction):
- generated $\mathbf H_\mathrm{gen}=\mathbf h_c i_c$;
- applied $\mathbf H_\mathrm{app}=\mathbf H_\mathrm{gen}+\mathbf H_\mathrm{eddy}$;
- felt $\mathbf H_\mathrm{felt}=\mathbf H_\mathrm{app}+\mathbf H_\mathrm{demag}$, the total $\mathbf H$
  in the material that drives the law, with $\mathbf B_\mathrm{in}=\mu_0\mathbf H_\mathrm{felt}+\mathbf J$.

For each, the report gives $\max_t\hat{\mathbf s}\cdot\mathbf H$ at the centroid, as a volume mean, and
at the weakest cell (minimum over cells, which decides full switching), in A/m and as $\mu_0H$ in T. The
switched fraction is $\langle J_\mathrm{irr}\cdot\hat{\mathbf s}\rangle/J_{s,\mathrm{irr}}$.

**Energy audit.** $\tfrac12CV_0^2=\text{losses}+\Delta E_C+W_\mathrm{magnet}+\text{residual}$, with
$W_\mathrm{magnet}$ computed two ways: $\int i\,e\,dt$ and $\iint\mathbf H_\mathrm{app}\cdot
d\mathbf J\,dV$.

## 8. Verification and validation

Gates from `docs/SPEC.md`; each is asserted by a test in `tests/`. Achieved values are maximum relative
errors unless stated.

| gate | check | achieved |
|---|---|---|
| P1 | library grades reproduce datasheet $B_r$, $H_{cj}$ (±0.5 %), $H_{cb}$, $(BH)_\mathrm{max}$, $H_k$ (±2 %) | all grades pass except LN10 $H_{cb}$ +2.6 % (§2.3) |
| P3 | fit to FEMM LNG37 table, RMS < 0.02 T | 0.96 mT RMS |
| P4 | $10^4$ cells × 1 step < 50 ms | 25–29 ms |
| F1 | rigid aligned cuboid pair vs Akoun–Yonnet, < 1e-5 | 2.4e-8 |
| F2 | rigid coaxial cylinders vs Lipschitz–Hankel, < 1e-4 | 1.8e-7 |
| F3 | open-circuit demag factor vs Aharoni / Chen–Brug–Goldfarb (Richardson) | $\mu=1$: ≤ 1e-5; $\mu=1.05$: ≤ 0.37 % |
| F4 | dipole limit at 20× size, < 0.1 % | 2.6e-5 |
| F5 | Newton's third law, < 1e-6; torque vs virtual work, < 1e-3 | ≤ 8e-10; 1.6e-8 |
| F6 | nonlinear Alnico rod: MoM vs 1-D slab solver | open-circuit $J$ +0.04…+0.16 %; force at 0.1 mm −0.54…−0.60 % |
| – | independent cross-check vs `magpylib.getFT` (F) | 7.7e-6 → 6.8e-7 as getFT meshing 4000 → 32000 |
| F6 | MoM vs axisymmetric FEM | same physics ($\mu_\perp=1$): LNG37 open-circuit $J$ 0.4839 vs 0.4852 T (0.3 %); linear $\mu_\mathrm{rec}=4$ ≤ 0.1 %. With $\mu_\perp=\mu_\mathrm{rec}$ the full-ring MoM mesh ($n_\phi=1$) cannot carry radial $J$: $J$ 0.4839 vs FEM 0.5090 T, force at 0.1 mm −1.482 vs −1.551 N (−4.4 %). FEM self-check vs Lipschitz–Hankel 2.8e-4, vs Chen 7.7e-5 |
| GCI | discretisation error, Richardson + GCI [celik2008] | Switching R1 (meshes 3×3×1, 5×5×3, 7×7×5): switched fraction GCI 0.29 %, weakest-cell felt field 0.42 %, relative remanence 0.04 %. E1 at 2 mm: $p$ = 0.54, GCI 0.89 %. E2 at contact: $F_z$ $p$ = 0.32, GCI 6.1 %; $\langle J\rangle$ $p$ = 1.70, GCI 0.56 %. Implementation reproduces Celik et al. Table 1 ($p$ = 1.53, GCI 2.2 %) |
| E1 / F7 | NdFeB block pair replay (fit, 95 % CI, RMS, χ²/dof) | one fitted parameter: $B_r$ = 0.860 T (Jacobian 95 % CI 0.854–0.867; bootstrap 0.857–0.868; mesh-limit 0.864, combined 0.856–0.872 T); RMS 0.240 N, max residual 0.85 N, χ²/dof 0.96 (dof 26, σ = 0.25 N). Library N42 ($B_r$ 1.32 T) overpredicts by 2.4× |
| E2 / F7 | LNG37 rod pair replay | no fitted parameter. Contact force measured 2.00 N; predicted 1.88 (MoM) / 1.95 (FEM) N with vendor LNG37, 1.52 / 1.58 N with LNG37-femm. χ²/dof (dof 27), gap = CSV mm: 2.52 / 1.73 (MoM, LNG37 / femm); gap = 0.10 mm sheets: 0.60 / 1.20 — the sheets reading fits. Predicted per-pull-off decay ≈ 0 (classical Preisach, no accommodation); measured repeat-to-repeat changes are within quantisation |
| C1 | single-layer solenoid vs Lorenz/Nagaoka, < 1e-4; thick coil vs Grover, < 1e-3 | 5.2e-7…5.2e-5; Brooks coil vs Grover 3.7e-5 |
| C2 | on-axis fields vs closed form, < 1e-8 | ≤ 7e-15 |
| C3 | R1 foil coil vs old validated model | $L$ vs old air-core 4e-5; $R_{20}$ 1e-6 |
| C4 | eddy loops vs long-cylinder Bessel, < 3 % | 0.14 % ($\delta=a$), 0.88 % ($f=1/\tau$); 12.2 / 3.0 / 0.88 / 0.36 % for $n_r$ = 2 / 4 / 8 / 16 |
| S1 | RLC vs analytic, < 0.1 %; pipeline vs direct ngspice | RLC 2.3e-8; pipeline vs direct 7e-6; thermal 2.7e-6; two-mesh 1.6e-7 |
| S2 | WR converges < 1e-4 in ≤ 15 iterations, independent of initial guess | R1: 5–7 iterations, residual 2e-5–7e-5 (both SCR models); zero vs 0.5× converged initial EMF differ by 4.4e-5; R3 10 iterations. Production runs fold a linear reference response into SPICE (splitting); plain Gauss–Jacobi diverges for Alnico |
| S3 | energy residual < 0.5 % of $\tfrac12CV_0^2$ | R1: 1e-6 of $E_0$ = 78.97 J (both SCR models); $W_\mathrm{mag}$ by $\int i\,e\,dt$ and by $\sum V\!\int H_\mathrm{app}\cdot dJ$ agree (4.155 / 4.155 J) |
| S4 | linear magnet: WR vs all-SPICE folded inductance, < 0.5 % | small case (test suite): passes with and without eddy loops. R1 with μ = 4 (dL/L = 0.20, strongest coupling): 0.57 % (field step 2.5e-3), 0.50 % (1.25e-3); without splitting the WR is not converged in 15 iterations (open item). Sign check passes (dL = +3.32 µH) |
| S5 | R1–R3 end-to-end from `magsim switch run` | R1, R2, R3 run from the CLI (≈ 2 min each); R2 layer sweep (8 points), R3 bus-voltage sweep (10), R1 voltage × temperature map (20): 0 failed. ngspice "Timestep too small" at bridge turn-off is handled by one retry with 1 GΩ node shunts (`rshunt`), kept for the rest of the run |

## 9. Limitations

- **No soft iron.** High-permeability keepers are a known accuracy trap for volume-integral MoM and
  are left to FEM.
- **Scalar Preisach per cell.** It has no vector (rotational) hysteresis: transverse response is
  linear ($\mu_\perp$), and off-axis fields switch only through $H_\parallel$. Minor loops have zero
  width below $2H_{c,\min}$ (§2.3). The stack depth is finite (innermost reversal dropped when full).
  Magnetic viscosity, ageing and thermally activated loss are not modelled. Temperature enters only
  through reversible coefficients.
- **Identification.** Alnico recoil permeabilities are class values from another manufacturer used as
  priors. Alnico 8 sits on the $\mu_\mathrm{rec}$ bound. LN10 misses $H_{cb}$ by 2.6 %. Several bonded
  NdFeB physical properties are assumptions.
- **Uniform cells.** Polarisation is piecewise constant, and accuracy follows mesh refinement (GCI
  reported). Near-field quadrature for non-aligned near pairs is less accurate than the aligned
  closed form.
- **Coil.** It is a filament model of planar turns. Helical pitch, cross-overs and the foil spiral
  transition are ignored (bounded in `geometry_report`). Current density is DC-uniform: skin and
  proximity effects are only reported (Dowell factor), not modelled.
- **Eddy currents.** A linear loop network with an assumed current pattern (annular or
  similar-rectangle bands), not 3-D diffusion. For cuboids the similar-rectangle path is exact only for
  a square in the thin-band limit. Eddy resistivity is temperature-independent.
- **Circuit.** Device models are behavioural, fitted to datasheet numbers. The CLA100E1200HB thyristor in
  the board snapshot (`res_exp/circuits/magnetiser/magnetiser.lib`) is a conductance-controlled switch with
  a smooth (tanh) gate/holding-current latch. It has no dv/dt, $t_q$ or reverse-recovery modelling, and
  its turn-on delay is about 1 µs. The original model's conductance rises in proportion to the gate drive, so with the board's RG = 2.2 kΩ it settles half-on (gate current 30.5 mA < 40 mA, $V_{AK}$ ≈ 70 V) and never reaches its hold threshold: 1381 A peak, 172.7 V left on the bank. `magnetiser_regen.lib` replaces it with a regenerative latch (I_GT 40 mA, I_L 150 mA, I_H 100 mA, delay shaped on datasheet Fig. 5, on-state refitted to the p. 2 table): it latches at 2.2 kΩ and 470 Ω. Because the gate is anode-fired, each thyristor needs $V_{AK}$ ≈ 92 V to fire at 2.2 kΩ, so the bus must exceed ≈ 185 V. Reference switching results use the latching model and are also reported with the original; the on-state voltage at kA is extrapolated beyond the datasheet table (200 A). The opto-triac turn-on (1 µs in the model, 30–70 µs per the
  firmware note) is an open question.
- **Thermal.** The coil thermal node is adiabatic and lumped (no conduction to the former), so it is
  valid for single pulses only.
- **Coupling.** The magnet is quasi-static: no magnetic relaxation or rate dependence beyond the eddy
  network. Mechanics (magnet motion during the pulse) is not modelled.
