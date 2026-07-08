# Why is the steady radiative manifold rank three? A predictive argument, pre-registered

*Working note for discussion. Two pages. Everything here is reproducible from
`scripts/`; the pre-registered predictions in §3 were written and committed before
`scripts/19_rank_prediction.jl` was run (see git history).*

## 1. The finding

For the steady heat equation on a fixed 2D radiator section with a Stefan–Boltzmann
boundary condition ($-k\,\partial_n u = \varepsilon\sigma(u^4 - T_s^4)$, deep-space sink),
sampled over $\mu = (\varepsilon, Q) \in [0.1, 0.95] \times [100, 800]$ W/m, the centered
snapshot matrix has singular values

$$\sigma_2/\sigma_1 = 1.15\cdot 10^{-1}, \qquad \sigma_3/\sigma_1 = 7.2\cdot 10^{-9},
\qquad \sigma_5/\sigma_1 = 6\cdot 10^{-16}.$$

The manifold is numerically **rank 2 after centering** (three shapes counting the mean),
despite temperatures spanning 300–970 K and a strongly nonlinear BC. This is not a
discretization artifact: the ratios replicate on a 2× finer grid (6.7e-9) and with P2
elements (6.8e-9), and every downstream conclusion survives 5-seed retraining (N = 5,
StableRNG, seed table in the report appendix).

## 2. The argument

Two structural facts about this problem class, not about radiation specifically:

**(a) Interior linearity.** With constant $k$ the PDE is linear inside $\Omega$; all
nonlinearity sits on $\Gamma_r$. Decompose $u = (Q/k)\,w + h$, where $-\nabla^2 w = \hat q$
is a fixed particular response to the unit load shape (parameter-independent!) and $h$ is
harmonic with the radiation BC absorbing the nonlinearity. The load-dependent part of the
manifold is therefore *one exact direction*, however nonlinear the BC.

**(b) Boundary-layer poverty of the harmonic part.** $h$ is harmonic on a rectangle,
insulated on three sides, with flux only through $\Gamma_r$. Because conduction smooths
the edge temperature, that flux is nearly uniform along $\Gamma_r$, and the harmonic
extension of nearly-uniform edge data is spanned by very few shapes (a level and a tilt).
The parameters move the *coefficients* of these shapes (strongly nonlinearly, e.g. the
radiative-equilibrium level $\propto (Q/\varepsilon)^{1/4}$), but coefficient nonlinearity
is invisible to POD rank; only *shape* variation counts.

Linearizing the BC around a reference state (Robin coefficient $4\varepsilon_0\sigma
u_0^3$) makes this quantitative: to first order in the parameter box the centered manifold
lies in $\mathrm{span}\{w_Q, w_\varepsilon\}$ (rank $d$ for $d$ parameters), and
$\sigma_{d+1}/\sigma_1$ is the second-order Taylor remainder, controlled by the box size
and the curvature of the parameter-to-solution map.

## 3. Pre-registered predictions (written before running the experiments)

The argument is only worth discussing if it predicts. Three experiments, three *different*
expected outcomes:

**P1 -second heater patch.** $\mu = (\varepsilon, Q_1, Q_2)$ with a second, disjoint
source patch. By (a), each load contributes its own exact direction. **Predict: centered
rank 3** (two source responses + level), i.e. $\sigma_3/\sigma_1 = O(10^{-1\pm 1})$ now a
*real* mode, and the remainder cliff moves to $\sigma_4/\sigma_1 = O(10^{-8\pm 1})$, with
$\sigma_5$ near machine precision.

**P2 -conductivity as the third parameter.** $\mu = (\varepsilon, Q, k)$, $k \in [2, 10]$.
The naive "$d+1$ shapes" rule says rank 4. The structural argument says otherwise: by (a),
$Q$ and $k$ enter the particular solution only through $Q/k$, and the level is set by a
$k$-free energy balance, so $k$ mostly re-parametrizes existing directions. **Predict: no
new $O(\sigma_2)$-scale mode; the spectrum looks like the 2-parameter one, with
$\sigma_4/\sigma_1 < 10^{-5}$** (weak new curvature is admitted, a hard fourth mode is
not). This is the falsification test that distinguishes the structural argument from
parameter counting: if a strong fourth mode appears, §2(a) is wrong as stated.

**P3 -box shrinking.** Shrink the $\varepsilon$-box (halfwidth full/½/¼ around 0.525) at
fixed $Q$-box, and separately the $Q$-box (around 450) at fixed $\varepsilon$-box. If
$\sigma_3$ is a second-order remainder of the joint map, halving a box side that
participates in the dominant curvature should reduce $\sigma_3/\sigma_1$ by ×2–×4
(order between 1 and 2); a box axis that does not participate should leave it nearly
unchanged. **Predict: log–log slope in [1, 2] on at least one axis; the pair of slopes
identifies which curvature (ε-, Q-, or cross-term) dominates the remainder.** We commit to
reporting both slopes whatever they are.

## 4. Results (figure `20_rank_prediction.png`; scorecard, misses included)

**P1 -hit on structure, miss on remainder magnitude.** With the second heater,
$\sigma_3/\sigma_1 = 3.5\cdot 10^{-2}$ becomes a genuine mode and the cliff moves to
$\sigma_4/\sigma_1 = 8.3\cdot 10^{-6}$, $\sigma_5 = 2.2\cdot 10^{-9}$: centered rank 3 as
predicted. The new remainder is however ~2 orders above the predicted $O(10^{-8\pm1})$
band; in hindsight the two loads double the total-load excursion, and the second-order
term grows with it. Structural prediction confirmed, magnitude underestimated.

**P2 -strong hit; the falsification test passed.** With $\mu = (\varepsilon, Q, k)$ the
spectrum is indistinguishable in shape from the 2-parameter one:
$\sigma_3/\sigma_1 = 2.3\cdot 10^{-8}$ (remainder level), $\sigma_4/\sigma_1 =
3.2\cdot 10^{-14}$. The naive $d{+}1$ rule predicts a fourth $O(\sigma_2)$ mode;
the $Q/k$-collapse argument predicted none; none appears, with nine orders of margin.
This is the experiment that distinguishes the structural argument from parameter counting.

**P3 -informative miss.** Shrinking boxes reduces the remainder far more slowly than a
second-order Taylor term should: log–log slopes 0.18–0.27 on the ε-axis and 0.68 on the
Q-axis (predicted: at least one axis in [1, 2]). The Q-axis dominance matches the
edge-flux-nonuniformity mechanism, but the scaling says the parameter box is not in the
asymptotic small-perturbation regime, so $\sigma_3$ is governed by something coarser than
the local quadratic remainder. We report this as a miss; understanding the sub-asymptotic
scaling is the most interesting open thread the experiments produced.

## 5. The transient contrast, honestly

Under switching load cycles the space-time manifold re-thickens ($\sigma_8/\sigma_1
\approx 3\cdot 10^{-5}$, smooth decay over 60 modes). This is *consistent* with the
argument: each trajectory sweeps a continuum of transient states that no small fixed
family spans, and the switching makes the phase dependence non-smooth, which is exactly
when Kolmogorov widths decay slowly; but we do not claim it as a prediction of the
linearization, which is a steady, smooth-family statement.

## 6. Where the surrogate error actually lives

Exact reduced-Galerkin solves (Newton on $\Phi_r^\top R(\bar u + \Phi_r c) = 0$; no
hyper-reduction needed offline at $n = 1891$; DEIM would be the online path) give the
honest floor for *any* surrogate confined to the POD subspace. Mean over 102 held-out
parameters, all solves converged:

| rank | L² projection floor | Galerkin floor | POD-MLP error | map share of error |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.7e-2 | 2.4e-2 | 1.7e-2 | basis-limited |
| **2** | 1.14e-9 | **1.14e-9** | 9.2e-4 | **100.00%** |
| 3 | 6.2e-16 | 2.8e-13 | 1.3e-3 | 100.00% |
| 5 | 1.8e-16 | 2.8e-13 | 1.7e-3 | 100.00% |

Two readings. At the manifold rank and above, the Galerkin floor *equals* the projection
floor, so every last part of the deployed surrogate's error is coefficient-map error;
the basis contributes nothing measurable. Below the manifold rank ($r = 1$) the picture
inverts: Galerkin is even slightly worse than L² projection (it optimizes the projected
residual, not the error) and the neural mapper already sits at the basis floor. The
crossover is the diagnostic: it tells you, per rank, whether to spend effort on the basis
or on the map.

Supporting statistics (all in `scripts/20_statistics.jl`; N stated, StableRNG seed table
in the report appendix): design-ranking Spearman with 10,000-resample bootstrap CIs,
0.9984 [0.9960, 0.9994] and 0.9983 [0.9965, 0.9992] at the two operating points; paired
bootstrap on per-sample errors, KAN − MLP = +2.0e-3 [1.6, 2.4]e-3 and direct − MLP =
+4.9e-3 [4.4, 5.4]e-3 (both exclude zero); ablations at r = 2: whitening off degrades
the mapper 200× (2.1e-1), and a random orthonormal basis caps accuracy at exactly its
own projection floor (2.13e-1), the basis-quality control.

## 7. Questions

1. Is the §2 argument a known result in the RB literature in this form (interior-linearity
   + boundary-layer poverty giving *hard* rank cliffs rather than generic fast decay), or
   worth writing up?
2. If worth writing: is the right shape a short numerical-analysis note (JCP/CMAME-style)
   or a workshop paper with the surrogate diagnostics attached?
3. Which baseline would you want to see: DeepONet on the same snapshots (our plan), full
   DEIM, or something else?
