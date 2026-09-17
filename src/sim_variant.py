"""GPD-S: Eq. 5 rearranged into something you can actually integrate.

The published model is fitted to instantaneous rates and never simulated. Try to
integrate it and it runs away, for three reasons that compound:

  * the density brake is inert. At N = 500 the correction 1 - eN/K is 1.000018
    for e = -1 and 0.999982 for e = +1, because the fitted K ~ 2.75e7 is five
    orders of magnitude past anything observed. The paper's own standard error
    for K, 4.2e-10, says the data cannot see it at all.
  * predation saturates at P*k, so growth r*N wins permanently once N > P*k/r,
    about 1200 aphids at P = 5 PU.
  * what's left is r ~ 0.39/day, doubling every 1.8 days.

The sign of e only decides how it ends. For e > 0 there's an equilibrium at
K/e ~ 1e7, finite but not a field density; for e < 0 nothing positive exists and
growth turns quadratic. A season blows up at e = 0 too.

So: move e onto the growth rate and make the density term a real logistic.

    dN/dt = r0*exp(a*e)*m*N*(1 - N/K) - P*k*N^2/(N^2 + d^2)

Wu et al. 2014 took the same route with r as a function of temperature. The
exponential keeps r positive for a standardised index. What comes out is the
Ludwig-Jones-Holling form, with the refuge / threshold / outbreak structure the
paper is actually talking about, at densities that occur in a wheat field.

Insecticide is an impulse N -> m*N, which is how the paper defines m in Section
2.4.3. Folding it into the growth rate for one day, as they do "for simplicity",
removes a fraction of a percent rather than half.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, least_squares

import gpd

N_CAP = 5e5

# Trajectory-matched on 1997-2001, pooled log-R^2 = 0.61, and it does not carry
# over to 2002. Two caveats travel with these numbers:
#   K sits on its lower bound. The optimiser would rather cap every season than
#   reach the 5870 (2000) and 7480 (2002) peaks, which says the forcing doesn't
#   separate the quiet years from those two.
#   k and d are only identified through k/d^2 = 3.13e-4. Over most of the range
#   N << d, so the Holling term degenerates to (P k/d^2) N^2 and the cost moves
#   1.2% across an eightfold range of d. Same disease as K in the original.
FITTED = dict(r0=0.4448, a=1.1263, K=7943.3, k=37352.0, d=10923.9)


def rhs(N, e, P, m, r0, a, K, k, d):
    N = max(float(N), 0.0)
    growth = r0 * np.exp(a * e) * m * N * (1.0 - N / K)
    predation = P * k * N ** 2 / (N ** 2 + d ** 2)
    return growth - predation


def equilibria(e, P, m, beta, n_grid=20001, n_max=None):
    """Positive equilibria, from sign changes of the rate on a grid."""
    n_max = n_max or beta["K"] * 1.2
    N = np.linspace(1e-6, n_max, n_grid)
    f = np.array([rhs(n, e, P, m, **beta) for n in N])
    idx = np.nonzero(np.sign(f[:-1]) * np.sign(f[1:]) < 0)[0]
    roots = []
    for i in idx:
        lo, hi = N[i], N[i + 1]
        roots.append(lo + (hi - lo) * f[i] / (f[i] - f[i + 1]))
    return np.array(roots)


def stability(roots, e, P, m, beta, h=1e-3):
    return np.array([rhs(x * (1 + h), e, P, m, **beta)
                     < rhs(x * (1 - h), e, P, m, **beta) for x in roots])


def simulate(df_year, weather, scaler, beta, year, N0=None, t_span=None,
             spray=None, n_out=400, P_scale=1.0, t_eval=None, rtol=1e-8,
             atol=1e-5):
    """One season. `spray` overrides the schedule as (day, survival), () for none.

    `P_scale` multiplies the observed enemy series, for augmentation scenarios.
    """
    e_of, P_obs, _m_obs = gpd.forcing(df_year, weather, scaler, year)
    if spray is None:
        sched = ((float(gpd._day_index(*gpd.SPRAY_DATES[year])), gpd.SPRAY_SURVIVAL)
                 if year in gpd.SPRAY_DATES else None)
    elif spray == ():
        sched = None
    else:
        sched = (float(spray[0]), float(spray[1]))

    def P_of(t):
        return P_scale * P_obs(t)

    t0 = float(df_year.t.min()) if t_span is None else t_span[0]
    t1 = float(df_year.t.max()) if t_span is None else t_span[1]
    y0 = max(float(df_year.N.iloc[0]) if N0 is None else N0, 1.0)
    if t_eval is None:
        t_eval = np.linspace(t0, t1, n_out)
    t_eval = np.asarray(t_eval, dtype=float)

    def cap(t, y):
        return y[0] - N_CAP
    cap.terminal, cap.direction = True, 1

    def leg(a_, b_, ya):
        """One segment, no discontinuity inside it."""
        pts = t_eval[(t_eval >= a_) & (t_eval <= b_)]
        sol = solve_ivp(lambda t, y: [rhs(y[0], e_of(t), P_of(t), 1.0, **beta)],
                        (a_, b_), [ya], t_eval=pts, events=cap, method="Radau",
                        rtol=rtol, atol=atol)
        end = sol.y[0, -1] if sol.y.shape[1] else ya
        return sol.t, sol.y[0], end, bool(sol.t_events[0].size)

    if sched is None or not (t0 < sched[0] < t1):
        t, N, _end, _hit = leg(t0, t1, y0)
        return t, np.clip(N, 0.0, None)

    t_spray, surv = sched
    ta, Na, end, hit = leg(t0, t_spray, y0)
    if hit:
        return ta, np.clip(Na, 0.0, None)
    tb, Nb, _end, _hit = leg(t_spray, t1, end * surv)
    return (np.concatenate([ta, tb]),
            np.clip(np.concatenate([Na, Nb]), 0.0, None))


def _residuals(theta, df, weather, scaler, years, fast=False):
    beta = dict(r0=theta[0], a=theta[1], K=10 ** theta[2], k=theta[3], d=theta[4])
    kw = dict(rtol=1e-6, atol=1e-3) if fast else {}
    out = []
    for year in years:
        g = df[df.year == year].sort_values("t")
        ts = g.t.to_numpy(float)
        t, N = simulate(g, weather, scaler, beta, year, t_eval=ts, **kw)
        pred = np.interp(ts, t, N) if len(t) != len(ts) else N
        out.append(np.log1p(np.clip(pred, 0, N_CAP)) - np.log1p(g.N.to_numpy(float)))
    return np.concatenate(out)


def fit(df, weather, scaler, years=gpd.VALIDATION_YEARS, seed=1, maxiter=40,
        popsize=12):
    """Global search then local polish on log-scale trajectory residuals.

    Log scale because the counts span four decades within a season; linear
    residuals would only see the peaks.
    """
    def cost(theta):
        try:
            r = _residuals(theta, df, weather, scaler, years, fast=True)
        except Exception:
            return 1e6
        return float(r @ r)

    # log10 K bounded below by the largest observed density. Anything smaller
    # can't reproduce the data, and an unbounded search games the log residuals
    # by capping every season.
    bounds = [(0.02, 2.0), (0.0, 3.0), (3.9, 5.5), (0.1, 2000.0), (1.0, 5000.0)]
    de = differential_evolution(cost, bounds, seed=seed, maxiter=maxiter,
                                popsize=popsize, tol=1e-8, polish=False)
    ls = least_squares(_residuals, de.x, args=(df, weather, scaler, years),
                       bounds=tuple(zip(*bounds)), xtol=1e-12, ftol=1e-12,
                       diff_step=1e-4)
    theta = ls.x
    beta = dict(r0=theta[0], a=theta[1], K=10 ** theta[2], k=theta[3], d=theta[4])
    return beta, float(ls.cost * 2)
