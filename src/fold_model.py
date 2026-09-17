"""The fold: what the Li et al. data actually support.

Non-dimensionalise Eq. 5 with n = N/d and tau = r*m*t and you get

    dn/dtau = n(1 - n/kappa) - rho*n^2/(n^2 + 1),
        kappa = K/(e d)        rho = P k/(r m d)

Four parameters, two groups. The two-stage fit then drives kappa to infinity,
since nothing in the data comes within eleven orders of magnitude of K/e, and
what's left is a one-parameter system

    dn/dtau = n - rho n^2/(n^2 + 1)

with interior equilibria solving n^2 - rho n + 1 = 0:

    n_+- = [rho +- sqrt(rho^2 - 4)]/2,      real iff rho >= 2.

A saddle-node in one control. The simplest object in the elementary catastrophe
hierarchy, not the cusp the paper fits nor the swallowtail above it, and rho = 2
is the same P* = 2 d r m/k threshold that keeps turning up. It is also Ludwig,
Jones & Holling (1978), which is where insect-outbreak bistability has lived
since. The paper rules the fold out in its opening move, on the grounds that
"stable behavior cannot be observed" in it, which is backwards here.

What's implemented:

    dN/dt = r(e) N (1 - N/K) - P k N^2/(N^2 + d^2),     r(e) = r0 + a*e

Linear r(e) matters. e is standardised on [-3.6, +1.9], so this is its
first-order expansion, and unlike an exponential it can go negative: in cold,
wet weather the population shrinks on its own. That mortality is real (the paper
itself blames rainfall washing aphids off the crop in 2002) and has to live
somewhere.

An earlier cut put weather on predation instead, k(e) = k0 exp(c e). It fitted
better, pooled log-R^2 0.645 against 0.563, and was nonsense: the optimiser
bought low-e mortality by inflating predation to a median of 540 and a maximum
of 45,000 aphids per predator unit per day, against 50-150 in the literature for
an adult Coccinella septempunctata. An unbounded exponential on a rate with a
hard biological ceiling always gets abused that way. So k is a bounded constant.

K is fixed from agronomy rather than fitted; sensitivity_K reports how little it
matters. Fitting it is exactly what hollowed out the published model.

The ceiling isn't cosmetic. Without it the two interior equilibria are a stable
refuge and an unstable escape threshold, above which N grows without bound. With
it, the threshold is followed by a third stable equilibrium near K, the outbreak
branch, completing the Ludwig-Jones-Holling picture. The closed-form roots track
the refuge to within a couple of percent and idealise the upper branch, which is
the ceiling's business.

Insecticide is an impulse N -> m N, so it doesn't enter rho: spraying knocks the
population down, it doesn't move the bistability threshold.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, least_squares

import gpd

K_FIXED = 1.0e4      # aphids/100 stems; largest ever recorded is 7480
N_CAP = 5.0e5

# Trajectory-matched on 1997-2001. Pooled log-R^2 = 0.563 on those years,
# negative on the held-out 2002. See README for the full read.
FITTED = dict(r0=0.2888, a=0.1205, k=156.40, d=906.33)

PARAM_NAMES = ("r0", "a", "k", "d")


def r_of(e, beta):
    """Intrinsic rate. Goes negative when the weather is bad enough."""
    return beta["r0"] + beta["a"] * np.asarray(e, dtype=float)


def k_of(e, beta):
    """Max consumption per predator unit per day. No weather dependence."""
    return np.full_like(np.asarray(e, dtype=float), beta["k"], dtype=float)


def rho(e, P, beta):
    """The single control: predation pressure over growth, scaled by d.

    rho >= 2 is bistable. Where r(e) <= 0 the ratio is meaningless and inf comes
    back, since the population is declining on its own and predators aren't what
    is holding it.
    """
    rv = r_of(e, beta)
    return np.where(rv > 0,
                    np.asarray(P, float) * k_of(e, beta) / (np.where(rv > 0, rv, 1.0)
                                                            * beta["d"]),
                    np.inf)


def dNdt(N, e, P, beta, K=K_FIXED):
    N = np.asarray(N, dtype=float)
    growth = r_of(e, beta) * N * (1.0 - N / K)
    predation = np.asarray(P, float) * k_of(e, beta) * N ** 2 / (N ** 2 + beta["d"] ** 2)
    return growth - predation


def equilibria(e, P, beta, closed=True, K=K_FIXED):
    """Interior equilibria, ascending. Empty when rho < 2.

    closed=True solves n^2 - rho n + 1 = 0, the K -> inf idealisation that makes
    the fold transparent: refuge and escape threshold, appearing together at
    rho = 2. closed=False solves the full cubic including the ceiling, which
    adds the stable outbreak branch near K.
    """
    rr = float(rho(e, P, beta))
    if closed:
        # inf where r(e) <= 0, i.e. nothing predator-held to report
        if not np.isfinite(rr) or rr < 2.0:
            return np.array([])
        disc = np.sqrt(rr * rr - 4.0)
        return beta["d"] * np.array([(rr - disc) / 2.0, (rr + disc) / 2.0])
    # with the ceiling: r(1-N/K)(N^2+d^2) = P k N, after dividing out N
    rv, kv = float(r_of(e, beta)), float(k_of(e, beta))
    if rv <= 0:
        return np.array([])
    coef = [-rv / K, rv, -(P * kv + rv * beta["d"] ** 2 / K), rv * beta["d"] ** 2]
    rts = np.roots(coef)
    rts = rts[np.abs(rts.imag) < 1e-9 * (1 + np.abs(rts.real))].real
    return np.sort(rts[rts > 0])


def stability(roots, e, P, beta, K=K_FIXED, h=1e-4):
    return np.array([dNdt(x * (1 + h), e, P, beta, K) < dNdt(x * (1 - h), e, P, beta, K)
                     for x in np.atleast_1d(roots)])


def threshold_P(e, beta):
    """Predator density at the fold, rho = 2. Zero where r(e) <= 0."""
    return np.maximum(2.0 * r_of(e, beta) * beta["d"] / k_of(e, beta), 0.0)


def simulate(df_year, weather, scaler, beta, year, N0=None, t_span=None,
             spray=None, P_scale=1.0, n_out=400, t_eval=None, K=K_FIXED,
             rtol=1e-8, atol=1e-5):
    """One season, insecticide as an impulse N -> m N."""
    e_of, P_obs, _m = gpd.forcing(df_year, weather, scaler, year)
    if spray is None:
        sched = ((float(gpd._day_index(*gpd.SPRAY_DATES[year])), gpd.SPRAY_SURVIVAL)
                 if year in gpd.SPRAY_DATES else None)
    elif spray == ():
        sched = None
    else:
        sched = (float(spray[0]), float(spray[1]))

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
        pts = t_eval[(t_eval >= a_) & (t_eval <= b_)]
        sol = solve_ivp(lambda t, y: [dNdt(max(y[0], 0.0), e_of(t),
                                           P_scale * P_obs(t), beta, K)],
                        (a_, b_), [ya], t_eval=pts, events=cap, method="Radau",
                        rtol=rtol, atol=atol)
        end = sol.y[0, -1] if sol.y.shape[1] else ya
        return sol.t, sol.y[0], end, bool(sol.t_events[0].size)

    if sched is None or not (t0 < sched[0] < t1):
        t, N, _e, _h = leg(t0, t1, y0)
        return t, np.clip(N, 0.0, None)
    ts, surv = sched
    ta, Na, end, hit = leg(t0, ts, y0)
    if hit:
        return ta, np.clip(Na, 0.0, None)
    tb, Nb, _e, _h = leg(ts, t1, end * surv)
    return np.concatenate([ta, tb]), np.clip(np.concatenate([Na, Nb]), 0.0, None)


def _unpack(theta):
    return dict(r0=theta[0], a=theta[1], k=theta[2], d=theta[3])


def _residuals(theta, df, weather, scaler, years, fast=False):
    beta = _unpack(theta)
    kw = dict(rtol=1e-6, atol=1e-3) if fast else {}
    out = []
    for year in years:
        g = df[df.year == year].sort_values("t")
        ts = g.t.to_numpy(float)
        t, N = simulate(g, weather, scaler, beta, year, t_eval=ts, **kw)
        pred = np.interp(ts, t, N) if len(t) != len(ts) else N
        out.append(np.log1p(np.clip(pred, 0, N_CAP)) - np.log1p(g.N.to_numpy(float)))
    return np.concatenate(out)


def fit(df, weather, scaler, years=gpd.VALIDATION_YEARS, seed=1, maxiter=45,
        popsize=14):
    def cost(theta):
        try:
            r = _residuals(theta, df, weather, scaler, years, fast=True)
        except Exception:
            return 1e6
        return float(r @ r)

    # k is held to what an adult C. septempunctata actually eats, roughly
    # 50-150 aphids/day. Leaving it free invites the optimiser to manufacture
    # mortality it has no biological right to.
    bounds = [(0.02, 1.5),     # r0
              (0.0, 1.5),      # a
              (30.0, 200.0),   # k
              (50.0, 4000.0)]  # d
    de = differential_evolution(cost, bounds, seed=seed, maxiter=maxiter,
                                popsize=popsize, tol=1e-8, polish=False)
    ls = least_squares(_residuals, de.x, args=(df, weather, scaler, years),
                       bounds=tuple(zip(*bounds)), xtol=1e-12, ftol=1e-12,
                       diff_step=1e-4)
    return _unpack(ls.x), float(ls.cost * 2)


def sensitivity_K(df, weather, scaler, beta, years=gpd.VALIDATION_YEARS,
                  Ks=(8e3, 1e4, 2e4, 5e4, 1e5)):
    """Fit quality as the fixed ceiling varies. It should barely move."""
    rows = []
    for K in Ks:
        obs, pred = [], []
        for year in years:
            g = df[df.year == year].sort_values("t")
            t, N = simulate(g, weather, scaler, beta, year, K=K)
            obs += list(g.N)
            pred += list(np.interp(g.t.to_numpy(float), t, N))
        rows.append(dict(K=K, r2_log=gpd.r_squared(np.log1p(obs), np.log1p(pred))))
    return rows
