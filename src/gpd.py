"""Li et al. 2020, BioSystems 198:104217, reimplemented.

    dN/dt = r*m*N*(1 - e*N^theta/K) - P*k*N^delta/(N^delta + d^delta)     [Eq. 5]

theta=1, delta=2 is the combination the authors picked; substituting
N = x + K/(3e) kills the quadratic term and leaves the cusp x^3 + ux + v.

N is aphids per 100 stems, e the weather index, P natural enemies in predator
units, m the insecticide survival rate. Data comes from the paper's two
supplementary spreadsheets.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import BSpline, make_interp_spline
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = Path.home() / "Downloads"
SURVEY_XLSX = DOWNLOADS / "1-s2.0-S0303264720301088-mmc1.xlsx"   # dataset S1
WEATHER_XLSX = DOWNLOADS / "1-s2.0-S0303264720301088-mmc2.xlsx"  # dataset S2

VALIDATION_YEARS = (1997, 1998, 1999, 2000, 2001)
TEST_YEAR = 2002

# Table 1. Parasitoids were recorded as mummified aphids.
PREDATOR_UNITS = {"lady": 1.0, "spid": 0.25, "hover": 0.5, "lace": 0.5,
                  "para": 1.0 / 120.0}

SPRAY_DATES = {1999: (5, 3), 2000: (5, 1), 2001: (4, 25), 2002: (5, 3)}
SPRAY_SURVIVAL = 0.5

# The four indices Table A1's Spearman screening kept, ordered by |rho|.
# sign: +1 positive index (Eq. 6), -1 negative (Eq. 7).
# power: position in the butterfly membership set, x_a = z^1/2 ... x_d = z^1/5.
CP_INDICES = [
    ("T10-average", +1, 2),   # temperature, 10 d    rho = +0.699
    ("H30-average", -1, 3),   # humidity, 30 d       rho = -0.407
    ("R5-average", -1, 4),    # rainfall, 5 d        rho = -0.351
    ("S30-average", +1, 5),   # sunshine, 30 d       rho = +0.314
]

PAPER_BETA = dict(r=0.393, K=27523156.818, k=94.315, d=696.650)   # Table 4
PAPER_BETA0 = dict(r=0.958, K=9648.885, k=1.576, d=9.706)         # their beta_0

THETA, DELTA = 1, 2

_MONTHS = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6}


def _day_index(month, day):
    """April 1st = day 1, the paper's time origin."""
    return (_dt.date(2001, month, day) - _dt.date(2001, 4, 1)).days + 1


def load_survey(path=SURVEY_XLSX):
    ws = openpyxl.load_workbook(path, data_only=True).active
    # year, date, then 6-wide blocks (S1, S2, S3, mean, SD) per taxon
    mean_col = {"aphid": 5, "lady": 10, "para": 15, "spid": 20, "hover": 25, "lace": 30}
    rows, year = [], None
    for r in ws.iter_rows(min_row=5, values_only=True):
        if isinstance(r[0], (int, float)):
            year = int(r[0])
        elif r[0] not in (None, ""):
            break                       # into the footnotes
        if r[1] in (None, ""):
            continue
        mon, day = str(r[1]).strip().split(".")
        month, day = _MONTHS[mon[:3]], int(day)
        rec = dict(year=year, month=month, day=day, date=str(r[1]).strip(),
                   t=_day_index(month, day))
        for name, col in mean_col.items():
            rec[name] = float(r[col])
        rows.append(rec)

    df = pd.DataFrame(rows)
    df["N"] = df.pop("aphid")
    df["P"] = sum(df[taxon] * pu for taxon, pu in PREDATOR_UNITS.items())
    df["m"] = [SPRAY_SURVIVAL if SPRAY_DATES.get(y) == (mo, d) else 1.0
               for y, mo, d in zip(df.year, df.month, df.day)]
    df = df.sort_values(["year", "t"]).reset_index(drop=True)
    df.insert(0, "no", np.arange(1, len(df) + 1))
    return df


def load_weather(path=WEATHER_XLSX):
    ws = openpyxl.load_workbook(path, data_only=True).active
    header = next(ws.iter_rows(min_row=2, max_row=2, values_only=True))
    rows = []
    for r in ws.iter_rows(min_row=3, values_only=True):
        if not isinstance(r[0], (int, float)):
            break
        rows.append(r)
    df = pd.DataFrame(rows, columns=[h for h in header])
    df = df.rename(columns={"Year": "year", "month": "month", "day": "day"})
    df["t"] = [_day_index(m, d) for m, d in zip(df.month, df.day)]
    return df


@dataclass
class CPScaler:
    """Constants for the catastrophe progression method, Section 2.4.1.

    Fitted on the April-May records of 1997-2001 and reused unchanged for 2002.
    Two things the paper doesn't spell out, both recovered by matching its
    published e values: the constants come from every daily record, not just the
    survey dates, and the combination rule is min, not mean.
    """
    lo: dict
    hi: dict
    mean: float
    std: float

    @classmethod
    def fit(cls, weather, years=VALIDATION_YEARS):
        base = weather[weather.year.isin(years)]
        lo = {c: float(base[c].min()) for c, _s, _p in CP_INDICES}
        hi = {c: float(base[c].max()) for c, _s, _p in CP_INDICES}
        raw = _cp_membership(base, lo, hi)
        return cls(lo, hi, float(raw.mean()), float(raw.std(ddof=1)))

    def transform(self, weather):
        return (_cp_membership(weather, self.lo, self.hi) - self.mean) / self.std


def _cp_membership(weather, lo, hi):
    """X_e, the comprehensive evaluation value. Eqs. 6-7 plus the membership set."""
    memberships = []
    for col, sign, power in CP_INDICES:
        z = weather[col].to_numpy(dtype=float)
        span = hi[col] - lo[col]
        zn = (z - lo[col]) / span if sign > 0 else (hi[col] - z) / span
        memberships.append(np.clip(zn, 0.0, 1.0) ** (1.0 / power))
    return np.min(np.vstack(memberships), axis=0)


def add_weather_index(survey, weather, scaler=None):
    scaler = scaler or CPScaler.fit(weather)
    w = weather.copy()
    w["e"] = scaler.transform(w)
    out = survey.merge(w[["year", "month", "day", "e"]],
                       on=["year", "month", "day"], how="left")
    assert out.e.notna().all(), "survey dates missing from the weather dataset"
    return out


def dNdt(N, e, P, m, r, K, k, d, theta=THETA, delta=DELTA):
    N = np.asarray(N, dtype=float)
    growth = r * m * N * (1.0 - e * N ** theta / K)
    predation = P * k * N ** delta / (N ** delta + d ** delta)
    return growth - predation


def equilibrium_poly(e, P, m, r, K, k, d):
    """Eq. 11, highest power first."""
    return np.array([1.0, -K / e, d ** 2 + k * P * K / (r * m * e), -K * d ** 2 / e])


def cusp_variables(e, P, m, r, K, k, d):
    """Control variables u, v and the discriminant, Eqs. 13 and 14.

    Delta uses the factored Eq. 14, not 27v^2 + 4u^3. With K ~ 2.75e7 both terms
    of the latter are around 1e41 while Delta is around 1e34, so up to 12 digits
    cancel and double precision has ~4 left. Against an 80-digit reference the
    naive form is good to 8e-8 (worst 8e-5); this one to 2e-16. The sign, and so
    the region, does survive the naive form on this dataset, but only just.
    """
    e = np.asarray(e, dtype=float)
    P = np.asarray(P, dtype=float)
    m = np.asarray(m, dtype=float)
    xi = K / e
    eta = k * P / (r * m)
    u = d ** 2 - xi ** 2 / 3.0 + eta * xi
    v = eta * xi ** 2 / 3.0 - 2.0 * xi ** 3 / 27.0 - 2.0 * d ** 2 * xi / 3.0
    delta = (xi ** 3 * eta ** 2 * (4 * eta - xi)
             + 4 * d ** 2 * (xi ** 4 - 5 * xi ** 3 * eta + 3 * xi ** 2 * eta ** 2
                             + 2 * xi ** 2 * d ** 2 + 3 * xi * eta * d ** 2 + d ** 4))
    return u, v, delta


def region(v, delta):
    """I: v >= 0, Delta > 0. II: Delta < 0. III: v < 0, Delta > 0."""
    v = np.atleast_1d(np.asarray(v, dtype=float))
    delta = np.atleast_1d(np.asarray(delta, dtype=float))
    return np.where(delta < 0, "II", np.where(v >= 0, "I", "III"))


def cusp_table(df, beta):
    u, v, delta = cusp_variables(df.e.to_numpy(), df.P.to_numpy(), df.m.to_numpy(),
                                 **beta)
    out = df.copy()
    out["u"], out["v"], out["delta"] = u, v, delta
    out["region"] = region(v, delta)
    return out


def equilibria(e, P, m, beta):
    """Positive roots of Eq. 11, ascending. N = 0 is always an equilibrium too."""
    roots = np.roots(equilibrium_poly(e, P, m, **beta))
    real = np.sort(roots[np.abs(roots.imag) < 1e-6 * (1 + np.abs(roots.real))].real)
    return real[real > 0]


def _averaged_knots(x, k=3):
    """MATLAB's aveknt, so spapi(k+1, ...) and make_interp_spline agree."""
    n = len(x)
    interior = [x[i + 1:i + k + 1].mean() for i in range(n - k - 1)]
    return np.r_[[x[0]] * (k + 1), interior, [x[-1]] * (k + 1)]


def bspline_derivative(t, y, k=3):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    spl: BSpline = make_interp_spline(t, y, k=k, t=_averaged_knots(t, k))
    return spl, spl.derivative()


def stage_one(df, years=VALIDATION_YEARS):
    """Spline through each season's counts, differentiated at the survey dates."""
    parts = []
    for year, grp in df[df.year.isin(years)].groupby("year"):
        grp = grp.sort_values("t").copy()
        _spl, der = bspline_derivative(grp.t.to_numpy(), grp.N.to_numpy())
        grp["dN_spline"] = der(grp.t.to_numpy())
        parts.append(grp)
    return pd.concat(parts).sort_values("no").reset_index(drop=True)


def stage_two(df, beta0=None):
    """Least squares for r, K, k, d against the stage-one derivatives."""
    beta0 = beta0 or PAPER_BETA0
    X = df[["N", "e", "P", "m"]].to_numpy(dtype=float)
    y = df["dN_spline"].to_numpy(dtype=float)

    def model(X, r, K, k, d):
        N, e, P, m = X.T
        return dNdt(N, e, P, m, r, K, k, d)

    p0 = [beta0["r"], beta0["K"], beta0["k"], beta0["d"]]
    popt, pcov = curve_fit(model, X, y, p0=p0, maxfev=200_000)
    beta = dict(zip(("r", "K", "k", "d"), popt))
    resid = y - model(X, *popt)
    r2 = 1.0 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))
    se = np.sqrt(np.diag(pcov))
    return beta, dict(zip(("r", "K", "k", "d"), se)), float(r2)


def r_squared(obs, pred):
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    resid = obs - pred
    return float(1.0 - resid @ resid / ((obs - obs.mean()) @ (obs - obs.mean())))


def forcing(df_year, weather, scaler, year):
    """e(t), P(t), m(t) for one season.

    e comes from the daily weather record rather than the survey dates, since
    the CP index is defined every day. P is interpolated between surveys.
    """
    w = weather[weather.year == year].sort_values("t")
    wt, we = w.t.to_numpy(float), scaler.transform(w)

    st = df_year.t.to_numpy(float)
    sP = df_year.P.to_numpy(float)
    spray = SPRAY_DATES.get(year)
    t_spray = _day_index(*spray) if spray else None

    def e_of(t):
        return float(np.interp(t, wt, we))

    def P_of(t):
        return float(np.interp(t, st, sP))

    def m_of(t):
        if t_spray is None:
            return 1.0
        return SPRAY_SURVIVAL if t_spray <= t < t_spray + 1.0 else 1.0

    return e_of, P_of, m_of


def simulate_season(df_year, weather, scaler, beta, year, N0=None, t_span=None,
                    n_out=400, n_cap=1e6):
    """Integrate Eq. 5 over one season. Returns (t, N, blew_up).

    The cap is not paranoia: for e < 0 the correction term exceeds 1, growth
    goes quadratic and the thing reaches infinity in finite time.
    """
    e_of, P_of, m_of = forcing(df_year, weather, scaler, year)
    t0 = float(df_year.t.min()) if t_span is None else t_span[0]
    t1 = float(df_year.t.max()) if t_span is None else t_span[1]
    y0 = float(df_year.N.iloc[0]) if N0 is None else N0
    y0 = max(y0, 1.0)               # N = 0 is absorbing

    def cap(t, y):
        return y[0] - n_cap
    cap.terminal, cap.direction = True, 1

    sol = solve_ivp(lambda t, y: [dNdt(max(y[0], 0.0), e_of(t), P_of(t), m_of(t),
                                       **beta)],
                    (t0, t1), [y0], t_eval=np.linspace(t0, t1, n_out),
                    events=cap, method="Radau", rtol=1e-8, atol=1e-5)
    return sol.t, np.clip(sol.y[0], 0.0, None), bool(sol.t_events[0].size)


def simulate_stepwise(df_year, weather, scaler, beta, year):
    """One interval ahead, restarting from each observation."""
    e_of, P_of, m_of = forcing(df_year, weather, scaler, year)
    t = df_year.t.to_numpy(float)
    N = df_year.N.to_numpy(float)
    pred = [N[0]]
    for i in range(len(t) - 1):
        def rhs(_t, y):
            return dNdt(max(y[0], 0.0), e_of(_t), P_of(_t), m_of(_t), **beta)

        def cap(_t, y):
            return y[0] - 1e6
        cap.terminal, cap.direction = True, 1
        sol = solve_ivp(lambda _t, y: [rhs(_t, y)], (t[i], t[i + 1]),
                        [max(N[i], 1.0)], events=cap, method="Radau",
                        rtol=1e-8, atol=1e-5)
        pred.append(max(sol.y[0, -1], 0.0))
    return np.array(pred)


def bifurcation_curve(u_min, n=800):
    """27v^2 + 4u^3 = 0 over u <= 0."""
    u = -np.logspace(np.log10(max(-u_min, 1e-12)), -12, n)
    v = np.sqrt(-4.0 * u ** 3 / 27.0)
    return u, v


def catastrophe_coordinate(u, v, delta):
    """w = v / (2(-u/3)^1.5), which locates a point regardless of scale.

    |w| < 1 is region II, w >= 1 region I, w <= -1 region III. Computed from
    Delta via w^2 = 1 + Delta/(4(-u)^3); forming it from u and v directly loses
    everything to cancellation, same problem as in cusp_variables.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    delta = np.asarray(delta, dtype=float)
    w2 = 1.0 + delta / (4.0 * (-u) ** 3)
    return np.sign(v) * np.sqrt(np.clip(w2, 0.0, None))
