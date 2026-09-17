"""Reproduce Li et al. (2020) end to end and build every figure.

    python3 src/run_analysis.py [--refit-sim]

Writes tables to out/ and figures to figures/.  --refit-sim re-runs the global
search for the GPD-S parameters (a few minutes) instead of using the stored fit.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fold_model as fm
import gpd
import sim_variant as sv

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
OUT = ROOT / "out"
DATA = ROOT / "data"
for p in (FIG, OUT):
    p.mkdir(exist_ok=True)

YEARS = (1997, 1998, 1999, 2000, 2001, 2002)
YCOL = dict(zip(YEARS, ["#4c3f91", "#1f6feb", "#12a4a4", "#4aa02c",
                        "#e8a33d", "#c0392b"]))
RCOL = {"I": "#2b8a3e", "II": "#e8a33d", "III": "#c0392b"}
RNAME = {"I": "I  low-density branch only",
         "II": "II  bistable (catastrophic region)",
         "III": "III  outbreak branch only"}

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "legend.frameon": False, "figure.facecolor": "white",
})


def date_label(row) -> str:
    return f"{row.date}"


def fig_field_data(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.6), sharex=True)
    for ax, year in zip(axes.ravel(), YEARS):
        g = df[df.year == year].sort_values("t")
        for reg, col in RCOL.items():
            mask = (g.region == reg).to_numpy()
            if mask.any():
                ax.scatter(g.t[mask], np.maximum(g.N[mask], 1), s=46, zorder=4,
                           color=col, edgecolor="k", linewidth=0.5, label=RNAME[reg])
        ax.plot(g.t, np.maximum(g.N, 1), "-", color="0.35", lw=1.2, zorder=3)
        ax.set_yscale("log")
        ax.set_ylim(0.7, 3.5e4)
        ax.set_title(f"{year}" + ("   (test year)" if year == gpd.TEST_YEAR else ""))

        ax2 = ax.twinx()
        ax2.plot(g.t, g.P, color="#1f6feb", lw=1.0, ls="--", marker="^", ms=3.5,
                 zorder=2)
        ax2.axhline(2 * gpd.PAPER_BETA["d"] * gpd.PAPER_BETA["r"] / gpd.PAPER_BETA["k"],
                    color="#1f6feb", lw=0.8, ls=":", alpha=0.7)
        ax2.set_ylim(0, 42)
        ax2.grid(False)
        ax2.tick_params(colors="#1f6feb", labelsize=7)
        if year in (1999, 2002):
            ax2.set_ylabel("natural enemies (PU / 100 stems)", color="#1f6feb",
                           fontsize=8)

        if year in gpd.SPRAY_DATES:
            ts = gpd._day_index(*gpd.SPRAY_DATES[year])
            ax.axvline(ts, color="#6f42c1", lw=1.1, ls="-.")
            ax.annotate("spray", (ts, 6e3), color="#6f42c1", fontsize=7.5,
                        ha="center", va="bottom",
                        bbox=dict(fc="white", ec="none", pad=0.6))
        if year in (2000, 2001, 2002):
            ax.set_xlabel("day (1 = 1 April)")
        if year in (1997, 2000):
            ax.set_ylabel("aphids / 100 stems")

    handles = [Line2D([], [], marker="o", ls="", color=RCOL[r], mec="k",
                      label=RNAME[r]) for r in ("I", "II", "III")]
    handles += [Line2D([], [], color="#1f6feb", ls="--", marker="^", ms=4,
                       label="natural enemies P (right axis)"),
                Line2D([], [], color="#1f6feb", ls=":",
                       label=r"bistability threshold $P^*=2dr/k$"),
                Line2D([], [], color="#6f42c1", ls="-.", label="insecticide")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Wheat aphid field surveys 1997-2002 (Dataset S1), points coloured "
                 "by cusp control region", y=0.98)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(FIG / "01_field_data.png", bbox_inches="tight")
    plt.close(fig)


def fig_cp_index(weather: pd.DataFrame, scaler: gpd.CPScaler, df: pd.DataFrame):
    fig = plt.figure(figsize=(13, 7.2))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.25], hspace=0.45,
                          wspace=0.3)

    # (a) the four fuzzy membership functions
    ax = fig.add_subplot(gs[0, 0])
    z = np.linspace(0, 1, 300)
    for (col, sign, power), c in zip(gpd.CP_INDICES, plt.cm.plasma([.1, .35, .6, .82])):
        ax.plot(z, z ** (1 / power), color=c, lw=1.6,
                label=fr"$x={{z}}^{{1/{power}}}$  {col.split('-')[0]}"
                      f" ({'+' if sign > 0 else '−'})")
    ax.plot(z, z, color="0.7", lw=0.8, ls=":")
    ax.set(xlabel="normalised index $z$", ylabel="membership $x$",
           title="(a) butterfly catastrophe\nmembership functions")
    ax.legend(fontsize=7)

    # (b) the normalised indices themselves, all years
    ax = fig.add_subplot(gs[0, 1:])
    w = weather[weather.year.isin(gpd.VALIDATION_YEARS)]
    for (col, sign, _p), c in zip(gpd.CP_INDICES, plt.cm.plasma([.1, .35, .6, .82])):
        span = scaler.hi[col] - scaler.lo[col]
        zz = ((w[col] - scaler.lo[col]) / span if sign > 0
              else (scaler.hi[col] - w[col]) / span)
        ax.plot(np.arange(len(zz)), zz, color=c, lw=0.9, label=col)
    for i in range(1, 5):
        ax.axvline(61 * i, color="0.6", lw=0.8)
    ax.set_xticks(61 * np.arange(5) + 30)
    ax.set_xticklabels(gpd.VALIDATION_YEARS)
    ax.set(ylabel="normalised (1 = good for aphids)",
           title="(b) normalised meteorological indices, April-May")
    ax.legend(fontsize=7, ncol=4, loc="lower center")

    # (c) resulting e, daily, per year
    for i, year in enumerate(YEARS):
        ax = fig.add_subplot(gs[1 + i // 3, i % 3])
        wy = weather[weather.year == year].sort_values("t")
        e = scaler.transform(wy)
        ax.plot(wy.t, e, color=YCOL[year], lw=1.3)
        ax.axhline(0, color="k", lw=0.7)
        ax.fill_between(wy.t, 0, e, where=e > 0, color="#c0392b", alpha=0.18)
        ax.fill_between(wy.t, 0, e, where=e <= 0, color="#2b8a3e", alpha=0.18)
        g = df[df.year == year]
        ax.scatter(g.t, g.e, s=26, color="k", zorder=5)
        ax.set_title(("(c) " if i == 0 else "") +
                     f"meteorological factor $e$, {year}", fontsize=9)
        ax.set_ylim(-4.2, 2.6)
        if i >= 3:
            ax.set_xlabel("day (1 = 1 April)")
        if i % 3 == 0:
            ax.set_ylabel("$e$")
    fig.suptitle("Catastrophe-progression estimate of the meteorological factor "
                 "(Section 2.4.1); black dots are survey dates", y=0.995)
    fig.savefig(FIG / "02_cp_index.png", bbox_inches="tight")
    plt.close(fig)


def fig_two_stage(s1: pd.DataFrame, beta_fit: dict, r2_fit: float):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    ax = axes[0]
    for year in gpd.VALIDATION_YEARS:
        g = s1[s1.year == year].sort_values("t")
        spl, _ = gpd.bspline_derivative(g.t.to_numpy(float), g.N.to_numpy(float))
        tt = np.linspace(g.t.min(), g.t.max(), 400)
        ax.plot(tt, spl(tt), color=YCOL[year], lw=1.4, label=str(year))
        ax.scatter(g.t, g.N, s=22, color=YCOL[year], edgecolor="k", linewidth=0.4,
                   zorder=4)
    ax.set(xlabel="day (1 = 1 April)", ylabel="aphids / 100 stems",
           title="(a) stage 1: cubic B-spline through the survey points")
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1]
    ax.axhline(0, color="k", lw=0.7)
    pred_paper = gpd.dNdt(s1.N, s1.e, s1.P, s1.m, **gpd.PAPER_BETA)
    pred_fit = gpd.dNdt(s1.N, s1.e, s1.P, s1.m, **beta_fit)
    ax.plot(s1.no, s1.dN_spline, "o-", ms=4, color="0.25", lw=1.0,
            label=r"spline $\hat{\dot y}(t)$")
    ax.plot(s1.no, pred_paper, "s--", ms=3.5, color="#c0392b", lw=1.0,
            label=r"model, published $\hat\beta$")
    ax.plot(s1.no, pred_fit, "^:", ms=3.5, color="#1f6feb", lw=1.0,
            label=r"model, our refit")
    ax.set(xlabel="survey number NO", ylabel=r"$dN/dt$  (aphids / 100 stems / day)",
           title="(b) stage 2: iterative least squares")
    ax.legend(fontsize=8)

    ax = axes[2]
    lim = np.array([min(s1.dN_spline.min(), pred_fit.min()) * 1.1,
                    max(s1.dN_spline.max(), pred_fit.max()) * 1.1])
    ax.plot(lim, lim, color="0.6", lw=0.8, ls="--")
    for year in gpd.VALIDATION_YEARS:
        m = (s1.year == year).to_numpy()
        ax.scatter(s1.dN_spline[m], pred_fit[m], s=34, color=YCOL[year],
                   edgecolor="k", linewidth=0.4, label=str(year))
    r2_paper = gpd.r_squared(s1.dN_spline, pred_paper)
    ax.set(xlabel=r"spline $\hat{\dot y}(t)$", ylabel="model $dN/dt$",
           title=f"(c) goodness of fit\n$R^2$ = {r2_fit:.4f} (refit), "
                 f"{r2_paper:.4f} (published $\\hat\\beta$), 0.6473 (paper)")
    ax.legend(fontsize=8)
    fig.suptitle("Two-stage estimation of the constant parameters "
                 "(Section 2.4.4 / Appendix A), 1997-2001", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "03_two_stage.png", bbox_inches="tight")
    plt.close(fig)


def fig_control_panel(df: pd.DataFrame):
    """Paper Fig. 5, on a log scale it can survive."""
    # dates the paper calls out as jumps or near-jumps
    HIGHLIGHT = {5, 6, 7, 8, 25, 26, 27, 33, 34, 48, 56, 61}
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))

    ax = axes[0]
    uu = np.abs(df.u.to_numpy())
    grid = np.logspace(np.log10(uu.min()) - 0.4, np.log10(uu.max()) + 0.4, 400)
    vb = 2 * (grid / 3) ** 1.5           # bifurcation set |v| = 2(-u/3)^{3/2}
    ax.fill_betweenx(np.log10(grid), -np.log10(vb), np.log10(vb),
                     color=RCOL["II"], alpha=0.13, zorder=0)
    ax.plot(np.log10(vb), np.log10(grid), color="k", lw=1.4)
    ax.plot(-np.log10(vb), np.log10(grid), color="k", lw=1.4)
    for reg in ("I", "II", "III"):
        g = df[df.region == reg]
        ax.scatter(np.sign(g.v) * np.log10(np.abs(g.v)), np.log10(np.abs(g.u)),
                   s=44, color=RCOL[reg], edgecolor="k", linewidth=0.4,
                   label=RNAME[reg], zorder=4)
    for i, (_, row) in enumerate(df[df.no.isin(HIGHLIGHT)].iterrows()):
        left = row.v < 0
        ax.annotate(f"{int(row.no)} ({row.N:.0f})",
                    (np.sign(row.v) * np.log10(abs(row.v)), np.log10(abs(row.u))),
                    fontsize=6.5, ha="left" if left else "right",
                    xytext=(6 if left else -6, 9 if i % 2 else -11),
                    textcoords="offset points", color="0.15",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="0.55"))
    ax.set(xlabel=r"$\mathrm{sign}(v)\,\log_{10}|v|$", ylabel=r"$\log_{10}|u|$",
           title="(a) control panel, all 62 survey dates\n"
                 "labelled: NO (aphids / 100 stems) at the paper's jump points")
    ax.legend(fontsize=8, loc="lower center")
    ax.annotate("every point lies on one of the two branches of the\n"
                "bifurcation set to within $10^{-6}$; see (b)",
                (0.03, 0.93), xycoords="axes fraction", fontsize=7.5, color="0.3",
                va="top")

    # (b) how close each point really is to the bifurcation set
    ax = axes[1]
    rho = (df.delta / (4 * (-df.u) ** 3)).to_numpy()   # = w^2 - 1, sign of Delta
    for reg in ("I", "II", "III"):
        g = (df.region == reg).to_numpy()
        ax.scatter(df.no[g], rho[g], s=44, color=RCOL[reg], edgecolor="k",
                   linewidth=0.4, label=RNAME[reg], zorder=4)
    ax.axhline(0, color="k", lw=1.3)
    ax.set_yscale("symlog", linthresh=1e-10)
    lim = np.abs(rho).max() * 3
    ax.set_ylim(-lim, lim)
    bounds = df.groupby("year").no.max()
    for b in bounds[:-1]:
        ax.axvline(b + 0.5, color="0.7", lw=0.8)
    for year, g in df.groupby("year"):
        ax.annotate(str(year), (g.no.mean(), ax.get_ylim()[1]), fontsize=7.5,
                    ha="center", va="top", color="0.35")
    ax.set(xlabel="survey number NO", ylabel=r"$\Delta\,/\,4(-u)^3$",
           title="(b) signed distance to the bifurcation set\n"
                 "below the line = region II (bistable)")
    ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("Cusp control variables from Eq. (13); reproduction of Fig. 5 "
                 "and Table A2", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "04_control_panel.png", bbox_inches="tight")
    plt.close(fig)


def fig_ep_plane(df: pd.DataFrame, beta: dict):
    """The control panel again, in variables a grower can read."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True)
    e_grid = np.linspace(-4.2, 2.8, 400)

    for ax, m in zip(axes, (1.0, 0.5)):
        Pstar = np.array([_p_star(e, m, beta) for e in e_grid])
        ax.fill_between(e_grid, Pstar, 42, color=RCOL["II"], alpha=0.22)
        ax.fill_between(e_grid, 0, Pstar, where=e_grid < 0, color=RCOL["I"],
                        alpha=0.18)
        ax.fill_between(e_grid, 0, Pstar, where=e_grid >= 0, color=RCOL["III"],
                        alpha=0.18)
        ax.plot(e_grid, Pstar, color="k", lw=1.5)
        ax.axvline(0, color="k", lw=1.5)
        ax.set_ylim(0, 42)
        ax.set_xlim(e_grid[0], e_grid[-1])
        ax.set_xlabel("meteorological factor $e$")
        ax.set_title(f"m = {m}" + ("  (no spray)" if m == 1 else "  (spray day)"))
        ax.annotate("I", (-3.4, 20), fontsize=22, color=RCOL["I"], alpha=0.5)
        ax.annotate("III", (1.6, 20), fontsize=22, color=RCOL["III"], alpha=0.5)
        ax.annotate("II", (-1.0, 33), fontsize=22, color=RCOL["II"], alpha=0.7)
        ax.annotate(fr"$P^*=2dr\,m/k={2*beta['d']*beta['r']*m/beta['k']:.2f}$",
                    (e_grid[0] + 0.15, _p_star(0, m, beta) + 0.8), fontsize=8)

        sel = df[df.m == m]
        for year in YEARS:
            g = sel[sel.year == year]
            if not len(g):
                continue
            ax.scatter(g.e, g.P, s=48, color=YCOL[year], edgecolor="k",
                       linewidth=0.5, label=str(year), zorder=5)
        if m == 1.0:
            for year in YEARS:  # season trajectory
                g = df[(df.year == year)].sort_values("t")
                ax.plot(g.e, g.P, color=YCOL[year], lw=0.8, alpha=0.55, zorder=3)
    axes[0].set_ylabel("natural enemies $P$ (PU / 100 stems)")
    axes[0].annotate("thin lines follow each season in order of date;\n"
                     "spray dates are plotted in the right-hand panel",
                     (0.015, 0.965), xycoords="axes fraction", fontsize=7,
                     color="0.3", va="top")
    axes[0].legend(fontsize=8, ncol=2, loc="upper right")
    fig.suptitle("The cusp control panel in natural variables: for the published "
                 "parameters the bifurcation set is $P = 2drm/k$, independent of $e$",
                 y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "05_ep_plane.png", bbox_inches="tight")
    plt.close(fig)


def _p_star(e, m, beta, hi=80.0):
    from scipy.optimize import brentq
    if abs(e) < 1e-9:
        e = 1e-9

    def f(P):
        return gpd.cusp_variables(e, P, m, **beta)[2]
    lo = 1e-9
    if f(lo) * f(hi) > 0:
        return np.nan
    return brentq(f, lo, hi, xtol=1e-12)


def fig_cusp_surface(df: pd.DataFrame, beta: dict, beta_s: dict):
    fig = plt.figure(figsize=(13.5, 5.6))

    # (a) the canonical cusp manifold with the survey path on it
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    uu = np.linspace(-1.6, 0.6, 140)
    vv = np.linspace(-1.2, 1.2, 140)
    U, V = np.meshgrid(uu, vv)
    X = np.empty_like(U)
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            rts = np.roots([1, 0, U[i, j], V[i, j]])
            rts = rts[np.abs(rts.imag) < 1e-9].real
            X[i, j] = rts.max() if V[i, j] < 0 else rts.min()
    ax.plot_surface(U, V, X, cmap="coolwarm", alpha=0.75, linewidth=0,
                    rstride=2, cstride=2, antialiased=True)
    ub = np.linspace(-1.6, 0, 200)
    ax.plot(ub, 2 * (-ub / 3) ** 1.5, np.zeros_like(ub) - 1.6, color="k", lw=1.2)
    ax.plot(ub, -2 * (-ub / 3) ** 1.5, np.zeros_like(ub) - 1.6, color="k", lw=1.2)
    ax.set(xlabel="$u$", ylabel="$v$", zlabel="$x$",
           title="(a) cusp manifold $x^3+ux+v=0$\nwith the bifurcation set beneath")
    ax.view_init(24, -58)

    # (b) equilibria of the published model as P varies
    ax = fig.add_subplot(1, 3, 2)
    for e_val, c in zip((0.5, 1.5), ("#c0392b", "#1f6feb")):
        Ps = np.linspace(0.1, 25, 500)
        for P in Ps:
            roots = gpd.equilibria(e_val, P, 1.0, beta)
            if len(roots) == 0:
                continue
            stab = _stable_gpd(roots, e_val, P, 1.0, beta)
            ax.scatter([P] * len(roots), roots, s=1.6,
                       c=[c if s else "0.65" for s in stab])
        ax.plot([], [], color=c, lw=2, label=f"$e$ = {e_val}")
    ax.axvline(2 * beta["d"] * beta["r"] / beta["k"], color="k", ls="--", lw=1)
    ax.set_yscale("log")
    ax.set_ylim(1, 1e9)
    ax.set(xlabel="natural enemies $P$ (PU / 100 stems)",
           ylabel="equilibrium aphids / 100 stems",
           title="(b) published parameters\ngrey = unstable")
    ax.annotate("upper branch $\\approx K/e \\approx 10^7$:\nnot a field density",
                (0.30, 0.86), xycoords="axes fraction", fontsize=7.5, color="0.25")
    ax.legend(fontsize=8, loc="lower left")

    # (c) same picture for GPD-S
    ax = fig.add_subplot(1, 3, 3)
    for e_val, c in zip((0.5, 1.5), ("#c0392b", "#1f6feb")):
        for P in np.linspace(0.1, 25, 400):
            roots = sv.equilibria(e_val, P, 1.0, beta_s)
            if len(roots) == 0:
                continue
            stab = sv.stability(roots, e_val, P, 1.0, beta_s)
            ax.scatter([P] * len(roots), roots, s=1.6,
                       c=[c if s else "0.65" for s in stab])
        ax.plot([], [], color=c, lw=2, label=f"$e$ = {e_val}")
    ax.set_yscale("log")
    ax.set_ylim(1, 3e4)
    ax.set(xlabel="natural enemies $P$ (PU / 100 stems)",
           ylabel="equilibrium aphids / 100 stems",
           title="(c) GPD-S\nrefuge / outbreak branches")
    ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("Cusp geometry: canonical manifold, and the equilibrium branches "
                 "it corresponds to", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "06_cusp_surface.png", bbox_inches="tight")
    plt.close(fig)


def _stable_gpd(roots, e, P, m, beta, h=1e-4):
    return [gpd.dNdt(x * (1 + h), e, P, m, **beta)
            < gpd.dNdt(x * (1 - h), e, P, m, **beta) for x in roots]


def fig_hysteresis(beta_s: dict):
    """Delay convention: sweep a control up, then back down."""
    from scipy.integrate import solve_ivp
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    for ax, (sweep, fixed, label) in zip(axes, [
            ("e", dict(P=9.0, m=1.0), "meteorological factor $e$"),
            ("P", dict(e=1.2, m=1.0), "natural enemies $P$ (PU / 100 stems)")]):
        lo, hi = (-1.5, 2.5) if sweep == "e" else (24.0, 2.0)
        T = 4000.0  # slow enough that the state tracks its attractor

        def run(a, b, N0):
            def ctrl(t):
                return a + (b - a) * t / T

            def rhs(t, y):
                kw = dict(fixed)
                kw[sweep] = ctrl(t)
                return [sv.rhs(y[0], e=kw.get("e"), P=kw.get("P"), m=kw["m"],
                               **beta_s)]
            sol = solve_ivp(rhs, (0, T), [N0], t_eval=np.linspace(0, T, 3000),
                            method="Radau", rtol=1e-9, atol=1e-7)
            return np.array([ctrl(t) for t in sol.t]), sol.y[0]

        cf, Nf = run(lo, hi, 5.0)
        cb, Nb = run(hi, lo, Nf[-1])
        ax.plot(cf, Nf, color="#c0392b", lw=2.0, label="forward sweep")
        ax.plot(cb, Nb, color="#1f6feb", lw=2.0, ls="--", label="reverse sweep")

        # equilibrium branches underneath
        grid = np.linspace(min(lo, hi), max(lo, hi), 300)
        for val in grid:
            kw = dict(fixed)
            kw[sweep] = val
            roots = sv.equilibria(kw.get("e"), kw.get("P"), kw["m"], beta_s)
            if not len(roots):
                continue
            stab = sv.stability(roots, kw.get("e"), kw.get("P"), kw["m"], beta_s)
            ax.scatter([val] * len(roots), roots, s=1.2,
                       c=["0.35" if s else "0.75" for s in stab], zorder=1)
        ax.set_yscale("log")
        ax.set(xlabel=label, ylabel="aphids / 100 stems",
               title=f"sweeping {label.split('$')[0].strip()}")
        ax.legend(fontsize=8, loc="upper left" if sweep == "e" else "upper right")

    axes[0].annotate("jump up:\noutbreak", (1.55, 300), fontsize=8, color="#c0392b")
    axes[1].annotate("the return path is a different\ncurve, hysteresis",
                     (0.05, 0.08), xycoords="axes fraction", fontsize=8, color="0.3")
    fig.suptitle("Sudden transition and hysteresis under the delay convention "
                 "(GPD-S, quasi-static sweeps); dots are the equilibrium branches",
                 y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "07_hysteresis.png", bbox_inches="tight")
    plt.close(fig)


def fig_simulation(df, weather, scaler, beta, beta_s):
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.0), sharex=True)
    rows = []
    for ax, year in zip(axes.ravel(), YEARS):
        g = df[df.year == year].sort_values("t")
        obs_t, obs_N = g.t.to_numpy(float), g.N.to_numpy(float)

        t_f, N_f, blew = gpd.simulate_season(g, weather, scaler, beta, year)
        step = gpd.simulate_stepwise(g, weather, scaler, beta, year)
        t_s, N_s = sv.simulate(g, weather, scaler, beta_s, year)
        pred_s = np.interp(obs_t, t_s, N_s)

        ax.plot(obs_t, np.maximum(obs_N, 1), "o-", color="k", ms=5, lw=1.3,
                label="observed", zorder=5)
        ax.plot(t_f, np.maximum(N_f, 1), color="#c0392b", lw=1.4,
                label="Eq. (5), free run")
        if blew:
            ax.scatter([t_f[-1]], [max(N_f[-1], 1)], marker="x", s=70,
                       color="#c0392b", zorder=6)
            ax.annotate("blow-up", (t_f[-1], max(N_f[-1], 1) * 1.6), fontsize=7,
                        color="#c0392b", ha="right")
        ax.plot(obs_t, np.maximum(step, 1), "s--", color="#e8a33d", ms=4, lw=1.1,
                label="Eq. (5), restart each interval")
        ax.plot(t_s, np.maximum(N_s, 1), color="#1f6feb", lw=1.8,
                label="GPD-S, free run")

        ax.set_yscale("log")
        ax.set_ylim(0.7, 3e6)
        r2_step = gpd.r_squared(np.log1p(obs_N), np.log1p(step))
        r2_s = gpd.r_squared(np.log1p(obs_N), np.log1p(pred_s))
        ax.set_title(f"{year}"
                     + ("  (test year)" if year == gpd.TEST_YEAR else "")
                     + f"\n$R^2_{{\\log}}$: restart {r2_step:.2f}, GPD-S {r2_s:.2f}",
                     fontsize=9)
        if year in gpd.SPRAY_DATES:
            ax.axvline(gpd._day_index(*gpd.SPRAY_DATES[year]), color="#6f42c1",
                       lw=1.0, ls="-.")
        if year in (2000, 2001, 2002):
            ax.set_xlabel("day (1 = 1 April)")
        if year in (1997, 2000):
            ax.set_ylabel("aphids / 100 stems")
        rows.append(dict(year=year, blew_up=blew, r2log_restart=r2_step,
                         r2log_gpds=r2_s, obs_peak=obs_N.max(),
                         gpds_peak=float(N_s.max())))

    axes[0, 0].legend(fontsize=7.5, loc="upper left")
    fig.suptitle("Season simulations. Eq. (5) runs away at $r\\approx0.39$/day with an "
                 "inert brake and a saturating predator; GPD-S is the repaired variant.",
                 y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "08_simulation.png", bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def fig_control_strategy(df, weather, scaler, beta_s):
    """Spray timing, spray efficacy and biological control, under GPD-S.

    The outcome measure is *aphid-days* (the integral of N over the season), not
    peak density: in several seasons the peak occurs before any plausible spray
    date, so peak density is insensitive to the thing being varied, while
    cumulative exposure, which is what drives yield loss, is not.
    """
    def outcome(year, spray, P_scale=1.0):
        g = df[df.year == year].sort_values("t")
        t, N = sv.simulate(g, weather, scaler, beta_s, year, spray=spray,
                           P_scale=P_scale)
        return N.max(), float(np.trapz(N, t))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))

    # (a) aphid-days vs spray date, at several efficacies, for the two bad years
    ax = axes[0]
    for year, ls in zip((2000, 2002), ("-", "--")):
        g = df[df.year == year]
        days = np.arange(int(g.t.min()) + 1, int(g.t.max()))
        base = outcome(year, ())[1]
        for surv, alpha in ((0.1, 1.0), (0.5, 0.55)):
            rel = [outcome(year, (float(d_), surv))[1] / base for d_ in days]
            ax.plot(days, rel, ls, color=YCOL[year], alpha=alpha, lw=1.7,
                    label=f"{year}, m = {surv}")
        if year in gpd.SPRAY_DATES:
            ax.axvline(gpd._day_index(*gpd.SPRAY_DATES[year]), color=YCOL[year],
                       lw=1.0, alpha=0.55)
    ax.axhline(1.0, color="0.5", ls=":", lw=1.2)
    ax.set(xlabel="spray date (day, 1 = 1 April)",
           ylabel="aphid-days, relative to no spray",
           title="(a) one spray: when does it pay?")
    ax.legend(fontsize=7.5, loc="lower left")
    ax.annotate("vertical lines = the dates actually used", (0.03, 0.96),
                xycoords="axes fraction", fontsize=7, color="0.35", va="top")

    # (b) the same surface over (date, efficacy) for the worst year
    ax = axes[1]
    year = 2002
    g = df[df.year == year].sort_values("t")
    days = np.arange(int(g.t.min()) + 1, int(g.t.max()))
    survs = np.linspace(0.05, 1.0, 20)
    base = outcome(year, ())[1]
    Z = np.array([[outcome(year, (float(d_), float(s_)))[1] / base for d_ in days]
                  for s_ in survs])
    im = ax.pcolormesh(days, survs, Z, cmap="magma_r", shading="auto")
    fig.colorbar(im, ax=ax).set_label("aphid-days, relative to no spray", fontsize=8)
    ax.grid(False)
    ax.set(xlabel="spray date", ylabel="survival rate $m$",
           title=f"(b) {year}: cumulative exposure vs\nspray date and efficacy")

    # (c) biological control alone
    ax = axes[2]
    scales = np.linspace(0.25, 4.0, 24)
    for year in YEARS:
        base = outcome(year, None)[1]
        rel = [outcome(year, None, P_scale=float(s_))[1] / base for s_ in scales]
        ax.plot(scales, rel, color=YCOL[year], lw=1.5, label=str(year))
    ax.axvline(1.0, color="k", lw=0.9, ls="--")
    ax.axhline(1.0, color="0.5", ls=":", lw=1.2)
    ax.set_yscale("log")
    ax.set(xlabel="natural enemies, multiple of observed",
           ylabel="aphid-days, relative to observed",
           title="(c) biological control alone")
    ax.legend(fontsize=8, ncol=2)
    fig.suptitle("Control strategies explored with GPD-S "
                 "(outcome = aphid-days, the driver of yield loss)", y=1.0)
    fig.tight_layout()
    fig.savefig(FIG / "09_control_strategy.png", bbox_inches="tight")
    plt.close(fig)


def fig_fold_model(df, weather, scaler, beta_f, beta_s):
    """The fold, plus all three models on one set of axes."""
    wx, sc, b = weather, scaler, beta_f
    fig=plt.figure(figsize=(13.5,8.6))
    gs=fig.add_gridspec(3,3,hspace=0.48,wspace=0.28,height_ratios=[1,1,1])

    # (a) the fold: equilibria vs rho, closed form
    ax=fig.add_subplot(gs[0,0])
    rr=np.linspace(2,14,400); disc=np.sqrt(rr**2-4)
    ax.plot(rr,b['d']*(rr-disc)/2,color='#2b8a3e',lw=2,label='refuge (stable)')
    ax.plot(rr,b['d']*(rr+disc)/2,color='#c0392b',lw=2,ls='--',label='threshold (unstable)')
    ax.axvline(2,color='k',lw=1.2)
    ax.annotate(r'fold at $\rho=2$',(2.15,b['d']*6),fontsize=8)
    ax.set(xlabel=r'$\rho = Pk/(r(e)d)$',ylabel='aphids / 100 stems',
           title='(a) the whole bifurcation\n'+r'$n_\pm=[\rho\pm\sqrt{\rho^2-4}]/2$')
    ax.legend(fontsize=8)

    # (b) rho(t) per season with the rho=2 line
    ax=fig.add_subplot(gs[0,1:])
    for y in YEARS:
        g=df[df.year==y].sort_values('t')
        r_=fm.rho(g.e.to_numpy(),g.P.to_numpy(),b)
        r_=np.where(np.isfinite(r_),r_,50.0)
        ax.plot(g.t,r_,'-o',ms=3.5,color=YCOL[y],lw=1.3,label=str(y))
    ax.axhline(2,color='k',lw=1.5)
    ax.axhspan(0,2,color='#c0392b',alpha=0.10)
    ax.set_yscale('log'); ax.set_ylim(0.05,60)
    ax.set(xlabel='day (1 = 1 April)',ylabel=r'$\rho$',
           title=r'(b) the single control variable through each season  ($\rho<2$ shaded: no refuge)')
    ax.legend(fontsize=8,ncol=6,loc='upper left')

    # (c) simulations, all three models
    for i,y in enumerate(YEARS):
        ax=fig.add_subplot(gs[1+i//3,i%3])
        g=df[df.year==y].sort_values('t')
        tf,Nf=fm.simulate(g,wx,sc,b,y)
        ts,Ns=sv.simulate(g,wx,sc,beta_s,y)
        tg,Ng,blew=gpd.simulate_season(g,wx,sc,gpd.PAPER_BETA,y)
        ax.plot(tg,np.maximum(Ng,1),color='#c0392b',lw=1.0,alpha=.75,label='GPD (paper)')
        ax.plot(ts,np.maximum(Ns,1),color='#e8a33d',lw=1.3,label='GPD-S')
        ax.plot(tf,np.maximum(Nf,1),color='#1f6feb',lw=2.0,label='fold model')
        ax.plot(g.t,np.maximum(g.N,1),'o',color='k',ms=5,zorder=5,label='observed')
        pr=np.interp(g.t.to_numpy(float),tf,Nf)
        ax.set_yscale('log'); ax.set_ylim(0.7,5e6)
        ax.set_title(f"{y}"+("  (held out)" if y==gpd.TEST_YEAR else "")
                     +f"   fold $R^2_{{\\log}}$={gpd.r_squared(np.log1p(g.N),np.log1p(pr)):.2f}",fontsize=9)
        if i>=3: ax.set_xlabel('day (1 = 1 April)')
        if i%3==0: ax.set_ylabel('aphids / 100 stems')
        if i==0: ax.legend(fontsize=7,loc='upper left')
    fig.suptitle('The fold model: one control variable, closed-form equilibria, four identifiable parameters',y=0.985)
    fig.savefig(FIG/'11_fold_model.png',bbox_inches='tight')
    plt.close(fig)


def fig_validation(df: pd.DataFrame, pub: pd.DataFrame, pubA2: pd.DataFrame):
    merged = df.merge(pub, on="no", suffixes=("", "_pub")).merge(pubA2, on="no")
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9))

    ax = axes[0]
    ax.plot([-4, 2.5], [-4, 2.5], color="0.6", ls="--", lw=0.8)
    ax.scatter(merged.e_pub, merged.e, s=26, color="#1f6feb", edgecolor="k",
               linewidth=0.3)
    ax.set(xlabel="published $e$ (Table 3)", ylabel="reproduced $e$",
           title=f"meteorological factor\nmax |diff| = "
                 f"{(merged.e - merged.e_pub).abs().max():.4f}")

    ax = axes[1]
    ax.plot([0, 40], [0, 40], color="0.6", ls="--", lw=0.8)
    ax.scatter(merged.P_pub, merged.P, s=26, color="#2b8a3e", edgecolor="k",
               linewidth=0.3)
    ax.set(xlabel="published $P$ (PU)", ylabel="reproduced $P$",
           title=f"predator units\nmax |diff| = "
                 f"{(merged.P - merged.P_pub).abs().max():.3f}")

    ax = axes[2]
    for col, pubcol, scale, lbl, c in (("u", "u_e12", 1e12, "$u$", "#c0392b"),
                                       ("v", "v_e18", 1e18, "$v$", "#1f6feb"),
                                       ("delta", "delta_e36", 1e36, r"$\Delta$",
                                        "#e8a33d")):
        rel = ((merged[col] / scale - merged[pubcol]).abs()
               / merged[pubcol].abs()).to_numpy()
        ax.scatter(merged.no, rel, s=20, color=c, label=lbl, edgecolor="none")
    ax.set_yscale("log")
    ax.set(xlabel="survey number NO", ylabel="relative difference",
           title="control variables vs Table A2")
    ax.legend(fontsize=8)

    ax = axes[3]
    labels = ["I", "II", "III"]
    conf = pd.crosstab(merged.region_pub, merged.region).reindex(
        index=labels, columns=labels, fill_value=0)
    ax.imshow(conf.to_numpy(), cmap="Blues", vmin=0)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, conf.to_numpy()[i, j], ha="center", va="center",
                    fontsize=12,
                    color="white" if conf.to_numpy()[i, j] > 20 else "k")
    ax.set_xticks(range(3), labels)
    ax.set_yticks(range(3), labels)
    ax.grid(False)
    agree = int((merged.region == merged.region_pub).sum())
    ax.set(xlabel="reproduced region", ylabel="published region",
           title=f"control-panel region\n{agree}/{len(merged)} identical")
    fig.suptitle("Validation of the reimplementation against Tables 3 and A2 "
                 "of the paper", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "10_validation.png", bbox_inches="tight")
    plt.close(fig)
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit-sim", action="store_true",
                    help="re-run the global search for the GPD-S parameters")
    args = ap.parse_args()

    print("loading supplementary datasets S1 and S2 ...")
    survey = gpd.load_survey()
    weather = gpd.load_weather()
    scaler = gpd.CPScaler.fit(weather)
    df = gpd.add_weather_index(survey, weather, scaler)
    df = gpd.cusp_table(df, gpd.PAPER_BETA)
    df.to_csv(OUT / "table3_reproduced.csv", index=False)

    pub = pd.read_csv(DATA / "table3_published.csv")
    pubA2 = pd.read_csv(DATA / "tableA2_published.csv")

    print("two-stage parameter estimation ...")
    s1 = gpd.stage_one(df)
    beta_fit, se_fit, r2_fit = gpd.stage_two(s1)
    est = pd.DataFrame({
        "parameter": ["r", "K", "k", "d"],
        "initial_beta0": [gpd.PAPER_BETA0[p] for p in ("r", "K", "k", "d")],
        "published": [gpd.PAPER_BETA[p] for p in ("r", "K", "k", "d")],
        "reproduced": [beta_fit[p] for p in ("r", "K", "k", "d")],
        "std_error": [se_fit[p] for p in ("r", "K", "k", "d")]})
    est.to_csv(OUT / "parameter_estimates.csv", index=False)
    print(est.to_string(index=False))
    print(f"  R^2 = {r2_fit:.4f}  (paper: 0.6473)")

    if args.refit_sim:
        print("fitting GPD-S by trajectory matching (slow) ...")
        beta_s, cost = sv.fit(df, weather, scaler)
        print("  ", {k: round(v, 4) for k, v in beta_s.items()}, f"cost={cost:.3f}")
    else:
        beta_s = sv.FITTED
    pd.DataFrame([beta_s]).to_csv(OUT / "gpds_parameters.csv", index=False)
    pd.DataFrame([fm.FITTED]).to_csv(OUT / "fold_parameters.csv", index=False)

    print("drawing figures ...")
    fig_field_data(df)
    fig_cp_index(weather, scaler, df)
    fig_two_stage(s1, beta_fit, r2_fit)
    fig_control_panel(df)
    fig_ep_plane(df, gpd.PAPER_BETA)
    fig_cusp_surface(df, gpd.PAPER_BETA, beta_s)
    fig_hysteresis(beta_s)
    sim_summary = fig_simulation(df, weather, scaler, gpd.PAPER_BETA, beta_s)
    sim_summary.to_csv(OUT / "simulation_summary.csv", index=False)
    fig_control_strategy(df, weather, scaler, beta_s)
    fig_fold_model(df, weather, scaler, fm.FITTED, beta_s)
    merged = fig_validation(df, pub, pubA2)
    merged.to_csv(OUT / "validation_merged.csv", index=False)

    print("\nvalidation against the published tables")
    print(f"  e             max |diff|  {(merged.e - merged.e_pub).abs().max():.5f}")
    print(f"  P (PU)        max |diff|  {(merged.P - merged.P_pub).abs().max():.4f}")
    for col, pubcol, scale in (("u", "u_e12", 1e12), ("v", "v_e18", 1e18),
                               ("delta", "delta_e36", 1e36)):
        rel = ((merged[col] / scale - merged[pubcol]).abs()
               / merged[pubcol].abs())
        print(f"  {col:<13} median rel  {rel.median():.2e}   max {rel.max():.2e}")
    print(f"  region        {(merged.region == merged.region_pub).sum()}/"
          f"{len(merged)} identical")
    print(f"\nfigures -> {FIG}\ntables  -> {OUT}")


if __name__ == "__main__":
    main()
