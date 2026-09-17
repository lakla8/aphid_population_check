# Wheat aphid population dynamics — catastrophe-theory model

A reimplementation, simulation and audit of

> Li Y., Hu Z., Li Z., Kong Y., Piyaratne M.K.D.K., Wang B., Zhao H. (2020).
> *Generalized population dynamics model of aphids in wheat based on catastrophe theory.*
> **BioSystems** 198:104217. https://doi.org/10.1016/j.biosystems.2020.104217

Everything is driven by the paper's two supplementary datasets: **S1** (`mmc1.xlsx`,
aphid and natural-enemy counts, 1997–2002) and **S2** (`mmc2.xlsx`, meteorological
rolling averages).

### Getting the data

The spreadsheets and the article PDF are Elsevier's, © 2020, all rights reserved,
so they are **not** in this repository. Download them yourself from the paper's
page at https://doi.org/10.1016/j.biosystems.2020.104217 and drop them in
`~/Downloads`, or edit `SURVEY_XLSX` / `WEATHER_XLSX` at the top of
[`src/gpd.py`](src/gpd.py) to point wherever you put them. The PDF is only needed
if you want to re-run `src/extract_tables.py`.

What *is* committed is `data/table3_published.csv` and `data/tableA2_published.csv`,
the numbers from the paper's Tables 3 and A2, scraped by `src/extract_tables.py`.
They are here because the validation cells need them and because factual tables
reproduced for verification are fair game; the scraper regenerates them from the
PDF if you would rather not take them on trust.

## Quick start

```bash
pip install numpy scipy pandas matplotlib openpyxl pymupdf sympy
```

```bash
jupyter lab aphid_simulation.ipynb
```

The notebook is the main entry point — it walks through the whole pipeline and has
an interactive simulator. To regenerate everything in batch instead:

```bash
python3 src/extract_tables.py    # rebuild reference tables from the article PDF
python3 src/run_analysis.py      # all 10 figures + summary tables
```

## The model

Eq. (5) of the paper, with the exponents the authors selected (θ = 1, δ = 2):

```
dN/dt = r·m·N·(1 − e·N/K) − P·k·N²/(N² + d²)
```

| symbol | meaning |
|---|---|
| `N` | aphids per 100 stems — the state variable |
| `e` | composite meteorological index (catastrophe-progression method) |
| `P` | natural enemies in standard predator units (PU) per 100 stems |
| `m` | survival rate under insecticide (1 = no spray) |
| `r, K, k, d` | intrinsic rate of increase, carrying capacity, predation rate, half-saturation |

Substituting `N = x + K/3e` removes the quadratic term, leaving the cusp
`x³ + ux + v = 0`. The sign of `Δ = 27v² + 4u³` divides the control panel into
region **I** (one low-density equilibrium), **II** (bistable — the catastrophic
region) and **III** (one high-density equilibrium).

## Layout

| path | contents |
|---|---|
| `aphid_simulation.ipynb` | the notebook — pipeline, simulator, statistical audit |
| `build_notebook.py` | generates the notebook (keeps cells reviewable in git) |
| `src/gpd.py` | the paper's model verbatim: loading, CP method, cusp algebra, two-stage estimation |
| `src/sim_variant.py` | **GPD-S** — the integrable variant, equilibria, calibration |
| `src/fold_model.py` | **fold model** — the one-parameter structure the data support |
| `src/extract_tables.py` | pulls Tables 3 and A2 out of the article PDF |
| `src/run_analysis.py` | batch driver → `figures/`, `out/` |
| `data/` | reference tables extracted from the paper |
| `figures/`, `out/` | generated output |

## Reproduction — it checks out

Validated against the paper's own Tables 3 and A2:

| quantity | agreement |
|---|---|
| meteorological factor `e`, all 62 dates | max diff **0.0013** |
| predator units `P` | max diff **0.053** |
| aphid counts `N`, `m`, day index | exact |
| `u`, `v`, `Δ` vs Table A2 | median rel. err **1.5×10⁻³** |
| control region | **62/62 identical** |
| two-stage refit | r 0.412 (paper 0.393), k 100.0 (94.3), d 721.9 (696.7), R² 0.6375 (0.6473) |

Three details the paper leaves implicit, recovered by matching its published values:

- the CP normalisation and standardisation constants come from **all daily
  April–May records of 1997–2001**, not just the survey dates;
- the combination rule is `X_e = min{x_a, x_b, x_c, x_d}` (non-complementary), not the mean;
- `Δ` must be evaluated through the factored Eq. (14) — see below.

## Findings

### 1. The control panel collapses to a threshold on predators alone

For the published parameters the bifurcation set is, to four decimals,

```
P* = 2·d·r·m / k
```

— **5.81 PU/100 stems** normally, **2.90** on a spray day — *independent of `e`*.
Classifying all 62 dates by "P > P*, else sign of e" reproduces the paper's regions
62/62. The enormous `u ~ 10¹²` and `v ~ 10¹⁸` values are almost entirely `K/e`
scaling that cancels out of `Δ`.

### 2. `Δ = 27v² + 4u³` should not be computed that way

Both terms are of order `10⁴¹` while `Δ` is of order `10³⁴`, so **up to 12.1
decimal digits cancel** and double precision is left with about 3.8 of its ~15.95.
Measured against an 80-digit evaluation from exact inputs:

| route | median rel. error | worst |
|---|---|---|
| naive `27v² + 4u³` | 8×10⁻⁸ | 8×10⁻⁵ |
| Eq. (14) factored | 2×10⁻¹⁶ | 3×10⁻¹⁵ |

On this dataset the naive form still gets the *sign* right everywhere, so the
region labels survive — but with under four significant digits of margin, that is
luck rather than a method. `gpd.cusp_variables` uses the factored form. (The two
expressions are algebraically identical; verified with `sympy`.)

### 3. Eq. (5) cannot be integrated as published

The paper fits it to *instantaneous rates* and never simulates it. Integrated, it
runs away, for three compounding reasons:

1. **The brake is inert.** At `N = 500`, `1 − eN/K` is 1.000018 for `e = −1` and
   0.999982 for `e = +1` — flipping `e` from +2 to −2 changes the growth rate by
   0.007%, because `K ≈ 2.75×10⁷` is five orders of magnitude above any observed count.
2. **The predator saturates** at `P·k`, so growth `r·N` overtakes it permanently
   once `N > P·k/r` — only ~1200 aphids at `P = 5` PU.
3. **What's left is `r ≈ 0.39`/day**, a doubling every 1.8 days.

The sign of `e` decides only how the divergence ends, not whether it happens
(a season blows up at `e = 0` too): for `e > 0` there is an equilibrium at
`N* ≈ K/e ≈ 10⁷`, finite but not a field density; for `e < 0` there is no positive
equilibrium at all and growth becomes quadratic, reaching infinity in finite time.

`K` is also simply not identifiable — R² is flat across five orders of magnitude of
it. The paper's own standard error, 4.2×10⁻¹⁰ on an estimate of 2.75×10⁷, is the
giveaway.

### 4. The regions do not predict outbreaks

Regions are meant to mean I = safe, III = outbreak. Taking the region at the
*start* of each of the 56 between-survey intervals and measuring the growth that
followed:

| region at start | n | mean growth/day | fraction that increased |
|---|---|---|---|
| I ("safe") | 26 | **+0.181** | 0.88 |
| II (bistable) | 15 | −0.108 | 0.60 |
| III ("outbreak") | 15 | **+0.106** | 0.73 |

Region I grew *faster* than region III; I vs III gives **p = 0.56**. The only
significant contrast is I vs II (p = 0.041), and region II is *defined* by
`P > 5.8 PU` — that's the predation term, not catastrophe geometry. Five of the six
largest actual jumps in the dataset started in region I.

Plain predictors do better: starting density (ρ = −0.53, p < 0.0001) and predator
load (ρ = −0.40, p = 0.003) are significant; the weather index `e` alone is not
(ρ = −0.18, p = 0.18).

The structural reason: **the region is a function of `(e, P, m)` only — `N` never
enters it.** It classifies conditions, not the state of the population, so it
cannot in principle report that a population jumped. The paper's headline 1997
transition, NO5 (335) → NO6 (591), is ×1.76 over six days — 9.9%/day, ordinary
aphid growth. The label changed because `e` crossed zero.

*Caveat: n = 56 intervals over 6 seasons rules out a large effect, not a subtle one.*

### 5. A numbering error in the paper

Section 3.3.1 states *"In 1999, sudden jumps occurred from NO 26 (5) to NO 27 (12)"*.
In the paper's own Table 3, NO26 = 146 and NO27 = 56; the values 5 and 12 belong to
**NO18 and NO19**. The accompanying narrative (early April, jointing stage, no
aphids yet) confirms NO18→19 was meant. The second citation in that sentence,
NO25 (97) → NO26 (146), is correct.

## GPD-S — the variant that can be simulated

`src/sim_variant.py` keeps everything carrying the paper's biology (the Holling
response with δ = 2, predator units, the insecticide, the CP index) and makes two
minimal changes:

```
dN/dt = r₀·exp(a·e)·m·N·(1 − N/K) − P·k·N²/(N² + d²)
```

- **`e` acts on the growth rate**, not the carrying capacity — the route the same
  group took in their Eq. (4) (Wu et al. 2014), where `r` is a function of the
  temperature factor. The exponential keeps `r > 0` for a standardised index.
- **density dependence is plain logistic**, so `K` is a real ceiling.

This is the Ludwig–Jones–Holling form, which keeps exactly the cusp geometry the
paper is about — refuge branch, unstable threshold, outbreak branch — at densities
that occur in a wheat field, and reproduces both catastrophe "flags" (sudden
transition and hysteresis) under quasi-static sweeps.

One further correction: the insecticide is applied as an **impulse** `N → m·N` at
the spray date. That is the paper's own definition of `m` (`m = N'_t0 / N_t0`,
Section 2.4.3); folding it into the growth rate for a single day, as the paper does
"for simplicity", removes a fraction of a percent instead of half.

### How well does GPD-S actually fit? Not very

| year | observed peak | simulated peak | R²(log) |
|---|---|---|---|
| 1997 | 591 | 866 | 0.75 |
| 1998 | 527 | 545 | 0.68 |
| 1999 | 562 | 1114 | −0.08 |
| 2000 | 5870 | 2854 | 0.70 |
| 2001 | 2958 | 665 | 0.13 |
| **2002** (held out) | 7480 | 667 | **−1.63** |

Pooled R²(log) = 0.61 on the fitted years, and it does **not** generalise to 2002.
Two non-identifiabilities survive the repair:

- **`K` sits on its lower bound** (the largest observed density). The optimiser
  keeps pushing it down — it would rather cap every season than reproduce the 5870
  and 7480 peaks.
- **`k` and `d` are identified only through `k/d²`** (= 3.13×10⁻⁴). Over most of the
  observed range `N ≪ d`, so the Holling-III term degenerates to `(P·k/d²)·N²`; the
  fit cost varies by 1.2% across an eight-fold range of `d`. This mirrors the `K`
  problem in the published model.

The substantive point: **the observed forcing does not contain enough signal** to
explain why 2000 and 2002 exploded while 1997 and 1998 stayed quiet. That is a
statement about the data and the covariates, not just this functional form — and it
is the same wall the original model runs into.

## The fold model — what the data actually support

Non-dimensionalising Eq. (5) with `n = N/d`, `τ = r·m·t`:

```
dn/dτ = n(1 − n/κ) − ρ·n²/(n²+1)        κ = K/(e·d)      ρ = P·k/(r·m·d)
```

Four parameters collapse to two groups, and the fit drives `κ → ∞`. What survives
is a **one-parameter** system with closed-form equilibria:

```
n± = [ρ ± √(ρ² − 4)] / 2          real iff  ρ ≥ 2
```

Verified against numerical roots to four significant figures. That's a **fold**
(saddle-node) in one control — the *simplest* object in the catastrophe hierarchy,
not the cusp the paper fits nor the swallowtail above it. And `ρ = 2` is exactly
the `P* = 2drm/k` threshold that keeps reappearing. This is Ludwig, Jones &
Holling (1978), where insect-outbreak bistability has lived for fifty years.

The paper ruled the fold out in its opening move — *"we do not consider the fold
and swallowtail catastrophe models because stable behavior cannot be observed in
these two"* — which is backwards for its own data.

### Implementation

```
dN/dt = r(e)·N·(1 − N/K) − P·k·N²/(N² + d²)        r(e) = r₀ + a·e
        r₀ = 0.2888   a = 0.1205   k = 156.4   d = 906.3   K = 10⁴ (fixed)
```

- **`r(e)` is linear so it can go negative** — in cold, wet weather the population
  shrinks unaided. Real mortality has to live somewhere; the paper itself invokes
  rainfall "washing" aphids off the crop in 2002. `r(e)` spans −0.149 to +0.518/day.
- **`K` is fixed, not fitted.** Pooled R²(log) moves only 0.54 → 0.56 across
  8×10³–10⁵. Fitting it is exactly what hollowed out the published model.
- **`k` is bounded to real appetites.** An earlier version put weather on predation
  as `k(e) = k₀e^{ce}`. It fitted *better* (0.645) and was nonsense — the optimiser
  bought low-`e` mortality by inflating predation to a median of 540 and a maximum
  of **45 000** aphids per predator unit per day, against a literature range of
  50–150 for an adult *C. septempunctata*. An unbounded exponential on a rate with
  a hard biological ceiling will always be abused that way.

### How it does

| year | observed peak | fold peak | R²(log) |
|---|---|---|---|
| 1997 | 591 | 1 466 | 0.59 |
| 1998 | 527 | 634 | 0.69 |
| 1999 | 562 | 712 | 0.24 |
| 2000 | 5 870 | 4 646 | 0.68 |
| 2001 | 2 958 | 580 | −0.31 |
| **2002** (held out) | 7 480 | 603 | **−1.65** |

Pooled 0.56 on the fitted years — **lower** than the indefensible exponential
variant (0.645) and than GPD-S (0.614). Right structure, honest fit; those are
different things.

**All three models fail on the same years.** That is the real finding: the failure
is in the data, not the functional form. `e` and `P` do not contain enough
information to explain why 2000 and 2002 exploded while 1997 and 1998 stayed quiet.
No re-specification of the right-hand side fixes a missing covariate — which is why
the next step is new *inputs* (immigration, crop growth stage, predator dynamics),
not a better curve.

## Verdict

| claim | holds? |
|---|---|
| tables and arithmetic reproduce | ✅ 62/62 regions |
| CP weather index well-constructed | ✅ |
| bistability requires `P > 2drm/k` | ✅ clean and checkable |
| Eq. (5) works as a population model | ❌ cannot be integrated |
| regions predict outbreaks | ❌ no signal (p = 0.56) |
| the identified transitions are catastrophes | ❌ gradual changes, relabelled |
| the cusp is the right normal form | ❌ the data support a **fold**, which the paper excluded a priori |

The paper is internally consistent and its catastrophe algebra is correct. But on
the six seasons it was built from, the classification carries no demonstrated
predictive power beyond "are there many predators", and the weather half of the
mechanism is the half that fails. The paper does not claim statistical validation —
it is framed as a descriptive framework — but its abstract does claim the model can
explain and forecast outbreak size and probability, and that is not demonstrated by
these data.
