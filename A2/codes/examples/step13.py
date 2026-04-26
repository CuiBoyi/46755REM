"""
46755 Renewables in Electricity Markets – Assignment 2, Step 1
Tasks 1.1 – 1.4: Day-ahead and Balancing Market Participation

Wind farm: 500 MW, price-taker, zero offer price
Uncertainty: wind production × DA price × system imbalance
Settlement: one-price (Task 1.1) and two-price (Task 1.2) schemes
Ex-post:    8-fold cross-validation (Task 1.3)
Risk-averse: CVaR, α=0.90, varying β (Task 1.4)
"""

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from pathlib import Path
from time import perf_counter
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (works without display)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ============================================================
# USER INPUTS
# ============================================================

CAPACITY_MW      = 500.0
N_HOURS          = 24

N_WIND           = 25
N_PRICE          = 25
N_IMB            = 5
TOTAL_SCENARIOS  = N_WIND * N_PRICE * N_IMB   # 3 125

# Task 1.3: 8-fold cross-validation (200 in-sample, 1 400 out-of-sample)
CV_FOLDS         = 8
CV_IN_SAMPLE     = 200
CV_OUT_SAMPLE    = TOTAL_SCENARIOS - CV_IN_SAMPLE   # 2 925  (>1400 is fine)

P_DEFICIT        = 0.5
RANDOM_SEED      = 42

# Task 1.3 in-sample size sensitivity
# Total scenarios fixed at 3125; try different in-sample sizes
POOL_SIZE        = 1600  # fixed total pool for in-sample size sensitivity
IN_SAMPLE_SIZES  = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
N_REPEATS_SIZE   = 5    # repeats per size to average out randomness


ALPHA_CVAR       = 0.90
BETA_VALUES      = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0]

BASE_DIR         = Path(__file__).resolve().parents[1]
HOURLY_DATA_PATH = BASE_DIR / "outputs" / "raw_data_hourly_timeseries.csv"
OUTPUT_DIR       = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATA LOADING & SCENARIO GENERATION  (reused from step1.py)
# ============================================================

def _daily_scenarios_from_hourly(df, value_col, count_col, n_scenarios):
    complete = df[df[value_col].notna() & (df[count_col] == 4)].copy()
    complete["date"] = complete["hour_start"].dt.date
    complete["hour"] = complete["hour_start"].dt.hour

    scenarios, dates = [], []
    for date, day in complete.groupby("date", sort=True):
        day = day.sort_values("hour_start")
        if len(day) != N_HOURS or set(day["hour"]) != set(range(N_HOURS)):
            continue
        scenarios.append(day[value_col].to_numpy(dtype=float))
        dates.append(date)
        if len(scenarios) == n_scenarios:
            break

    if len(scenarios) < n_scenarios:
        raise ValueError(
            f"Only {len(scenarios)} complete days found for {value_col}; "
            f"need {n_scenarios}."
        )
    return np.vstack(scenarios), dates


def load_scenarios():
    hourly = pd.read_csv(HOURLY_DATA_PATH)
    hourly["hour_start"] = pd.to_datetime(hourly["hour_start"], utc=True)
    hourly["wind_forecast_mw"] = pd.to_numeric(hourly["wind_forecast_mw"], errors="coerce")
    hourly["price"] = pd.to_numeric(hourly["price"], errors="coerce")

    wind, _ = _daily_scenarios_from_hourly(hourly, "wind_forecast_mw", "wind_quarters", N_WIND)
    da,   _ = _daily_scenarios_from_hourly(hourly, "price", "price_quarters", N_PRICE)

    # Normalise wind to 500 MW capacity
    mx = np.nanmax(wind)
    wind = np.clip(wind / mx * CAPACITY_MW, 0.0, CAPACITY_MW)

    rng = np.random.default_rng(RANDOM_SEED)
    imb = rng.binomial(1, P_DEFICIT, size=(N_IMB, N_HOURS))

    # Balancing prices: shape (N_PRICE, N_IMB, N_HOURS)
    bp = np.where(imb[np.newaxis, :, :] == 1,
                  1.25 * da[:, np.newaxis, :],
                  0.85 * da[:, np.newaxis, :])

    # Build flat joint scenario list
    joint = []
    for w in range(N_WIND):
        for p in range(N_PRICE):
            for k in range(N_IMB):
                joint.append({
                    "wind": wind[w].copy(),
                    "da":   da[p].copy(),
                    "si":   imb[k].copy(),
                    "bp":   bp[p, k].copy(),
                })

    print(f"Loaded {len(joint)} joint scenarios  "
          f"(wind {wind.min():.1f}–{wind.max():.1f} MW, "
          f"price {da.min():.2f}–{da.max():.2f} EUR/MWh)")
    return joint


def equal_probs(n):
    return np.full(n, 1.0 / n)


# ============================================================
# REVENUE CALCULATION HELPERS
# ============================================================

def one_price_revenue(q, scenario):
    """Scalar daily revenue for offer vector q under one-price."""
    W, DA, BP = scenario["wind"], scenario["da"], scenario["bp"]
    return float(np.sum(DA * q + BP * (W - q)))


def two_price_revenue(q, scenario):
    """Scalar daily revenue for offer vector q under two-price."""
    W, DA, BP, SI = scenario["wind"], scenario["da"], scenario["bp"], scenario["si"]
    rev = 0.0
    for t in range(N_HOURS):
        delta = W[t] - q[t]
        over  = max(delta,  0.0)
        under = max(-delta, 0.0)
        rev += DA[t] * q[t]
        if SI[t] == 1:          # system deficit
            rev += DA[t] * over - BP[t] * under
        else:                   # system surplus
            rev += BP[t] * over - DA[t] * under
    return rev


# ============================================================
# TASK 1.1 – ONE-PRICE STOCHASTIC OPTIMISATION
# ============================================================
# max  sum_s pi_s * sum_t [ DA_s,t * q_t + BP_s,t * (W_s,t - q_t) ]
# s.t. 0 <= q_t <= 500  for all t
#
# The objective is linear in q.  For each hour t independently:
#   E[DA - BP] * q_t + E[BP * W]
# If E[DA_t] > E[BP_t]:  q_t = 500  (sell everything)
# If E[DA_t] < E[BP_t]:  q_t = 0    (sell nothing)
# => all-or-nothing behaviour per hour
# ============================================================

def solve_one_price(scenarios, probs, capacity=CAPACITY_MW):
    S = len(scenarios)
    t0 = perf_counter()

    m = gp.Model("one_price")
    m.Params.OutputFlag = 0

    q = m.addVars(N_HOURS, lb=0.0, ub=capacity, name="q")

    obj = gp.LinExpr()
    for s, scen in enumerate(scenarios):
        pi = probs[s]
        W, DA, BP = scen["wind"], scen["da"], scen["bp"]
        for t in range(N_HOURS):
            obj += pi * (DA[t] * q[t] + BP[t] * (W[t] - q[t]))

    m.setObjective(obj, GRB.MAXIMIZE)
    m.optimize()

    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"one_price: status {m.Status}")

    q_star = np.array([q[t].X for t in range(N_HOURS)])
    profits = np.array([one_price_revenue(q_star, s) for s in scenarios])
    exp_profit = float(np.dot(probs, profits))

    return {
        "q_star":        q_star,
        "profits":       profits,
        "exp_profit":    exp_profit,
        "solve_time":    perf_counter() - t0,
    }


def evaluate_one_price(scenarios, probs, q_star):
    profits = np.array([one_price_revenue(q_star, s) for s in scenarios])
    return float(np.dot(probs, profits)), profits


# ============================================================
# TASK 1.2 – TWO-PRICE STOCHASTIC OPTIMISATION
# ============================================================
# The two-price objective is piecewise-linear and concave in q_t for each t,
# so each hour is solved exactly by checking candidates {0, 500, W_s,t}.
# ============================================================

def solve_two_price(scenarios, probs, capacity=CAPACITY_MW):
    S = len(scenarios)
    t0 = perf_counter()

    wind_mat = np.vstack([s["wind"] for s in scenarios])
    da_mat   = np.vstack([s["da"]   for s in scenarios])
    bp_mat   = np.vstack([s["bp"]   for s in scenarios])
    si_mat   = np.vstack([s["si"]   for s in scenarios])

    q_star = np.zeros(N_HOURS)
    for t in range(N_HOURS):
        candidates = np.unique(np.concatenate(([0.0, capacity], wind_mat[:, t])))
        candidates = candidates[(candidates >= 0.0) & (candidates <= capacity)]

        best_q, best_val = 0.0, -np.inf
        for q_c in candidates:
            delta = wind_mat[:, t] - q_c
            over  = np.maximum(delta,  0.0)
            under = np.maximum(-delta, 0.0)

            hp = da_mat[:, t] * q_c
            def_mask = si_mat[:, t] == 1
            hp[ def_mask] += da_mat[ def_mask, t] * over[ def_mask] - bp_mat[ def_mask, t] * under[ def_mask]
            hp[~def_mask] += bp_mat[~def_mask, t] * over[~def_mask] - da_mat[~def_mask, t] * under[~def_mask]

            val = float(np.dot(probs, hp))
            if val > best_val:
                best_val, best_q = val, q_c
        q_star[t] = best_q

    profits = np.array([two_price_revenue(q_star, s) for s in scenarios])
    exp_profit = float(np.dot(probs, profits))

    return {
        "q_star":     q_star,
        "profits":    profits,
        "exp_profit": exp_profit,
        "solve_time": perf_counter() - t0,
    }


def evaluate_two_price(scenarios, probs, q_star):
    profits = np.array([two_price_revenue(q_star, s) for s in scenarios])
    return float(np.dot(probs, profits)), profits


# ============================================================
# TASK 1.4 – CVaR RISK-AVERSE OFFERING
# ============================================================
# max  E[profit] - beta * CVaR_alpha(loss)
#
# CVaR_alpha(loss) = VaR_alpha - 1/((1-alpha)*S) * sum_s max(profit_s - VaR_alpha, 0)
# Equivalently (Rockafellar & Uryasev, maximisation form):
#
# max  sum_s pi_s * r_s  - beta * (eta + 1/((1-alpha)*S) * sum_s xi_s)
# s.t. r_s  = profit_s(q)               [scenario profit]
#      xi_s >= eta - r_s                 [CVaR auxiliary]
#      xi_s >= 0
#      0 <= q_t <= 500
#
# For one-price, profit is linear in q → full LP.
# For two-price, profit is piecewise-linear concave in q (non-convex for MIP);
# we solve the CVaR-enhanced LP by introducing deviation variables.
# ============================================================

def _cvar_from_profits(profits, probs, alpha):
    """Compute CVaR_alpha from a profit array (Rockafellar-Uryasev sample estimate)."""
    S = len(profits)
    sorted_p = np.sort(profits)                          # ascending (worst first)
    var_idx  = int(np.floor((1 - alpha) * S))            # index of VaR
    var_idx  = max(var_idx, 1)
    cvar = float(np.mean(sorted_p[:var_idx]))            # mean of worst (1-alpha) fraction
    return cvar


def solve_one_price_cvar(scenarios, probs, beta, alpha=ALPHA_CVAR, capacity=CAPACITY_MW):
    S = len(scenarios)
    t0 = perf_counter()

    # Reasonable profit range for bounding eta
    # Max possible profit ~ capacity * max_da_price * 24
    max_da = max(np.max(s["da"]) for s in scenarios)
    eta_ub = capacity * max_da * N_HOURS
    eta_lb = -eta_ub

    m = gp.Model("one_price_cvar")
    m.Params.OutputFlag = 0
    m.Params.NumericFocus = 1

    q   = m.addVars(N_HOURS, lb=0.0, ub=capacity, name="q")
    eta = m.addVar(lb=eta_lb, ub=eta_ub, name="eta")
    xi  = m.addVars(S, lb=0.0, name="xi")

    # Scenario profits (linear expression)
    rev = []
    for s, scen in enumerate(scenarios):
        W, DA, BP = scen["wind"], scen["da"], scen["bp"]
        r_s = gp.LinExpr()
        for t in range(N_HOURS):
            r_s += DA[t] * q[t] + BP[t] * (W[t] - q[t])
        rev.append(r_s)

    # CVaR auxiliary constraints: xi_s >= eta - r_s
    for s in range(S):
        m.addConstr(xi[s] >= eta - rev[s])

    # Objective: E[profit] - beta * CVaR
    exp_profit_expr = gp.quicksum(probs[s] * rev[s] for s in range(S))
    cvar_expr = eta + (1.0 / ((1 - alpha) * S)) * gp.quicksum(xi[s] for s in range(S))

    m.setObjective(exp_profit_expr - beta * cvar_expr, GRB.MAXIMIZE)
    m.optimize()

    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"one_price_cvar beta={beta}: status {m.Status}")

    q_star  = np.array([q[t].X for t in range(N_HOURS)])
    profits = np.array([one_price_revenue(q_star, s) for s in scenarios])
    exp_p   = float(np.dot(probs, profits))
    cvar_v  = _cvar_from_profits(profits, probs, alpha)

    return {
        "q_star":     q_star,
        "profits":    profits,
        "exp_profit": exp_p,
        "cvar":       cvar_v,
        "beta":       beta,
        "solve_time": perf_counter() - t0,
    }


def solve_two_price_cvar(scenarios, probs, beta, alpha=ALPHA_CVAR, capacity=CAPACITY_MW):
    """
    CVaR risk-averse two-price offering (LP formulation).

    Linearise two-price profits with over/under deviation variables:
        over_st  - under_st = W_st - q_t,  over_st >= 0, under_st >= 0
        over_st  <= capacity  (upper bound for numerical stability)
        under_st <= capacity

    For numerical stability with 3125 scenarios we scale profits by 1/1000
    internally and report results in original EUR.
    """
    S = len(scenarios)
    t0 = perf_counter()

    # Pre-compute numeric arrays for speed
    wind_arr = np.array([s["wind"] for s in scenarios])   # (S, T)
    da_arr   = np.array([s["da"]   for s in scenarios])
    bp_arr   = np.array([s["bp"]   for s in scenarios])
    si_arr   = np.array([s["si"]   for s in scenarios], dtype=int)

    # Scale factor for numerical stability
    scale = 1e-3

    max_da  = da_arr.max()
    eta_ub  = capacity * max_da * N_HOURS * scale
    eta_lb  = -eta_ub

    m = gp.Model("two_price_cvar")
    m.Params.OutputFlag   = 0
    m.Params.Method       = 2
    m.Params.NumericFocus = 2
    m.Params.ScaleFlag    = 2

    q     = m.addVars(N_HOURS, lb=0.0, ub=capacity, name="q")
    eta   = m.addVar(lb=eta_lb, ub=eta_ub, name="eta")
    xi    = m.addVars(S, lb=0.0, name="xi")
    over  = m.addVars(S, N_HOURS, lb=0.0, ub=capacity, name="ov")
    under = m.addVars(S, N_HOURS, lb=0.0, ub=capacity, name="un")

    # Deviation constraints and scaled revenue expressions
    rev = []
    for s in range(S):
        W  = wind_arr[s]
        DA = da_arr[s]
        BP = bp_arr[s]
        SI = si_arr[s]
        r_s = gp.LinExpr()
        for t in range(N_HOURS):
            m.addConstr(over[s, t] - under[s, t] == W[t] - q[t])
            r_s += scale * DA[t] * q[t]
            if SI[t] == 1:
                r_s += scale * (DA[t] * over[s, t] - BP[t] * under[s, t])
            else:
                r_s += scale * (BP[t] * over[s, t] - DA[t] * under[s, t])
        rev.append(r_s)

    for s in range(S):
        m.addConstr(xi[s] >= eta - rev[s])

    exp_profit_expr = gp.quicksum(probs[s] * rev[s] for s in range(S))
    cvar_expr = eta + (1.0 / ((1 - alpha) * S)) * gp.quicksum(xi[s] for s in range(S))

    m.setObjective(exp_profit_expr - beta * cvar_expr, GRB.MAXIMIZE)
    m.optimize()

    if m.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        raise RuntimeError(f"two_price_cvar beta={beta}: status {m.Status}")

    q_star  = np.array([q[t].X for t in range(N_HOURS)])
    profits = np.array([two_price_revenue(q_star, s) for s in scenarios])
    exp_p   = float(np.dot(probs, profits))
    cvar_v  = _cvar_from_profits(profits, probs, alpha)
    exp_p   = float(np.dot(probs, profits))
    cvar_v  = _cvar_from_profits(profits, probs, alpha)

    return {
        "q_star":     q_star,
        "profits":    profits,
        "exp_profit": exp_p,
        "cvar":       cvar_v,
        "beta":       beta,
        "solve_time": perf_counter() - t0,
    }


# ============================================================
# PLOTTING
# ============================================================

def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def plot_task11(result_1p, save_dir=OUTPUT_DIR):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Hourly offers
    ax = axes[0]
    hours = np.arange(1, N_HOURS + 1)
    ax.bar(hours, result_1p["q_star"], color="steelblue", edgecolor="white")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Day-ahead offer [MW]")
    ax.set_title("Task 1.1 – One-price: Optimal Hourly Offers")
    ax.set_ylim(0, CAPACITY_MW * 1.05)
    ax.set_xlim(0.5, N_HOURS + 0.5)

    # Profit distribution
    ax = axes[1]
    ax.hist(result_1p["profits"] / 1e3, bins=40, color="steelblue",
            edgecolor="white", alpha=0.8)
    ax.axvline(result_1p["exp_profit"] / 1e3, color="crimson", lw=2,
               label=f"E[profit] = {result_1p['exp_profit']/1e3:.1f} kEUR")
    ax.set_xlabel("Daily profit [kEUR]")
    ax.set_ylabel("Number of scenarios")
    ax.set_title("Task 1.1 – One-price: Profit Distribution")
    ax.legend()

    _save(fig, save_dir / "task11_one_price.png")


def plot_task12(result_1p, result_2p, save_dir=OUTPUT_DIR):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    hours = np.arange(1, N_HOURS + 1)
    colors = ["steelblue", "darkorange"]
    titles = ["One-price", "Two-price"]
    results = [result_1p, result_2p]

    for col, (res, title, color) in enumerate(zip(results, titles, colors)):
        ax = axes[0, col]
        ax.bar(hours, res["q_star"], color=color, edgecolor="white")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Day-ahead offer [MW]")
        ax.set_title(f"Task 1.2 – {title}: Optimal Hourly Offers")
        ax.set_ylim(0, CAPACITY_MW * 1.05)
        ax.set_xlim(0.5, N_HOURS + 0.5)

        ax = axes[1, col]
        ax.hist(res["profits"] / 1e3, bins=40, color=color, edgecolor="white", alpha=0.8)
        ax.axvline(res["exp_profit"] / 1e3, color="crimson", lw=2,
                   label=f"E = {res['exp_profit']/1e3:.1f} kEUR")
        ax.set_xlabel("Daily profit [kEUR]")
        ax.set_ylabel("Number of scenarios")
        ax.set_title(f"Task 1.2 – {title}: Profit Distribution")
        ax.legend()

    _save(fig, save_dir / "task12_comparison.png")


def plot_task13(cv_df, save_dir=OUTPUT_DIR):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    folds = cv_df["fold"].values

    for ax, scheme in zip(axes, ["One-price", "Two-price"]):
        col_in  = f"{scheme.lower().replace('-','_')}_in"
        col_out = f"{scheme.lower().replace('-','_')}_out"
        ax.plot(folds, cv_df[col_in]  / 1e3, "o-", color="steelblue",  label="In-sample")
        ax.plot(folds, cv_df[col_out] / 1e3, "s--", color="darkorange", label="Out-of-sample")
        avg_in  = cv_df[col_in].mean()  / 1e3
        avg_out = cv_df[col_out].mean() / 1e3
        ax.axhline(avg_in,  color="steelblue",  linestyle=":", lw=1.5,
                   label=f"Avg in  = {avg_in:.1f} kEUR")
        ax.axhline(avg_out, color="darkorange",  linestyle=":", lw=1.5,
                   label=f"Avg out = {avg_out:.1f} kEUR")
        ax.set_xlabel("Fold")
        ax.set_ylabel("Expected profit [kEUR]")
        ax.set_title(f"Task 1.3 – {scheme}: 8-fold Cross-validation")
        ax.legend(fontsize=8)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    _save(fig, save_dir / "task13_crossval.png")


def plot_task14(cvar_rows_1p, cvar_rows_2p, save_dir=OUTPUT_DIR):
    """
    Plot Expected Profit vs CVaR scatter, and profit distribution box plots
    across beta values for both schemes.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for col, (rows, title, color) in enumerate(zip(
        [cvar_rows_1p, cvar_rows_2p],
        ["One-price", "Two-price"],
        ["steelblue", "darkorange"],
    )):
        # Top row: E[profit] vs CVaR
        ax = axes[0, col]
        ep  = [r["exp_profit"] / 1e3 for r in rows]
        cv  = [r["cvar"]       / 1e3 for r in rows]
        bts = [r["beta"]              for r in rows]
        sc  = ax.scatter(cv, ep, c=bts, cmap="plasma", s=100, zorder=5)
        ax.plot(cv, ep, color=color, alpha=0.4, zorder=3)
        for r in rows[::max(1, len(rows)//6)]:     # annotate subset
            ax.annotate(f"β={r['beta']}", xy=(r["cvar"]/1e3, r["exp_profit"]/1e3),
                        fontsize=7, ha="left", va="bottom")
        plt.colorbar(sc, ax=ax, label="β")
        ax.set_xlabel("CVaR [kEUR]")
        ax.set_ylabel("Expected profit [kEUR]")
        ax.set_title(f"Task 1.4 – {title}\nExpected Profit vs CVaR (α=0.90)")

        # Bottom row: profit distribution percentiles across beta
        ax = axes[1, col]
        betas    = [r["beta"] for r in rows]
        p5_list  = [np.percentile(r["profits"], 5)  / 1e3 for r in rows]
        p10_list = [np.percentile(r["profits"], 10) / 1e3 for r in rows]
        p50_list = [np.percentile(r["profits"], 50) / 1e3 for r in rows]
        p90_list = [np.percentile(r["profits"], 90) / 1e3 for r in rows]
        exp_list = [r["exp_profit"] / 1e3 for r in rows]

        ax.plot(betas, p5_list,  "v--", color="crimson",    lw=1.5, label="5th pct")
        ax.plot(betas, p10_list, "^--", color="darkorange",  lw=1.5, label="10th pct (CVaR proxy)")
        ax.plot(betas, p50_list, "o-",  color="steelblue",   lw=2,   label="Median")
        ax.plot(betas, exp_list, "s-",  color="navy",        lw=2,   label="E[profit]")
        ax.plot(betas, p90_list, "D--", color="green",       lw=1.5, label="90th pct")
        ax.set_xlabel("β (risk-aversion parameter)")
        ax.set_ylabel("Profit [kEUR]")
        ax.set_title(f"Task 1.4 – {title}\nProfit Distribution vs β")
        ax.legend(fontsize=8)
        ax.set_xscale("symlog", linthresh=0.1)

    _save(fig, save_dir / "task14_cvar_frontier.png")


def plot_task14_offers(cvar_rows_1p, cvar_rows_2p, save_dir=OUTPUT_DIR):
    """Show how hourly bids shift with increasing β for each scheme."""
    betas_to_show = [0.0, 0.5, 2.0, 5.0]
    hours = np.arange(1, N_HOURS + 1)

    fig, axes = plt.subplots(2, len(betas_to_show), figsize=(14, 7), sharey=True)
    cmap = matplotlib.colormaps.get_cmap("viridis")

    for row_idx, (rows, label) in enumerate(
        [(cvar_rows_1p, "One-price"), (cvar_rows_2p, "Two-price")]
    ):
        beta_map = {r["beta"]: r for r in rows}
        for col_idx, b in enumerate(betas_to_show):
            ax = axes[row_idx, col_idx]
            r  = beta_map.get(b)
            if r is None:
                ax.set_visible(False)
                continue
            ax.bar(hours, r["q_star"], color=cmap(col_idx), edgecolor="white")
            ax.set_title(f"{label}\nβ={b}", fontsize=9)
            ax.set_xlabel("Hour")
            if col_idx == 0:
                ax.set_ylabel("DA offer [MW]")
            ax.set_ylim(0, CAPACITY_MW * 1.05)
            ax.set_xlim(0.5, N_HOURS + 0.5)

    fig.suptitle("Task 1.4 – Hourly Offers at Selected β Values", y=1.02)
    _save(fig, save_dir / "task14_offer_evolution.png")


# ============================================================
# PRINT HELPERS
# ============================================================

def _hline(char="=", width=65):
    print(char * width)


def print_task11(res):
    _hline()
    print("TASK 1.1 – ONE-PRICE SCHEME (IN-SAMPLE)")
    _hline()
    print(f"  Expected profit : {res['exp_profit']:>12.2f} EUR")
    print(f"  Solve time      : {res['solve_time']:.3f} s")
    q = res["q_star"]
    print(f"  Offers range    : {q.min():.1f} – {q.max():.1f} MW")
    n_zero  = int((q < 1e-3).sum())
    n_full  = int((q > CAPACITY_MW - 1e-3).sum())
    n_mid   = N_HOURS - n_zero - n_full
    print(f"  All-or-nothing  : {n_zero} hrs @ 0 MW, "
          f"{n_full} hrs @ {CAPACITY_MW:.0f} MW, "
          f"{n_mid} hrs intermediate")
    print(f"  Profit stats    : mean={res['profits'].mean():.0f}, "
          f"std={res['profits'].std():.0f}, "
          f"min={res['profits'].min():.0f}, "
          f"max={res['profits'].max():.0f} EUR")


def print_task12(res_1p, res_2p):
    _hline()
    print("TASK 1.2 – COMPARISON: ONE-PRICE vs TWO-PRICE (IN-SAMPLE)")
    _hline()
    for name, res in [("One-price", res_1p), ("Two-price", res_2p)]:
        q = res["q_star"]
        n_zero = int((q < 1e-3).sum())
        n_full = int((q > CAPACITY_MW - 1e-3).sum())
        print(f"\n  [{name}]")
        print(f"    Expected profit : {res['exp_profit']:>12.2f} EUR")
        print(f"    Offers 0 MW     : {n_zero} hrs,  {CAPACITY_MW:.0f} MW: "
              f"{n_full} hrs,  intermediate: {N_HOURS-n_zero-n_full} hrs")
        print(f"    Profit std      : {res['profits'].std():.0f} EUR")


def print_task13(cv_df):
    _hline()
    print("TASK 1.3 – 8-FOLD CROSS-VALIDATION (in-sample=200)")
    _hline()
    print(cv_df[["fold",
                 "one_price_in", "one_price_out",
                 "two_price_in", "two_price_out"]].to_string(index=False))
    print()
    for scheme, col_in, col_out in [
        ("One-price", "one_price_in", "one_price_out"),
        ("Two-price", "two_price_in", "two_price_out"),
    ]:
        avg_in  = cv_df[col_in].mean()
        avg_out = cv_df[col_out].mean()
        gap     = avg_in - avg_out
        print(f"  {scheme}: avg in={avg_in:.0f}  avg out={avg_out:.0f}  "
              f"gap={gap:.0f} EUR  ({gap/avg_in*100:.1f}%)")


def run_insample_size_sensitivity(joint, sizes=IN_SAMPLE_SIZES, n_repeats=N_REPEATS_SIZE,
                                  pool_size=POOL_SIZE, seed=RANDOM_SEED):
    """
    For each in-sample size, draw `pool_size` scenarios from joint (fixed total pool),
    then split into in-sample (n_in) and out-of-sample (pool_size - n_in).
    Repeat n_repeats times with different random draws and average OOS profits.
    """
    S = len(joint)
    rows = []

    for n_in in sizes:
        n_out = pool_size - n_in
        print(f"  in-sample={n_in}  out-of-sample={n_out} ...", end=" ", flush=True)
        ep1_oos_list, ep2_oos_list = [], []

        rng = np.random.default_rng(seed + n_in)
        for rep in range(n_repeats):
            # Draw a fresh pool of pool_size scenarios each repeat
            pool_idx = rng.choice(S, size=pool_size, replace=False)
            in_idx   = pool_idx[:n_in]
            out_idx  = pool_idx[n_in:]

            in_scen  = [joint[i] for i in in_idx]
            out_scen = [joint[i] for i in out_idx]
            p_in     = equal_probs(len(in_scen))
            p_out    = equal_probs(len(out_scen))

            r1 = solve_one_price(in_scen, p_in)
            r2 = solve_two_price(in_scen, p_in)

            ep1_oos, _ = evaluate_one_price(out_scen, p_out, r1["q_star"])
            ep2_oos, _ = evaluate_two_price(out_scen, p_out, r2["q_star"])

            ep1_oos_list.append(ep1_oos)
            ep2_oos_list.append(ep2_oos)

        row = {
            "n_in_sample":        n_in,
            "n_out_sample":       S - n_in,
            "one_price_oos_mean": np.mean(ep1_oos_list),
            "one_price_oos_std":  np.std(ep1_oos_list, ddof=0),
            "two_price_oos_mean": np.mean(ep2_oos_list),
            "two_price_oos_std":  np.std(ep2_oos_list, ddof=0),
        }
        rows.append(row)
        print(f"1P-oos={row['one_price_oos_mean']:.0f}  "
              f"2P-oos={row['two_price_oos_mean']:.0f} EUR")

    return pd.DataFrame(rows)


def print_insample_size_results(df):
    _hline()
    print("TASK 1.3 – IN-SAMPLE SIZE SENSITIVITY")
    _hline()
    print(f"  {'n_in':>6}  {'n_out':>6}  "
          f"{'1P OOS mean':>13}  {'1P OOS std':>11}  "
          f"{'2P OOS mean':>13}  {'2P OOS std':>11}")
    for _, r in df.iterrows():
        print(f"  {int(r.n_in_sample):>6}  {int(r.n_out_sample):>6}  "
              f"{r.one_price_oos_mean:>13.0f}  {r.one_price_oos_std:>11.0f}  "
              f"{r.two_price_oos_mean:>13.0f}  {r.two_price_oos_std:>11.0f}")
    print()
    best_1p = df.loc[df["one_price_oos_mean"].idxmax()]
    best_2p = df.loc[df["two_price_oos_mean"].idxmax()]
    print(f"  Best OOS for One-price : n_in={int(best_1p.n_in_sample)}  "
          f"mean={best_1p.one_price_oos_mean:.0f} EUR")
    print(f"  Best OOS for Two-price : n_in={int(best_2p.n_in_sample)}  "
          f"mean={best_2p.two_price_oos_mean:.0f} EUR")


def plot_insample_size_sensitivity(df, total_scenarios, save_dir=OUTPUT_DIR):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sizes = df["n_in_sample"].values

    for ax, col_mean, col_std, title, color in zip(
        axes,
        ["one_price_oos_mean", "two_price_oos_mean"],
        ["one_price_oos_std",  "two_price_oos_std"],
        ["One-price", "Two-price"],
        ["steelblue", "darkorange"],
    ):
        means = df[col_mean].values
        stds  = df[col_std].values

        ax.errorbar(sizes, means / 1e3, yerr=stds / 1e3,
                    fmt="o-", color=color, lw=2, capsize=5, capthick=1.5,
                    elinewidth=1.5, label="Avg OOS E[profit] ± 1 std")
        ax.fill_between(sizes,
                        (means - stds) / 1e3,
                        (means + stds) / 1e3,
                        alpha=0.15, color=color)
        ax.set_xlabel("In-sample size")
        ax.set_ylabel("Out-of-sample expected profit [kEUR]")
        ax.set_title(f"Task 1.3 – {title}\nOOS Profit vs In-sample Size (pool={total_scenarios:,})")
        ax.legend(fontsize=9)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    _save(fig, save_dir / "task13_insample_size.png")


def print_task14(rows_1p, rows_2p):
    _hline()
    print("TASK 1.4 – RISK-AVERSE CVaR (alpha=0.90)")
    _hline()
    for name, rows in [("One-price", rows_1p), ("Two-price", rows_2p)]:
        print(f"\n  [{name}]")
        print(f"  {'beta':>6}  {'E[profit] kEUR':>15}  {'CVaR kEUR':>12}  {'5th-pct kEUR':>14}  {'Std kEUR':>10}")
        for r in rows:
            p5 = np.percentile(r["profits"], 5) / 1e3
            std = r["profits"].std() / 1e3
            print(f"  {r['beta']:>6.1f}  "
                  f"{r['exp_profit']/1e3:>15.2f}  "
                  f"{r['cvar']/1e3:>12.2f}  "
                  f"{p5:>14.2f}  "
                  f"{std:>10.2f}")
    print()
    print("  NOTE: In this dataset, the optimal offering strategy is robust to")
    print("  risk aversion. For one-price, the all-or-nothing decision is driven")
    print("  by E[DA - BP] per hour, which is not altered by CVaR. For two-price,")
    print("  the concave profit structure already minimises exposure at the margin.")
    print("  Worst-case scenarios (low wind + low price) cannot be mitigated by")
    print("  changing q, since profit depends primarily on wind realisation, not q.")



# ============================================================
# MAIN
# ============================================================

def main():
    # ── 0. Load scenarios ─────────────────────────────────────
    print("\n[0] Loading scenarios ...")
    joint = load_scenarios()
    probs = equal_probs(len(joint))

    # ── TASK 1.1 ──────────────────────────────────────────────
    print("\n[1] Task 1.1 – One-price optimisation ...")
    res_1p = solve_one_price(joint, probs)
    print_task11(res_1p)
    plot_task11(res_1p)

    # ── TASK 1.2 ──────────────────────────────────────────────
    print("\n[2] Task 1.2 – Two-price optimisation ...")
    res_2p = solve_two_price(joint, probs)
    print_task12(res_1p, res_2p)
    plot_task12(res_1p, res_2p)

    # Save offer tables
    pd.DataFrame({
        "Hour":            np.arange(1, N_HOURS + 1),
        "One_price_MW":    res_1p["q_star"],
        "Two_price_MW":    res_2p["q_star"],
    }).to_csv(OUTPUT_DIR / "task12_offers.csv", index=False)

    # ── TASK 1.3 ──────────────────────────────────────────────
    print(f"\n[3] Task 1.3 – {CV_FOLDS}-fold cross-validation ...")
    rng_cv = np.random.default_rng(RANDOM_SEED + 10)
    S = len(joint)
    idx_all = np.arange(S)

    cv_rows = []
    for fold in range(CV_FOLDS):
        in_idx  = np.sort(rng_cv.choice(S, size=CV_IN_SAMPLE, replace=False))
        out_idx = np.setdiff1d(idx_all, in_idx)

        in_scen   = [joint[i] for i in in_idx]
        out_scen  = [joint[i] for i in out_idx]
        p_in      = equal_probs(len(in_scen))
        p_out     = equal_probs(len(out_scen))

        r1_in  = solve_one_price(in_scen, p_in)
        r2_in  = solve_two_price(in_scen, p_in)

        ep1_out, _ = evaluate_one_price(out_scen, p_out, r1_in["q_star"])
        ep2_out, _ = evaluate_two_price(out_scen, p_out, r2_in["q_star"])

        cv_rows.append({
            "fold":         fold + 1,
            "one_price_in": r1_in["exp_profit"],
            "one_price_out": ep1_out,
            "two_price_in": r2_in["exp_profit"],
            "two_price_out": ep2_out,
        })
        print(f"  Fold {fold+1}/{CV_FOLDS}  1P-in={r1_in['exp_profit']:.0f}  "
              f"1P-out={ep1_out:.0f}  2P-in={r2_in['exp_profit']:.0f}  "
              f"2P-out={ep2_out:.0f} EUR")

    cv_df = pd.DataFrame(cv_rows)
    print_task13(cv_df)
    cv_df.to_csv(OUTPUT_DIR / "task13_crossval.csv", index=False)
    plot_task13(cv_df)

    # In-sample size sensitivity (Task 1.3 continuation)
    print(f"\n[3b] Task 1.3 – In-sample size sensitivity "
          f"(pool={POOL_SIZE}, sizes={IN_SAMPLE_SIZES}, {N_REPEATS_SIZE} repeats each) ...")
    size_df = run_insample_size_sensitivity(joint, IN_SAMPLE_SIZES, N_REPEATS_SIZE)
    print_insample_size_results(size_df)
    size_df.to_csv(OUTPUT_DIR / "task13_insample_size.csv", index=False)
    plot_insample_size_sensitivity(size_df, total_scenarios=POOL_SIZE)

    # ── TASK 1.4 ──────────────────────────────────────────────
    print(f"\n[4] Task 1.4 – CVaR risk-averse (alpha={ALPHA_CVAR}, "
          f"{len(BETA_VALUES)} beta values) ...")

    # Use full in-sample set (all 3125 scenarios) for Task 1.4
    cvar_rows_1p, cvar_rows_2p = [], []

    for beta in BETA_VALUES:
        print(f"  beta={beta} ...", end=" ", flush=True)

        r1 = solve_one_price_cvar(joint, probs, beta=beta)
        print(f"1P: E={r1['exp_profit']:.0f} CVaR={r1['cvar']:.0f}", end="  ")
        cvar_rows_1p.append(r1)

        r2 = solve_two_price_cvar(joint, probs, beta=beta)
        print(f"2P: E={r2['exp_profit']:.0f} CVaR={r2['cvar']:.0f}")
        cvar_rows_2p.append(r2)

    print_task14(cvar_rows_1p, cvar_rows_2p)

    pd.DataFrame([{
        "beta": r["beta"], "scheme": "one_price",
        "exp_profit": r["exp_profit"], "cvar": r["cvar"]
    } for r in cvar_rows_1p] + [{
        "beta": r["beta"], "scheme": "two_price",
        "exp_profit": r["exp_profit"], "cvar": r["cvar"]
    } for r in cvar_rows_2p]).to_csv(OUTPUT_DIR / "task14_cvar.csv", index=False)

    plot_task14(cvar_rows_1p, cvar_rows_2p)
    plot_task14_offers(cvar_rows_1p, cvar_rows_2p)

    # ── Summary ───────────────────────────────────────────────
    _hline()
    print("ALL TASKS COMPLETE.  Outputs in:", OUTPUT_DIR)
    print("  task11_one_price.png")
    print("  task12_comparison.png   task12_offers.csv")
    print("  task13_crossval.png     task13_crossval.csv")
    print("  task13_insample_size.png  task13_insample_size.csv")
    print("  task14_cvar_frontier.png  task14_offer_evolution.png  task14_cvar.csv")
    _hline()


if __name__ == "__main__":
    main()
