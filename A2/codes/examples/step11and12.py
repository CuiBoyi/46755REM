import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from pathlib import Path
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# USER INPUTS
# ============================================================

CAPACITY_MW = 500.0
N_HOURS = 24

N_WIND = 25
N_PRICE = 25
N_IMB = 5
SCHEMES = ("One-price", "Two-price")
CV_TOTAL_SCENARIOS = 1600
CV_N_FOLDS = 8
CV_IN_SAMPLE_SIZE = 200
SENSITIVITY_IN_SAMPLE_SIZES = [50, 100, 200]
SENSITIVITY_N_REPEATS = 8
CVAR_ALPHA = 0.90
CVAR_BETA_VALUES = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
CVAR_ROBUSTNESS_BETA = 1.0

BASE_DIR = Path(__file__).resolve().parents[1]
HOURLY_DATA_PATH = BASE_DIR / "outputs" / "raw_data_hourly_timeseries.csv"

# The raw wind data is an aggregate forecast with values above the 500 MW model
# capacity. Scaling preserves the profile shape for a 500 MW portfolio.
NORMALIZE_WIND_TO_CAPACITY = True

# Real-time system imbalance generation:
# SI = 1 -> system deficit
# SI = 0 -> system surplus
P_DEFICIT = 0.5  # Bernoulli probability for each hour

RANDOM_SEED = 42
GUROBI_OUTPUT_FLAG = 0


# ============================================================
# STEP 0. LOAD / PREPARE YOUR READY DATA
# ============================================================
# This script expects:
#
# wind_scenarios:      shape = (N_WIND, 24)
# da_price_scenarios:  shape = (N_PRICE, 24)
#
# Units:
# - wind in MW
# - prices in EUR/MWh
#
# The hourly input is built from inputs/raw_data.xlsx and saved as:
# outputs/raw_data_hourly_timeseries.csv
# ============================================================

def _daily_scenarios_from_hourly(df, value_col, count_col, n_scenarios, n_hours=N_HOURS):
    """
    Convert a continuous hourly time series into daily 24-hour scenarios.

    Only complete days are used. This keeps missing or partial source days from
    silently entering the stochastic program.
    """
    complete = df[df[value_col].notna() & (df[count_col] == 4)].copy()
    complete["date"] = complete["hour_start"].dt.date
    complete["hour"] = complete["hour_start"].dt.hour

    scenarios = []
    scenario_dates = []

    for date, day in complete.groupby("date", sort=True):
        day = day.sort_values("hour_start")
        if len(day) != n_hours or set(day["hour"]) != set(range(n_hours)):
            continue

        scenarios.append(day[value_col].to_numpy(dtype=float))
        scenario_dates.append(date)

        if len(scenarios) == n_scenarios:
            break

    if len(scenarios) < n_scenarios:
        raise ValueError(
            f"Only found {len(scenarios)} complete daily scenarios for {value_col}; "
            f"expected {n_scenarios}."
        )

    return np.vstack(scenarios), scenario_dates


def load_ready_data(hourly_data_path=HOURLY_DATA_PATH):
    """
    Load real hourly wind and day-ahead price data.

    Expected outputs:
        wind_scenarios: np.ndarray of shape (25, 24)
        da_price_scenarios: np.ndarray of shape (25, 24)
    """
    hourly_data_path = Path(hourly_data_path)
    if not hourly_data_path.exists():
        raise FileNotFoundError(
            f"Hourly data file not found: {hourly_data_path}. "
            "Create it first from inputs/raw_data.xlsx."
        )

    hourly = pd.read_csv(hourly_data_path)
    hourly["hour_start"] = pd.to_datetime(hourly["hour_start"], utc=True)
    hourly["wind_forecast_mw"] = pd.to_numeric(hourly["wind_forecast_mw"], errors="coerce")
    hourly["price"] = pd.to_numeric(hourly["price"], errors="coerce")

    wind_scenarios, wind_dates = _daily_scenarios_from_hourly(
        hourly, "wind_forecast_mw", "wind_quarters", N_WIND
    )
    da_price_scenarios, price_dates = _daily_scenarios_from_hourly(
        hourly, "price", "price_quarters", N_PRICE
    )

    if NORMALIZE_WIND_TO_CAPACITY:
        max_wind = np.nanmax(wind_scenarios)
        if max_wind <= 0:
            raise ValueError("Cannot normalize wind data because all wind values are non-positive.")
        wind_scenarios = wind_scenarios / max_wind * CAPACITY_MW

    assert wind_scenarios.shape == (N_WIND, N_HOURS), \
        f"wind_scenarios must have shape {(N_WIND, N_HOURS)}"
    assert da_price_scenarios.shape == (N_PRICE, N_HOURS), \
        f"da_price_scenarios must have shape {(N_PRICE, N_HOURS)}"

    wind_scenarios = np.clip(wind_scenarios, 0.0, CAPACITY_MW)

    print(f"Loaded wind scenarios from {hourly_data_path}")
    print(f"Wind scenario dates:  {wind_dates[0]} to {wind_dates[-1]}")
    print(f"Price scenario dates: {price_dates[0]} to {price_dates[-1]}")
    print(f"Wind range:  {wind_scenarios.min():.2f} to {wind_scenarios.max():.2f} MW")
    print(f"Price range: {da_price_scenarios.min():.2f} to {da_price_scenarios.max():.2f} EUR/MWh")

    return wind_scenarios, da_price_scenarios


# ============================================================
# STEP 1. GENERATE IMBALANCE SCENARIOS
# ============================================================

def generate_imbalance_scenarios(n_imb=N_IMB, n_hours=N_HOURS, p_deficit=P_DEFICIT, seed=RANDOM_SEED):
    """
    Generate n_imb scenarios of system imbalance (SI).
    SI[t] = 1 means system deficit at hour t
    SI[t] = 0 means system surplus at hour t
    """
    rng = np.random.default_rng(seed)
    imbalance_scenarios = rng.binomial(1, p_deficit, size=(n_imb, n_hours))
    return imbalance_scenarios


# ============================================================
# STEP 2. GENERATE BALANCING PRICE SCENARIOS
# ============================================================
# Rule from your assignment:
#   if SI = 1 (deficit): BP = 1.25 * DA
#   if SI = 0 (surplus): BP = 0.85 * DA
# ============================================================

def generate_balancing_prices(da_price_scenarios, imbalance_scenarios):
    """
    For each price scenario p and imbalance scenario k, generate a BP path:
        BP[t] = 1.25 * DA[t] if SI[t] = 1
              = 0.85 * DA[t] if SI[t] = 0

    Returns:
        bp_scenarios: shape = (N_PRICE, N_IMB, 24)
    """
    n_price, n_hours = da_price_scenarios.shape
    n_imb, n_hours2 = imbalance_scenarios.shape
    assert n_hours == n_hours2

    bp_scenarios = np.zeros((n_price, n_imb, n_hours))

    for p in range(n_price):
        for k in range(n_imb):
            si = imbalance_scenarios[k]
            da = da_price_scenarios[p]
            bp_scenarios[p, k, :] = np.where(si == 1, 1.25 * da, 0.85 * da)

    return bp_scenarios


# ============================================================
# STEP 3. BUILD JOINT SCENARIOS
# ============================================================
# Joint scenario index s corresponds to a triple:
#   (wind scenario w, price scenario p, imbalance scenario k)
#
# Total number = 25 * 25 * 5 = 3125
# ============================================================

def build_joint_scenarios(wind_scenarios, da_price_scenarios, imbalance_scenarios, bp_scenarios):
    joint = []

    for w in range(wind_scenarios.shape[0]):
        for p in range(da_price_scenarios.shape[0]):
            for k in range(imbalance_scenarios.shape[0]):
                joint.append({
                    "w_idx": w,
                    "p_idx": p,
                    "k_idx": k,
                    "wind": wind_scenarios[w].copy(),
                    "da": da_price_scenarios[p].copy(),
                    "si": imbalance_scenarios[k].copy(),
                    "bp": bp_scenarios[p, k].copy()
                })

    return joint


def build_equal_probabilities(n_scenarios):
    """Equal probabilities over the supplied scenario group."""
    return np.full(n_scenarios, 1.0 / n_scenarios)


def scenario_subset(joint_scenarios, indices):
    return [joint_scenarios[idx] for idx in indices]


def complement_indices(pool_indices, selected_indices):
    selected = set(selected_indices)
    return np.array([idx for idx in pool_indices if idx not in selected], dtype=int)


def build_cross_validation_folds(
    joint_scenarios,
    total_scenarios=CV_TOTAL_SCENARIOS,
    n_folds=CV_N_FOLDS,
    seed=RANDOM_SEED,
):
    """
    Select a reproducible 1600-scenario pool and split it into 8 folds of 200.

    In each ex-post run, one fold is used as the in-sample optimization group
    and the remaining seven folds are used as the out-of-sample group.
    """
    n_total = len(joint_scenarios)
    if total_scenarios > n_total:
        raise ValueError(
            f"Requested {total_scenarios} scenarios, but only {n_total} are available."
        )
    if total_scenarios % n_folds != 0:
        raise ValueError("total_scenarios must be divisible by n_folds.")

    rng = np.random.default_rng(seed)
    selected_indices = rng.choice(n_total, size=total_scenarios, replace=False)
    rng.shuffle(selected_indices)

    fold_size = total_scenarios // n_folds
    if fold_size != CV_IN_SAMPLE_SIZE:
        raise ValueError(
            f"Fold size is {fold_size}, but CV_IN_SAMPLE_SIZE is {CV_IN_SAMPLE_SIZE}."
        )

    return [
        np.sort(selected_indices[i * fold_size:(i + 1) * fold_size])
        for i in range(n_folds)
    ]


def select_scenario_pool(
    joint_scenarios,
    total_scenarios=CV_TOTAL_SCENARIOS,
    seed=RANDOM_SEED,
):
    """Select a reproducible scenario pool for ex-post analysis."""
    n_total = len(joint_scenarios)
    if total_scenarios > n_total:
        raise ValueError(
            f"Requested {total_scenarios} scenarios, but only {n_total} are available."
        )

    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total, size=total_scenarios, replace=False))


# ============================================================
# STEP 4A. SOLVE ONE-PRICE SCHEME
# ============================================================
# Revenue per hour in scenario s:
#
#   revenue = λ_DA * q + λ_B * (W - q)
#
# where:
#   q = day-ahead offer
#   W = realized wind production
#
# This is linear.
# ============================================================

def solve_one_price(joint_scenarios, scenario_probabilities, capacity_mw=CAPACITY_MW):
    n_scen = len(joint_scenarios)
    n_hours = len(joint_scenarios[0]["wind"])
    solve_start = perf_counter()

    m = gp.Model("wind_offer_one_price")
    m.Params.OutputFlag = GUROBI_OUTPUT_FLAG

    # First-stage decision: hourly day-ahead offer
    q = m.addVars(n_hours, lb=0.0, ub=capacity_mw, name="q")

    # Expected profit objective
    obj = gp.LinExpr()

    for s in range(n_scen):
        prob = scenario_probabilities[s]
        W = joint_scenarios[s]["wind"]
        DA = joint_scenarios[s]["da"]
        BP = joint_scenarios[s]["bp"]

        for t in range(n_hours):
            obj += prob * (DA[t] * q[t] + BP[t] * (W[t] - q[t])) #W[t] - q[t]=deviation between actual wind and the day-ahead offer.



    m.setObjective(obj, GRB.MAXIMIZE)
    m.optimize()

    if m.Status != GRB.OPTIMAL:
        raise RuntimeError("One-price model did not solve to optimality.")

    q_star = np.array([q[t].X for t in range(n_hours)])

    scenario_profits = one_price_profits_from_offer(scenario_arrays(joint_scenarios), q_star)

    expected_profit = np.dot(scenario_probabilities, scenario_profits)
    solve_time_seconds = perf_counter() - solve_start

    return {
        "model": m,
        "offer_mw": q_star,
        "scenario_profits": scenario_profits,
        "expected_profit": expected_profit,
        "solve_time_seconds": solve_time_seconds
    }


def evaluate_one_price(joint_scenarios, scenario_probabilities, offer_mw):
    """Evaluate a fixed day-ahead offer under the one-price scheme."""
    scenario_profits = one_price_profits_from_offer(scenario_arrays(joint_scenarios), offer_mw)
    expected_profit = np.dot(scenario_probabilities, scenario_profits)

    return {
        "model": None,
        "offer_mw": offer_mw,
        "scenario_profits": scenario_profits,
        "expected_profit": expected_profit
    }


# ============================================================
# STEP 4B. SOLVE TWO-PRICE SCHEME
# ============================================================
# Use positive/negative deviation decomposition:
#
#   W - q = over - under
#   over  >= 0   (positive deviation)
#   under >= 0   (negative deviation)
#
# Under the two-price scheme:
#
# if SI = 1 (system deficit):
#   positive deviation is beneficial  -> settled at DA
#   negative deviation is harmful     -> settled at BP
#
# if SI = 0 (system surplus):
#   positive deviation is harmful     -> settled at BP
#   negative deviation is beneficial  -> settled at DA
#
# Revenue:
#   λ_DA * q
#   + settlement_for_over
#   - settlement_for_under
# ============================================================

def solve_two_price(joint_scenarios, scenario_probabilities, capacity_mw=CAPACITY_MW):
    n_scen = len(joint_scenarios)
    n_hours = len(joint_scenarios[0]["wind"])
    solve_start = perf_counter()

    probs = np.asarray(scenario_probabilities, dtype=float)
    wind = np.vstack([scenario["wind"] for scenario in joint_scenarios])
    da = np.vstack([scenario["da"] for scenario in joint_scenarios])
    bp = np.vstack([scenario["bp"] for scenario in joint_scenarios])
    si = np.vstack([scenario["si"] for scenario in joint_scenarios])

    q_star = np.zeros(n_hours)

    # The two-price objective is separable by hour and concave piecewise-linear
    # in q. Its breakpoints are the realized wind values, so checking those
    # values plus the bounds gives the exact sample optimum without adding a
    # large variable/constraint block to Gurobi.
    for t in range(n_hours):
        candidates = np.unique(np.concatenate(([0.0, capacity_mw], wind[:, t])))
        candidates = candidates[(candidates >= 0.0) & (candidates <= capacity_mw)]

        best_q = 0.0
        best_expected_profit = -np.inf

        for q_candidate in candidates:
            delta = wind[:, t] - q_candidate
            over_val = np.maximum(delta, 0.0)
            under_val = np.maximum(-delta, 0.0)

            hourly_profit = da[:, t] * q_candidate
            deficit = si[:, t] == 1
            surplus = ~deficit

            hourly_profit[deficit] += da[deficit, t] * over_val[deficit]
            hourly_profit[deficit] -= bp[deficit, t] * under_val[deficit]
            hourly_profit[surplus] += bp[surplus, t] * over_val[surplus]
            hourly_profit[surplus] -= da[surplus, t] * under_val[surplus]

            expected_profit = np.dot(probs, hourly_profit)
            if expected_profit > best_expected_profit:
                best_expected_profit = expected_profit
                best_q = q_candidate

        q_star[t] = best_q

    scenario_profits = two_price_profits_from_offer(scenario_arrays(joint_scenarios), q_star)
    expected_profit = np.dot(scenario_probabilities, scenario_profits)
    solve_time_seconds = perf_counter() - solve_start

    return {
        "model": None,
        "offer_mw": q_star,
        "scenario_profits": scenario_profits,
        "expected_profit": expected_profit,
        "solve_time_seconds": solve_time_seconds
    }


def evaluate_two_price(joint_scenarios, scenario_probabilities, offer_mw):
    """Evaluate a fixed day-ahead offer under the two-price scheme."""
    scenario_profits = two_price_profits_from_offer(scenario_arrays(joint_scenarios), offer_mw)
    expected_profit = np.dot(scenario_probabilities, scenario_profits)

    return {
        "model": None,
        "offer_mw": offer_mw,
        "scenario_profits": scenario_profits,
        "expected_profit": expected_profit
    }


def solve_and_evaluate(scheme, in_sample_scenarios, out_of_sample_scenarios):
    in_probs = build_equal_probabilities(len(in_sample_scenarios))
    out_probs = build_equal_probabilities(len(out_of_sample_scenarios))

    if scheme == "One-price":
        in_result = solve_one_price(in_sample_scenarios, in_probs, CAPACITY_MW)
        out_result = evaluate_one_price(out_of_sample_scenarios, out_probs, in_result["offer_mw"])
    elif scheme == "Two-price":
        in_result = solve_two_price(in_sample_scenarios, in_probs, CAPACITY_MW)
        out_result = evaluate_two_price(out_of_sample_scenarios, out_probs, in_result["offer_mw"])
    else:
        raise ValueError(f"Unsupported scheme: {scheme}")

    return in_result, out_result


def empirical_profit_cvar(profits, alpha=CVAR_ALPHA):
    """Lower-tail empirical CVaR for profit outcomes."""
    # For profit, risk is represented by the left tail.
    # At alpha=0.90, CVaR is the average profit of the worst 10% scenarios.
    tail_count = int(np.ceil((1.0 - alpha) * len(profits)))
    tail_count = max(1, tail_count)
    return np.mean(np.sort(profits)[:tail_count])



def scenario_arrays(joint_scenarios):
    # Convert the list-of-dicts scenario representation into matrices so the
    # risk analysis can evaluate many offers quickly.
    return {
        "wind": np.vstack([scenario["wind"] for scenario in joint_scenarios]),
        "da": np.vstack([scenario["da"] for scenario in joint_scenarios]),
        "bp": np.vstack([scenario["bp"] for scenario in joint_scenarios]),
        "si": np.vstack([scenario["si"] for scenario in joint_scenarios]),
    }


def one_price_profits_from_offer(arrays, offer_mw):
    # One-price settlement: all deviations are settled at the balancing price.
    wind = arrays["wind"]
    da = arrays["da"]
    bp = arrays["bp"]
    return np.sum(da * offer_mw + bp * (wind - offer_mw), axis=1)


def two_price_profits_from_offer(arrays, offer_mw):
    # Two-price settlement: positive and negative deviations receive different
    # settlement prices depending on the system imbalance state.
    wind = arrays["wind"]
    da = arrays["da"]
    bp = arrays["bp"]
    si = arrays["si"]

    delta = wind - offer_mw
    over = np.maximum(delta, 0.0)
    under = np.maximum(-delta, 0.0)

    profits = da * offer_mw
    deficit = si == 1
    surplus = ~deficit
    profits = profits + np.where(deficit, da * over - bp * under, bp * over - da * under)
    return np.sum(profits, axis=1)


def solve_one_price_cvar_gurobi(
    joint_scenarios,
    beta,
    alpha=CVAR_ALPHA,
    capacity_mw=CAPACITY_MW,
):
    """Gurobi CVaR model for the one-price scheme."""
    arrays = scenario_arrays(joint_scenarios)
    wind = arrays["wind"]
    da = arrays["da"]
    bp = arrays["bp"]

    n_scen, n_hours = wind.shape
    probs = build_equal_probabilities(n_scen)

    m = gp.Model("one_price_cvar")
    m.Params.OutputFlag = GUROBI_OUTPUT_FLAG

    # Decision variables
    q = m.addVars(n_hours, lb=0.0, ub=capacity_mw, name="q")
    zeta = m.addVar(lb=-GRB.INFINITY, name="zeta")
    eta = m.addVars(n_scen, lb=0.0, name="eta")

    # Scenario profit expressions
    profit = {}
    for s in range(n_scen):
        profit[s] = gp.quicksum(
            da[s, t] * q[t] + bp[s, t] * (wind[s, t] - q[t])
            for t in range(n_hours)
        )

    # CVaR shortfall constraints:
    # eta_s >= zeta - profit_s
    for s in range(n_scen):
        m.addConstr(eta[s] >= zeta - profit[s], name=f"cvar_shortfall_{s}")

    # Objective:
    # max E[profit] + beta * (zeta - 1/(1-alpha) * E[eta])
    expected_profit = gp.quicksum(probs[s] * profit[s] for s in range(n_scen))
    cvar_term = zeta - (1.0 / (1.0 - alpha)) * gp.quicksum(
        probs[s] * eta[s] for s in range(n_scen)
    )

    m.setObjective(expected_profit + beta * cvar_term, GRB.MAXIMIZE)
    m.optimize()

    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"One-price Gurobi CVaR model failed. Status code: {m.Status}")

    offer = np.array([q[t].X for t in range(n_hours)])
    return np.clip(offer, 0.0, capacity_mw), m


def solve_two_price_cvar_gurobi(
    joint_scenarios,
    beta,
    alpha=CVAR_ALPHA,
    capacity_mw=CAPACITY_MW,
):
    """Gurobi CVaR model for the two-price scheme."""
    arrays = scenario_arrays(joint_scenarios)
    wind = arrays["wind"]
    da = arrays["da"]
    bp = arrays["bp"]
    si = arrays["si"]

    n_scen, n_hours = wind.shape
    probs = build_equal_probabilities(n_scen)

    m = gp.Model("two_price_cvar")
    m.Params.OutputFlag = GUROBI_OUTPUT_FLAG

    # First-stage offer
    q = m.addVars(n_hours, lb=0.0, ub=capacity_mw, name="q")

    # Scenario-dependent deviation variables
    over = m.addVars(n_scen, n_hours, lb=0.0, name="over")
    under = m.addVars(n_scen, n_hours, lb=0.0, name="under")

    # CVaR variables
    zeta = m.addVar(lb=-GRB.INFINITY, name="zeta")
    eta = m.addVars(n_scen, lb=0.0, name="eta")

    # Deviation balance:
    # wind[s,t] - q[t] = over[s,t] - under[s,t]
    for s in range(n_scen):
        for t in range(n_hours):
            m.addConstr(
                wind[s, t] - q[t] == over[s, t] - under[s, t],
                name=f"dev_balance_{s}_{t}"
            )

            # Tight physical bounds
            m.addConstr(over[s, t] <= wind[s, t], name=f"over_bound_{s}_{t}")
            m.addConstr(under[s, t] <= capacity_mw - wind[s, t], name=f"under_bound_{s}_{t}")

    # Scenario profit expressions
    profit = {}
    for s in range(n_scen):
        expr = gp.LinExpr()
        for t in range(n_hours):
            expr += da[s, t] * q[t]

            if si[s, t] == 1:  # system deficit
                expr += da[s, t] * over[s, t]
                expr -= bp[s, t] * under[s, t]
            else:  # system surplus
                expr += bp[s, t] * over[s, t]
                expr -= da[s, t] * under[s, t]

        profit[s] = expr

    # CVaR shortfall constraints:
    # eta_s >= zeta - profit_s
    for s in range(n_scen):
        m.addConstr(eta[s] >= zeta - profit[s], name=f"cvar_shortfall_{s}")

    # Objective:
    # max E[profit] + beta * (zeta - 1/(1-alpha) * E[eta])
    expected_profit = gp.quicksum(probs[s] * profit[s] for s in range(n_scen))
    cvar_term = zeta - (1.0 / (1.0 - alpha)) * gp.quicksum(
        probs[s] * eta[s] for s in range(n_scen)
    )

    m.setObjective(expected_profit + beta * cvar_term, GRB.MAXIMIZE)
    m.optimize()

    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Two-price Gurobi CVaR model failed. Status code: {m.Status}")

    offer = np.array([q[t].X for t in range(n_hours)])
    return np.clip(offer, 0.0, capacity_mw), m


def solve_risk_averse_offer(
    joint_scenarios,
    beta,
    scheme,
    initial_offer=None,
    alpha=CVAR_ALPHA,
    capacity_mw=CAPACITY_MW,
):
    """
    Solve:
        max E[profit] + beta * CVaR_alpha(profit)

    using empirical lower-tail CVaR over the supplied in-sample scenarios.
    """
    solve_start = perf_counter()
    probabilities = build_equal_probabilities(len(joint_scenarios))

    if scheme == "One-price":
        profit_function = one_price_profits_from_offer
        risk_neutral_solver = solve_one_price
        cvar_solver = solve_one_price_cvar_gurobi
    elif scheme == "Two-price":
        profit_function = two_price_profits_from_offer
        risk_neutral_solver = solve_two_price
        cvar_solver = solve_two_price_cvar_gurobi
    else:
        raise ValueError(f"Unsupported scheme: {scheme}")

    if beta == 0.0:
        # beta = 0 => risk-neutral solution
        if initial_offer is None:
            initial_offer = risk_neutral_solver(
                joint_scenarios,
                probabilities,
                capacity_mw
            )["offer_mw"]

        offer = np.asarray(initial_offer, dtype=float)
        success = True
        message = "risk-neutral solution"
    else:
        offer, model = cvar_solver(
            joint_scenarios,
            beta,
            alpha=alpha,
            capacity_mw=capacity_mw,
        )
        success = (model.Status == GRB.OPTIMAL)
        message = f"Gurobi status code {model.Status}"

    arrays = scenario_arrays(joint_scenarios)
    profits = profit_function(arrays, offer)
    expected_profit = np.mean(profits)
    cvar = empirical_profit_cvar(profits, alpha=alpha)
    solve_time_seconds = perf_counter() - solve_start

    return {
        "scheme": scheme,
        "beta": beta,
        "alpha": alpha,
        "offer_mw": offer,
        "scenario_profits": profits,
        "expected_profit": expected_profit,
        "cvar": cvar,
        "solve_time_seconds": solve_time_seconds,
        "success": success,
        "message": message,
    }



def evaluate_risk_averse_offer(joint_scenarios, offer_mw, scheme, alpha=CVAR_ALPHA):
    # Evaluate a fixed in-sample offer on another scenario set, usually the
    # out-of-sample scenarios.
    arrays = scenario_arrays(joint_scenarios)
    if scheme == "One-price":
        profits = one_price_profits_from_offer(arrays, offer_mw)
    elif scheme == "Two-price":
        profits = two_price_profits_from_offer(arrays, offer_mw)
    else:
        raise ValueError(f"Unsupported scheme: {scheme}")

    return {
        "expected_profit": np.mean(profits),
        "cvar": empirical_profit_cvar(profits, alpha=alpha),
        "profit_std": np.std(profits, ddof=0),
        "profit_min": np.min(profits),
        "profit_max": np.max(profits),
        "scenario_profits": profits,
    }


def run_ex_post_cross_validation(joint_scenarios):
    folds = build_cross_validation_folds(joint_scenarios)
    selected_indices = np.sort(np.concatenate(folds))
    records = []
    offer_records = []
    group_records = []

    print("\n" + "=" * 70)
    print("EX-POST 8-FOLD CROSS-VALIDATION")
    print("=" * 70)
    print(f"Total CV scenarios = {len(selected_indices)}")
    print(f"In-sample scenarios per run = {CV_IN_SAMPLE_SIZE}")
    print(f"Out-of-sample scenarios per run = {CV_TOTAL_SCENARIOS - CV_IN_SAMPLE_SIZE}")

    for fold_id, in_sample_indices in enumerate(folds, start=1):
        out_of_sample_indices = complement_indices(selected_indices, in_sample_indices)
        in_sample_scenarios = scenario_subset(joint_scenarios, in_sample_indices)
        out_of_sample_scenarios = scenario_subset(joint_scenarios, out_of_sample_indices)

        print(f"\nFold {fold_id}: {len(in_sample_scenarios)} in-sample, "
              f"{len(out_of_sample_scenarios)} out-of-sample")

        for scheme in SCHEMES:
            in_result, out_result = solve_and_evaluate(
                scheme,
                in_sample_scenarios,
                out_of_sample_scenarios,
            )
            records.append({
                "Fold": fold_id,
                "Scheme": scheme,
                "In_Sample_Scenarios": len(in_sample_scenarios),
                "Out_Of_Sample_Scenarios": len(out_of_sample_scenarios),
                "In_Sample_Expected_Profit_EUR": in_result["expected_profit"],
                "Out_Of_Sample_Expected_Profit_EUR": out_result["expected_profit"],
                "Generalization_Gap_EUR": in_result["expected_profit"] - out_result["expected_profit"],
                "Solve_Time_Seconds": in_result["solve_time_seconds"],
            })

            for hour, offer in enumerate(in_result["offer_mw"], start=1):
                offer_records.append({
                    "Fold": fold_id,
                    "Scheme": scheme,
                    "Hour": hour,
                    "DA_Offer_MW": offer,
                })

        for group_name, indices in [
            ("in_sample", in_sample_indices),
            ("out_of_sample", out_of_sample_indices),
        ]:
            for idx in indices:
                scenario = joint_scenarios[idx]
                group_records.append({
                    "Fold": fold_id,
                    "scenario_id": idx,
                    "group": group_name,
                    "w_idx": scenario["w_idx"],
                    "p_idx": scenario["p_idx"],
                    "k_idx": scenario["k_idx"],
                })

    cv_results = pd.DataFrame(records)
    cv_offers = pd.DataFrame(offer_records)
    cv_groups = pd.DataFrame(group_records)
    cv_summary = (
        cv_results
        .groupby("Scheme", as_index=False)
        .agg(
            Avg_In_Sample_Expected_Profit_EUR=("In_Sample_Expected_Profit_EUR", "mean"),
            Avg_Out_Of_Sample_Expected_Profit_EUR=("Out_Of_Sample_Expected_Profit_EUR", "mean"),
            Avg_Generalization_Gap_EUR=("Generalization_Gap_EUR", "mean"),
            Std_Out_Of_Sample_Expected_Profit_EUR=("Out_Of_Sample_Expected_Profit_EUR", "std"),
            Avg_Solve_Time_Seconds=("Solve_Time_Seconds", "mean"),
        )
    )

    print("\nCross-validation summary:")
    print(cv_summary.to_string(index=False))

    return cv_results, cv_summary, cv_offers, cv_groups


def run_in_sample_size_sensitivity(joint_scenarios):
    scenario_pool_indices = select_scenario_pool(joint_scenarios)
    records = []

    print("\n" + "=" * 70)
    print("IN-SAMPLE SIZE SENSITIVITY")
    print("=" * 70)
    print(f"Total scenario pool = {len(scenario_pool_indices)}")
    print(f"Sizes tested = {SENSITIVITY_IN_SAMPLE_SIZES}")
    print(f"Random repeats per size = {SENSITIVITY_N_REPEATS}")

    for in_sample_size in SENSITIVITY_IN_SAMPLE_SIZES:
        if not 0 < in_sample_size < len(scenario_pool_indices):
            raise ValueError(
                f"Invalid sensitivity in-sample size {in_sample_size} "
                f"for pool size {len(scenario_pool_indices)}."
            )

        print(f"\nTesting in-sample size {in_sample_size}")

        for repeat in range(1, SENSITIVITY_N_REPEATS + 1):
            rng = np.random.default_rng(RANDOM_SEED + 1000 * in_sample_size + repeat)
            in_sample_indices = np.sort(
                rng.choice(scenario_pool_indices, size=in_sample_size, replace=False)
            )
            out_of_sample_indices = complement_indices(scenario_pool_indices, in_sample_indices)
            in_sample_scenarios = scenario_subset(joint_scenarios, in_sample_indices)
            out_of_sample_scenarios = scenario_subset(joint_scenarios, out_of_sample_indices)

            for scheme in SCHEMES:
                in_result, out_result = solve_and_evaluate(
                    scheme,
                    in_sample_scenarios,
                    out_of_sample_scenarios,
                )
                records.append({
                    "In_Sample_Size": in_sample_size,
                    "Out_Of_Sample_Size": len(out_of_sample_scenarios),
                    "Repeat": repeat,
                    "Scheme": scheme,
                    "In_Sample_Expected_Profit_EUR": in_result["expected_profit"],
                    "Out_Of_Sample_Expected_Profit_EUR": out_result["expected_profit"],
                    "Generalization_Gap_EUR": in_result["expected_profit"] - out_result["expected_profit"],
                    "Abs_Generalization_Gap_EUR": abs(
                        in_result["expected_profit"] - out_result["expected_profit"]
                    ),
                    "Solve_Time_Seconds": in_result["solve_time_seconds"],
                })

    sensitivity_results = pd.DataFrame(records)
    sensitivity_summary = (
        sensitivity_results
        .groupby(["Scheme", "In_Sample_Size"], as_index=False)
        .agg(
            Avg_Out_Of_Sample_Size=("Out_Of_Sample_Size", "mean"),
            Avg_In_Sample_Expected_Profit_EUR=("In_Sample_Expected_Profit_EUR", "mean"),
            Avg_Out_Of_Sample_Expected_Profit_EUR=("Out_Of_Sample_Expected_Profit_EUR", "mean"),
            Avg_Generalization_Gap_EUR=("Generalization_Gap_EUR", "mean"),
            Avg_Abs_Generalization_Gap_EUR=("Abs_Generalization_Gap_EUR", "mean"),
            Std_Out_Of_Sample_Expected_Profit_EUR=("Out_Of_Sample_Expected_Profit_EUR", "std"),
            Avg_Solve_Time_Seconds=("Solve_Time_Seconds", "mean"),
        )
    )

    print("\nSensitivity summary:")
    print(sensitivity_summary.to_string(index=False))

    return sensitivity_results, sensitivity_summary


def run_risk_averse_cvar_analysis(joint_scenarios):
    folds = build_cross_validation_folds(joint_scenarios)
    selected_indices = np.sort(np.concatenate(folds))

    # Use the first CV fold to trace the beta frontier,
    # then evaluate on the remaining seven folds.
    in_sample_indices = folds[0]
    out_of_sample_indices = complement_indices(selected_indices, in_sample_indices)
    in_sample_scenarios = scenario_subset(joint_scenarios, in_sample_indices)
    out_of_sample_scenarios = scenario_subset(joint_scenarios, out_of_sample_indices)

    records = []
    offer_records = []

    print("\n" + "=" * 70)
    print("RISK-AVERSE CVaR ANALYSIS")
    print("=" * 70)
    print(f"CVaR alpha = {CVAR_ALPHA:.2f}")
    print(f"Betas = {CVAR_BETA_VALUES}")
    print(f"Frontier in-sample scenarios = {len(in_sample_scenarios)}")
    print(f"Frontier out-of-sample scenarios = {len(out_of_sample_scenarios)}")

    for scheme in SCHEMES:
        for beta in CVAR_BETA_VALUES:
            result = solve_risk_averse_offer(
                joint_scenarios=in_sample_scenarios,
                beta=beta,
                scheme=scheme,
            )

            out_eval = evaluate_risk_averse_offer(
                joint_scenarios=out_of_sample_scenarios,
                offer_mw=result["offer_mw"],
                scheme=scheme,
            )

            records.append({
                "Scheme": scheme,
                "Beta": beta,
                "Alpha": CVAR_ALPHA,
                "In_Sample_Size": len(in_sample_scenarios),
                "Out_Of_Sample_Size": len(out_of_sample_scenarios),
                "In_Sample_Expected_Profit_EUR": result["expected_profit"],
                "In_Sample_CVaR_EUR": result["cvar"],
                "In_Sample_Profit_Std_EUR": np.std(result["scenario_profits"], ddof=0),
                "Out_Of_Sample_Expected_Profit_EUR": out_eval["expected_profit"],
                "Out_Of_Sample_CVaR_EUR": out_eval["cvar"],
                "Out_Of_Sample_Profit_Std_EUR": out_eval["profit_std"],
                "Out_Of_Sample_Min_Profit_EUR": out_eval["profit_min"],
                "Out_Of_Sample_Max_Profit_EUR": out_eval["profit_max"],
                "Solve_Time_Seconds": result["solve_time_seconds"],
                "Optimization_Success": result["success"],
                "Optimization_Message": result["message"],
            })

            for hour, offer in enumerate(result["offer_mw"], start=1):
                offer_records.append({
                    "Scheme": scheme,
                    "Beta": beta,
                    "Hour": hour,
                    "DA_Offer_MW": offer,
                })

    frontier = pd.DataFrame(records)
    offers = pd.DataFrame(offer_records)

    plot_path = BASE_DIR / "outputs" / "risk_averse_profit_vs_cvar.png"
    plot_risk_averse_frontier(frontier, plot_path)

    robustness = run_risk_averse_fold_robustness(joint_scenarios)

    print("\nRisk-averse frontier:")
    print(frontier[[
        "Scheme",
        "Beta",
        "In_Sample_Expected_Profit_EUR",
        "In_Sample_CVaR_EUR",
        "Out_Of_Sample_Expected_Profit_EUR",
        "Out_Of_Sample_CVaR_EUR",
        "Out_Of_Sample_Profit_Std_EUR",
    ]].to_string(index=False))

    return frontier, offers, robustness, plot_path


def run_risk_averse_fold_robustness(joint_scenarios):
    folds = build_cross_validation_folds(joint_scenarios)
    selected_indices = np.sort(np.concatenate(folds))
    records = []

    print(f"\nRisk-averse fold robustness at beta = {CVAR_ROBUSTNESS_BETA}")

    for fold_id, in_sample_indices in enumerate(folds, start=1):
        out_of_sample_indices = complement_indices(selected_indices, in_sample_indices)
        in_sample_scenarios = scenario_subset(joint_scenarios, in_sample_indices)
        out_of_sample_scenarios = scenario_subset(joint_scenarios, out_of_sample_indices)

        for scheme in SCHEMES:
            result = solve_risk_averse_offer(
                joint_scenarios=in_sample_scenarios,
                beta=CVAR_ROBUSTNESS_BETA,
                scheme=scheme,
            )

            out_eval = evaluate_risk_averse_offer(
                joint_scenarios=out_of_sample_scenarios,
                offer_mw=result["offer_mw"],
                scheme=scheme,
            )

            records.append({
                "Fold": fold_id,
                "Scheme": scheme,
                "Beta": CVAR_ROBUSTNESS_BETA,
                "In_Sample_Expected_Profit_EUR": result["expected_profit"],
                "In_Sample_CVaR_EUR": result["cvar"],
                "Out_Of_Sample_Expected_Profit_EUR": out_eval["expected_profit"],
                "Out_Of_Sample_CVaR_EUR": out_eval["cvar"],
                "Out_Of_Sample_Profit_Std_EUR": out_eval["profit_std"],
                "Mean_Offer_MW": np.mean(result["offer_mw"]),
                "Offer_Std_MW": np.std(result["offer_mw"], ddof=0),
                "Solve_Time_Seconds": result["solve_time_seconds"],
            })

    robustness = pd.DataFrame(records)
    summary = (
        robustness
        .groupby("Scheme", as_index=False)
        .agg(
            Avg_Out_Of_Sample_Expected_Profit_EUR=("Out_Of_Sample_Expected_Profit_EUR", "mean"),
            Std_Out_Of_Sample_Expected_Profit_EUR=("Out_Of_Sample_Expected_Profit_EUR", "std"),
            Avg_Out_Of_Sample_CVaR_EUR=("Out_Of_Sample_CVaR_EUR", "mean"),
            Std_Out_Of_Sample_CVaR_EUR=("Out_Of_Sample_CVaR_EUR", "std"),
            Avg_Mean_Offer_MW=("Mean_Offer_MW", "mean"),
            Std_Mean_Offer_MW=("Mean_Offer_MW", "std"),
        )
    )
    print(summary.to_string(index=False))
    return robustness


def plot_risk_averse_frontier(frontier, output_path):
    schemes = ["One-price", "Two-price"]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2))

    for ax, scheme in zip(axes, schemes):
        # Plot each scheme on its own axis because one-price and two-price
        # outcomes have different profit and CVaR scales.
        data = frontier[frontier["Scheme"] == scheme].copy()
        data = data.sort_values("Beta")
        ax.plot(
            data["Out_Of_Sample_CVaR_EUR"],
            data["Out_Of_Sample_Expected_Profit_EUR"],
            marker="o",
            linewidth=2,
        )
        for _, row in data.iterrows():
            ax.annotate(
                f"{row['Beta']:g}",
                (row["Out_Of_Sample_CVaR_EUR"], row["Out_Of_Sample_Expected_Profit_EUR"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )

        ax.set_title(scheme)
        ax.set_xlabel("Out-of-sample CVaR of profit (EUR)")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Out-of-sample expected profit (EUR)")
    fig.suptitle(f"Risk-averse frontier: expected profit vs CVaR (alpha={CVAR_ALPHA:.2f})")

    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    # 1) load ready wind and DA price scenarios
    wind_scenarios, da_price_scenarios = load_ready_data()

    # 2) generate 5 imbalance scenarios
    imbalance_scenarios = generate_imbalance_scenarios(
        n_imb=N_IMB,
        n_hours=N_HOURS,
        p_deficit=P_DEFICIT,
        seed=RANDOM_SEED
    )

    # 3) generate balancing price scenarios from DA price + SI
    bp_scenarios = generate_balancing_prices(da_price_scenarios, imbalance_scenarios)

    # 4) build 25 * 25 * 5 = 3125 joint scenarios
    joint_scenarios = build_joint_scenarios(
        wind_scenarios=wind_scenarios,
        da_price_scenarios=da_price_scenarios,
        imbalance_scenarios=imbalance_scenarios,
        bp_scenarios=bp_scenarios
    )

    print(f"Number of joint scenarios = {len(joint_scenarios)}")

    # 5) run ex-post 8-fold cross-validation with 200 in-sample and
    #    1400 out-of-sample scenarios per run.
    cv_results, cv_summary, cv_offers, cv_groups = run_ex_post_cross_validation(
        joint_scenarios=joint_scenarios
    )
    sensitivity_results, sensitivity_summary = run_in_sample_size_sensitivity(
        joint_scenarios=joint_scenarios
    )
    (
        risk_averse_frontier,
        risk_averse_offers,
        risk_averse_fold_robustness,
        risk_averse_plot_path,
    ) = run_risk_averse_cvar_analysis(
        joint_scenarios=joint_scenarios
    )

    # 6) save outputs
    cv_results.to_csv("ex_post_cv_results.csv", index=False)
    cv_summary.to_csv("ex_post_cv_summary.csv", index=False)
    cv_offers.to_csv("ex_post_cv_offers.csv", index=False)
    cv_groups.to_csv("ex_post_cv_scenario_groups.csv", index=False)
    sensitivity_results.to_csv("in_sample_size_sensitivity_results.csv", index=False)
    sensitivity_summary.to_csv("in_sample_size_sensitivity_summary.csv", index=False)
    risk_averse_frontier.to_csv("risk_averse_cvar_frontier.csv", index=False)
    risk_averse_offers.to_csv("risk_averse_cvar_offers.csv", index=False)
    risk_averse_fold_robustness.to_csv("risk_averse_cvar_fold_robustness.csv", index=False)

    print("\nSaved files:")
    print("  - ex_post_cv_results.csv")
    print("  - ex_post_cv_summary.csv")
    print("  - ex_post_cv_offers.csv")
    print("  - ex_post_cv_scenario_groups.csv")
    print("  - in_sample_size_sensitivity_results.csv")
    print("  - in_sample_size_sensitivity_summary.csv")
    print("  - risk_averse_cvar_frontier.csv")
    print("  - risk_averse_cvar_offers.csv")
    print("  - risk_averse_cvar_fold_robustness.csv")
    print(f"  - {risk_averse_plot_path}")


if __name__ == "__main__":
    main()
