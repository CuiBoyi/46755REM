import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from pathlib import Path
from time import perf_counter


# ============================================================
# USER INPUTS
# ============================================================

CAPACITY_MW = 500.0
N_HOURS = 24

N_WIND = 25
N_PRICE = 25
N_IMB = 5
IN_SAMPLE_SIZE = 125

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

# If you already have scenario probabilities, replace these later
USE_EQUAL_PROBABILITIES = True


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


def build_joint_probabilities(n_wind=N_WIND, n_price=N_PRICE, n_imb=N_IMB):
    """
    Equal probabilities by default.
    If you have non-uniform probabilities, replace this.
    """
    n_total = n_wind * n_price * n_imb
    probs = np.full(n_total, 1.0 / n_total)
    return probs


def build_equal_probabilities(n_scenarios):
    """Equal probabilities over the supplied scenario group."""
    return np.full(n_scenarios, 1.0 / n_scenarios)


def split_in_sample_out_of_sample(joint_scenarios, in_sample_size=IN_SAMPLE_SIZE, seed=RANDOM_SEED):
    """
    Randomly split joint scenarios into an in-sample optimization group and an
    out-of-sample evaluation group.
    """
    n_total = len(joint_scenarios)
    if not 0 < in_sample_size < n_total:
        raise ValueError(
            f"in_sample_size must be between 1 and {n_total - 1}; got {in_sample_size}."
        )

    rng = np.random.default_rng(seed)
    in_sample_indices = np.sort(rng.choice(n_total, size=in_sample_size, replace=False))
    in_sample_set = set(in_sample_indices)
    out_of_sample_indices = np.array(
        [idx for idx in range(n_total) if idx not in in_sample_set],
        dtype=int
    )

    in_sample = [joint_scenarios[idx] for idx in in_sample_indices]
    out_of_sample = [joint_scenarios[idx] for idx in out_of_sample_indices]

    return in_sample, out_of_sample, in_sample_indices, out_of_sample_indices


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
    m.Params.OutputFlag = 1

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

    # Scenario-wise profits
    scenario_profits = np.zeros(n_scen)
    for s in range(n_scen):
        W = joint_scenarios[s]["wind"]
        DA = joint_scenarios[s]["da"]
        BP = joint_scenarios[s]["bp"]

        profit = 0.0
        for t in range(n_hours):
            profit += DA[t] * q_star[t] + BP[t] * (W[t] - q_star[t])

        scenario_profits[s] = profit

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
    n_scen = len(joint_scenarios)
    scenario_profits = np.zeros(n_scen)

    for s in range(n_scen):
        W = joint_scenarios[s]["wind"]
        DA = joint_scenarios[s]["da"]
        BP = joint_scenarios[s]["bp"]

        scenario_profits[s] = np.sum(DA * offer_mw + BP * (W - offer_mw))

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

    # Scenario-wise profits
    scenario_profits = np.zeros(n_scen)

    for s in range(n_scen):
        DA = joint_scenarios[s]["da"]
        BP = joint_scenarios[s]["bp"]
        SI = joint_scenarios[s]["si"]
        W = joint_scenarios[s]["wind"]

        profit = 0.0
        for t in range(n_hours):
            delta = W[t] - q_star[t]
            over_val = max(delta, 0.0)
            under_val = max(-delta, 0.0)

            profit += DA[t] * q_star[t]

            if SI[t] == 1:
                profit += DA[t] * over_val
                profit -= BP[t] * under_val
            else:
                profit += BP[t] * over_val
                profit -= DA[t] * under_val

        scenario_profits[s] = profit

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
    n_scen = len(joint_scenarios)
    scenario_profits = np.zeros(n_scen)

    for s in range(n_scen):
        DA = joint_scenarios[s]["da"]
        BP = joint_scenarios[s]["bp"]
        SI = joint_scenarios[s]["si"]
        W = joint_scenarios[s]["wind"]

        profit = 0.0
        for t in range(len(offer_mw)):
            delta = W[t] - offer_mw[t]
            over_val = max(delta, 0.0)
            under_val = max(-delta, 0.0)

            profit += DA[t] * offer_mw[t]

            if SI[t] == 1:
                profit += DA[t] * over_val
                profit -= BP[t] * under_val
            else:
                profit += BP[t] * over_val
                profit -= DA[t] * under_val

        scenario_profits[s] = profit

    expected_profit = np.dot(scenario_probabilities, scenario_profits)

    return {
        "model": None,
        "offer_mw": offer_mw,
        "scenario_profits": scenario_profits,
        "expected_profit": expected_profit
    }


# ============================================================
# STEP 5. SUMMARIZE RESULTS
# ============================================================

def summarize_results(name, result, offer_label="Hourly day-ahead offers"):
    q = result["offer_mw"]
    profits = result["scenario_profits"]

    summary = pd.DataFrame({
        "Hour": np.arange(1, len(q) + 1),
        "DA_Offer_MW": q
    })

    print("\n" + "=" * 70)
    print(f"{name} RESULTS")
    print("=" * 70)
    print(f"Expected profit: {result['expected_profit']:.2f} EUR")
    if "solve_time_seconds" in result:
        print(f"Solve time:      {result['solve_time_seconds']:.4f} seconds")
    print(f"Scenario profit mean: {profits.mean():.2f} EUR")
    print(f"Scenario profit std:  {profits.std(ddof=0):.2f} EUR")
    print(f"Scenario profit min:  {profits.min():.2f} EUR")
    print(f"Scenario profit max:  {profits.max():.2f} EUR")
    print(f"\n{offer_label}:")
    print(summary.to_string(index=False))

    return summary


def build_scenario_group_table(in_sample_indices, out_of_sample_indices, joint_scenarios):
    rows = []

    for idx in in_sample_indices:
        scenario = joint_scenarios[idx]
        rows.append({
            "scenario_id": idx,
            "group": "in_sample",
            "w_idx": scenario["w_idx"],
            "p_idx": scenario["p_idx"],
            "k_idx": scenario["k_idx"],
        })

    for idx in out_of_sample_indices:
        scenario = joint_scenarios[idx]
        rows.append({
            "scenario_id": idx,
            "group": "out_of_sample",
            "w_idx": scenario["w_idx"],
            "p_idx": scenario["p_idx"],
            "k_idx": scenario["k_idx"],
        })

    return pd.DataFrame(rows).sort_values("scenario_id")


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

    # 5) split scenarios into in-sample and out-of-sample groups
    (
        in_sample_scenarios,
        out_of_sample_scenarios,
        in_sample_indices,
        out_of_sample_indices,
    ) = split_in_sample_out_of_sample(
        joint_scenarios=joint_scenarios,
        in_sample_size=IN_SAMPLE_SIZE,
        seed=RANDOM_SEED
    )

    print(f"In-sample scenarios = {len(in_sample_scenarios)}")
    print(f"Out-of-sample scenarios = {len(out_of_sample_scenarios)}")

    # 6) probabilities within each scenario group
    in_sample_probabilities = build_equal_probabilities(len(in_sample_scenarios))
    out_of_sample_probabilities = build_equal_probabilities(len(out_of_sample_scenarios))

    # 7) solve one-price and two-price schemes on the in-sample group
    one_price_in_sample_result = solve_one_price(
        joint_scenarios=in_sample_scenarios,
        scenario_probabilities=in_sample_probabilities,
        capacity_mw=CAPACITY_MW
    )

    two_price_in_sample_result = solve_two_price(
        joint_scenarios=in_sample_scenarios,
        scenario_probabilities=in_sample_probabilities,
        capacity_mw=CAPACITY_MW
    )

    # 8) evaluate the in-sample offers on the out-of-sample group
    one_price_out_of_sample_result = evaluate_one_price(
        joint_scenarios=out_of_sample_scenarios,
        scenario_probabilities=out_of_sample_probabilities,
        offer_mw=one_price_in_sample_result["offer_mw"]
    )

    two_price_out_of_sample_result = evaluate_two_price(
        joint_scenarios=out_of_sample_scenarios,
        scenario_probabilities=out_of_sample_probabilities,
        offer_mw=two_price_in_sample_result["offer_mw"]
    )

    # 9) summarize
    one_price_summary = summarize_results(
        "ONE-PRICE IN-SAMPLE",
        one_price_in_sample_result,
        offer_label="Optimal hourly day-ahead offers from in-sample solve"
    )
    summarize_results(
        "ONE-PRICE OUT-OF-SAMPLE",
        one_price_out_of_sample_result,
        offer_label="In-sample offers evaluated out-of-sample"
    )
    two_price_summary = summarize_results(
        "TWO-PRICE IN-SAMPLE",
        two_price_in_sample_result,
        offer_label="Optimal hourly day-ahead offers from in-sample solve"
    )
    summarize_results(
        "TWO-PRICE OUT-OF-SAMPLE",
        two_price_out_of_sample_result,
        offer_label="In-sample offers evaluated out-of-sample"
    )

    # 10) save outputs
    one_price_summary.to_csv("one_price_optimal_offer.csv", index=False)
    two_price_summary.to_csv("two_price_optimal_offer.csv", index=False)

    scenario_group_table = build_scenario_group_table(
        in_sample_indices=in_sample_indices,
        out_of_sample_indices=out_of_sample_indices,
        joint_scenarios=joint_scenarios
    )
    scenario_group_table.to_csv("scenario_groups.csv", index=False)

    pd.DataFrame({
        "Scheme": ["One-price", "One-price", "Two-price", "Two-price"],
        "Group": ["In-sample", "Out-of-sample", "In-sample", "Out-of-sample"],
        "N_Scenarios": [
            len(in_sample_scenarios),
            len(out_of_sample_scenarios),
            len(in_sample_scenarios),
            len(out_of_sample_scenarios),
        ],
        "Expected_Profit_EUR": [
            one_price_in_sample_result["expected_profit"],
            one_price_out_of_sample_result["expected_profit"],
            two_price_in_sample_result["expected_profit"],
            two_price_out_of_sample_result["expected_profit"],
        ]
    }).to_csv("expected_profit_comparison.csv", index=False)

    print("\nSaved files:")
    print("  - one_price_optimal_offer.csv")
    print("  - two_price_optimal_offer.csv")
    print("  - scenario_groups.csv")
    print("  - expected_profit_comparison.csv")


if __name__ == "__main__":
    main()
