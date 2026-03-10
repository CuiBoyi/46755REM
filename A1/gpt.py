# %%
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB


# %%
# ==========================================
# 1. Read data and set parameters
# ==========================================
year = 2030
url = f"https://raw.githubusercontent.com/PyPSA/technology-data/v0.11.0/outputs/costs_{year}.csv"
costs = pd.read_csv(url, index_col=[0, 1])

costs.loc[costs.unit.str.contains("/kW"), "value"] *= 1e3
costs.unit = costs.unit.str.replace("/kW", "/MW", regex=False)

defaults = {
    "FOM": 0,
    "VOM": 0,
    "efficiency": 1,
    "fuel": 0,
    "investment": 0,
    "lifetime": 25,
    "discount rate": 0.07,
}
costs = costs.value.unstack().fillna(defaults)
costs.at["OCGT", "fuel"] = costs.at["gas", "fuel"]
costs["marginal_cost"] = costs["VOM"] + costs["fuel"] / costs["efficiency"]

# Read one-hour demand factor (12:00)
data_day1 = pd.read_csv("data_day1.csv")
demand_factor = data_day1.loc[12, "electricity"]

# Generator data: marginal cost [€/MWh], capacity [MW]
mc_gen = {
    "Wind": 0.0,
    "Solar": 0.0,
    "OCGT": float(costs.at["OCGT", "marginal_cost"]),
}
cap_gen = {
    "Wind": 300.0,
    "Solar": 200.0,
    "OCGT": 600.0,
}

# Demand data: bid price [€/MWh], max demand [MW]
bid_dem = {
    "D1": 150.0,
    "D2": 120.0,
}
max_dem = {
    "D1": 400.0 * demand_factor,
    "D2": 500.0 * demand_factor,
}


# %%
# ==========================================
# 2. Build Gurobi model
# ==========================================
m = gp.Model("Market_Clearing_Step1")
m.setParam("OutputFlag", 0)

# Variables
g = {i: m.addVar(lb=0, name=f"g_{i}") for i in mc_gen}
d = {j: m.addVar(lb=0, name=f"d_{j}") for j in bid_dem}

# Objective: maximize social welfare
social_welfare = (
    gp.quicksum(bid_dem[j] * d[j] for j in bid_dem)
    - gp.quicksum(mc_gen[i] * g[i] for i in mc_gen)
)
m.setObjective(social_welfare, GRB.MAXIMIZE)

# Power balance
# With this sign convention, Pi of this constraint is the MCP
power_balance = m.addConstr(
    gp.quicksum(d[j] for j in bid_dem) - gp.quicksum(g[i] for i in mc_gen) == 0,
    name="Power_Balance"
)

# Upper-bound constraints
cap_g_constr = {
    i: m.addConstr(g[i] <= cap_gen[i], name=f"Cap_g_{i}")
    for i in mc_gen
}
max_d_constr = {
    j: m.addConstr(d[j] <= max_dem[j], name=f"Max_d_{j}")
    for j in bid_dem
}

m.optimize()


# %%
# ==========================================
# 3. Post-processing and reporting
# ==========================================
def classify_dispatch(x, xmax, tol=1e-6):
    if x <= tol:
        return "Rejected"
    elif x >= xmax - tol:
        return "Accepted (Full)"
    else:
        return "Marginal"


if m.status == GRB.OPTIMAL:
    MCP = power_balance.Pi

    # --- Economic results ---
    dispatch = {i: g[i].X for i in mc_gen}
    consumption = {j: d[j].X for j in bid_dem}

    total_operating_cost = sum(mc_gen[i] * dispatch[i] for i in mc_gen)
    total_welfare = m.ObjVal

    producer_profit = {
        i: (MCP - mc_gen[i]) * dispatch[i]
        for i in mc_gen
    }

    demand_utility = {
        j: consumption[j] * (bid_dem[j] - MCP)
        for j in bid_dem
    }

    total_generation = sum(dispatch.values())
    total_demand = sum(consumption.values())

    # ==========================================
    # MAIN RESULTS
    # ==========================================
    print("=" * 70)
    print("MARKET CLEARING OUTCOMES")
    print("=" * 70)
    print(f"Market-Clearing Price (MCP): {MCP:.2f} €/MWh")
    print(f"Total Generation:            {total_generation:.2f} MW")
    print(f"Total Demand:                {total_demand:.2f} MW")
    print(f"Total Operating Cost:        {total_operating_cost:.2f} €")
    print(f"Total Social Welfare:        {total_welfare:.2f} €")
    print()

    # ==========================================
    # PRODUCER RESULTS
    # ==========================================
    print("=" * 70)
    print("PRODUCER RESULTS")
    print("=" * 70)
    for i in mc_gen:
        status = classify_dispatch(dispatch[i], cap_gen[i])
        revenue = MCP * dispatch[i]
        cost = mc_gen[i] * dispatch[i]
        profit = producer_profit[i]

        print(f"{i:>8s} | Status: {status:<15s} | Dispatch: {dispatch[i]:8.2f} / {cap_gen[i]:8.2f} MW")
        print(f"         Revenue = {revenue:10.2f} € | Cost = {cost:10.2f} € | Profit = {profit:10.2f} €")
    print()

    # ==========================================
    # DEMAND RESULTS
    # ==========================================
    print("=" * 70)
    print("DEMAND RESULTS")
    print("=" * 70)
    for j in bid_dem:
        status = classify_dispatch(consumption[j], max_dem[j])
        payment = MCP * consumption[j]
        gross_value = bid_dem[j] * consumption[j]
        utility = demand_utility[j]

        print(f"{j:>8s} | Status: {status:<15s} | Consumption: {consumption[j]:8.2f} / {max_dem[j]:8.2f} MW")
        print(f"         Value = {gross_value:10.2f} € | Payment = {payment:10.2f} € | Utility = {utility:10.2f} €")
    print()

    # ==========================================
    # KKT VERIFICATION
    # ==========================================
    print("=" * 70)
    print("KKT CONDITIONS VERIFICATION")
    print("=" * 70)

    # ---------- Producers ----------
    print("--- PRODUCERS (Generators) ---")
    for i in mc_gen:
        mc = mc_gen[i]
        p = dispatch[i]
        pmax = cap_gen[i]

        mu_max = cap_g_constr[i].Pi
        mu_min = max(0.0, mu_max + mc - MCP)

        status = classify_dispatch(p, pmax)

        stationarity_ok = abs(MCP - (mc + mu_max - mu_min)) < 1e-5
        cs_upper = mu_max * (pmax - p)
        cs_lower = mu_min * p

        if status == "Accepted (Full)":
            interpretation = "Accepted at upper bound: MCP can exceed MC because the capacity constraint is binding."
        elif status == "Marginal":
            interpretation = "Marginal unit: mu_max = mu_min = 0, so MCP = MC."
        else:
            interpretation = "Rejected unit: p = 0, so MCP is below its bid/MC."

        print(f"\nGenerator: {i} [{status}]")
        print(f"  Bid Price (MC): {mc:.2f} €/MWh | Dispatch: {p:.2f} MW / {pmax:.2f} MW")
        print(f"  Dual Variables: mu_max = {mu_max:.2f}, mu_min = {mu_min:.2f}")
        print(f"  Stationarity: MCP ({MCP:.2f}) == MC ({mc:.2f}) + mu_max ({mu_max:.2f}) - mu_min ({mu_min:.2f}) -> {stationarity_ok}")
        print(f"  Complementary Slackness:")
        print(f"    mu_max * (P_max - P) = {mu_max:.2f} * {pmax - p:.2f} = {cs_upper:.6f}")
        print(f"    mu_min * P           = {mu_min:.2f} * {p:.2f} = {cs_lower:.6f}")
        print(f"  Interpretation: {interpretation}")

    # ---------- Consumers ----------
    print("\n--- CONSUMERS (Demands) ---")
    for j in bid_dem:
        bid = bid_dem[j]
        q = consumption[j]
        qmax = max_dem[j]

        nu_max = max_d_constr[j].Pi
        nu_min = max(0.0, nu_max + MCP - bid)

        status = classify_dispatch(q, qmax)

        stationarity_ok = abs(bid - MCP - nu_max + nu_min) < 1e-5
        cs_upper = nu_max * (qmax - q)
        cs_lower = nu_min * q

        if status == "Accepted (Full)":
            interpretation = "Accepted at upper bound: bid is above MCP, and the maximum demand limit is binding."
        elif status == "Marginal":
            interpretation = "Marginal demand: nu_max = nu_min = 0, so bid = MCP."
        else:
            interpretation = "Rejected demand: q = 0, so bid is below MCP."

        print(f"\nDemand: {j} [{status}]")
        print(f"  Bid Price: {bid:.2f} €/MWh | Consumption: {q:.2f} MW / {qmax:.2f} MW")
        print(f"  Dual Variables: nu_max = {nu_max:.2f}, nu_min = {nu_min:.2f}")
        print(f"  Stationarity: Bid ({bid:.2f}) - MCP ({MCP:.2f}) == nu_max ({nu_max:.2f}) - nu_min ({nu_min:.2f}) -> {stationarity_ok}")
        print(f"  Complementary Slackness:")
        print(f"    nu_max * (D_max - D) = {nu_max:.2f} * {qmax - q:.2f} = {cs_upper:.6f}")
        print(f"    nu_min * D           = {nu_min:.2f} * {q:.2f} = {cs_lower:.6f}")
        print(f"  Interpretation: {interpretation}")

else:
    print("Optimization did not reach an optimal solution.")