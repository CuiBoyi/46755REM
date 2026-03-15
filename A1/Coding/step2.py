
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from pathlib import Path
from plot import plot_storage_operation_compact, plot_storage_price_and_profit, plot_cost_profit_contributions, plot_mcp_comparison, plot_system_comparison, print_assignment_results, print_sensitivity_results, plot_storage_hourly
from utili import load_wind_scen

# ============================================================
# 1. Small helper container
# ============================================================
class Expando:
    pass


# ============================================================
# 2. LP input data container (array-based)
# ============================================================
class LPInputData:
    def __init__(
        self,
        var_names: list[str],
        constr_names: list[str],
        obj: np.ndarray,
        A: np.ndarray,
        rhs: np.ndarray,
        sense: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        objective_sense: int,
        model_name: str,
    ):
        self.var_names = var_names
        self.constr_names = constr_names
        self.obj = np.asarray(obj, dtype=float)
        self.A = np.asarray(A, dtype=float)
        self.rhs = np.asarray(rhs, dtype=float)
        self.sense = np.asarray(sense, dtype=object)
        self.lb = np.asarray(lb, dtype=float)
        self.ub = np.asarray(ub, dtype=float)
        self.objective_sense = objective_sense
        self.model_name = model_name


# ============================================================
# 3. LP optimization wrapper
# ============================================================
class LPOptimizationProblem:
    def __init__(self, input_data: LPInputData):
        self.data = input_data
        self.results = Expando()
        self._build_model()

    def _build_variables(self):
        self.variables = []
        for i, name in enumerate(self.data.var_names):
            var = self.model.addVar(
                lb=self.data.lb[i],
                ub=self.data.ub[i],
                name=name
            )
            self.variables.append(var)

    def _build_objective(self):
        obj_expr = gp.quicksum(
            self.data.obj[j] * self.variables[j]
            for j in range(len(self.variables))
        )
        self.model.setObjective(obj_expr, self.data.objective_sense)

    def _build_constraints(self):
        self.constraints = []
        n_constr = len(self.data.constr_names)
        n_var = len(self.data.var_names)

        for i in range(n_constr):
            expr = gp.quicksum(
                self.data.A[i, j] * self.variables[j]
                for j in range(n_var)
                if abs(self.data.A[i, j]) > 1e-12
            )
            constr = self.model.addLConstr(
                expr,
                self.data.sense[i],
                self.data.rhs[i],
                name=self.data.constr_names[i],
            )
            self.constraints.append(constr)

    def _build_model(self):
        self.model = gp.Model(name=self.data.model_name)
        self._build_variables()
        self._build_objective()
        self._build_constraints()
        self.model.update()

    def run(self):
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            self._save_results()
        else:
            print(f"Optimization failed. Status = {self.model.status}")

    def _save_results(self):
        self.results.objective_value = self.model.ObjVal
        self.results.x = np.array([v.X for v in self.model.getVars()], dtype=float)
        self.results.var_names = [v.VarName for v in self.model.getVars()]
        self.results.duals = np.array([c.Pi for c in self.model.getConstrs()], dtype=float)
        self.results.constr_names = [c.ConstrName for c in self.model.getConstrs()]

    def display_results(self, max_rows: int = 20):
        print("\n------------------- RESULTS -------------------")
        print(f"Optimal objective value: {self.results.objective_value:.4f}")

        print("\n--- First variables ---")
        for name, val in list(zip(self.results.var_names, self.results.x))[:max_rows]:
            print(f"{name}: {val:.4f}")

        print("\n--- First duals ---")
        for name, val in list(zip(self.results.constr_names, self.results.duals))[:max_rows]:
            print(f"{name}: {val:.4f}")


# ============================================================
# 4. Core market data (array-based)
# ============================================================
def get_base_market_data():
    thermal_names = [f"g{i}" for i in range(1, 13)]
    wind_names = [f"w{i}" for i in range(1, 7)]
    load_names = [f"d{i}" for i in range(1, 18)]

    Pmax = np.array([152, 152, 350, 591, 60, 155, 155, 400, 400, 300, 310, 350], dtype=float)
    Pmin = np.zeros(12, dtype=float)
    #Pmin = np.array([30.4, 30.4, 75, 206.85, 12, 54.25, 54.25, 100, 100, 300, 108.5, 140], dtype=float)

    gen_cost = np.array([13.32, 13.32, 20.70, 20.93, 26.11, 10.52, 10.52, 6.02, 5.47, 0.00, 10.52, 10.89], dtype=float)

    Wcap = np.array([157.92, 162.83, 154.34, 126.48, 142.32, 147.68], dtype=float)
    wind_cost = np.zeros(6, dtype=float)

    load_share_percent = np.array(
        [3.8, 3.4, 6.3, 2.6, 2.5, 4.8, 4.4, 6.0, 6.1, 6.8, 9.3, 6.8, 11.1, 3.5, 11.7, 6.4, 4.5],
        dtype=float,
    )

    base_load_bid = np.array(
        [24.445, 24.923, 21.456, 25.880, 26.000, 23.250, 23.7282, 21.815, 21.695,
         20.8586, 17.869, 20.858, 15.717, 24.804, 15.000, 21.3369, 23.608],
        dtype=float,
    )

    system_demand = np.array([
        1775.835, 1669.815, 1590.300, 1563.795, 1563.795, 1590.300,
        1961.370, 2279.430, 2517.975, 2544.480, 2544.480, 2517.975,
        2517.975, 2517.975, 2464.965, 2464.965, 2623.995, 2650.500,
        2650.500, 2544.480, 2411.955, 2199.915, 1934.865, 1669.815
    ], dtype=float)

    return {
        "thermal_names": thermal_names,
        "wind_names": wind_names,
        "load_names": load_names,
        "Pmax": Pmax,
        "Pmin": Pmin,
        "gen_cost": gen_cost,
        "Wcap": Wcap,
        "wind_cost": wind_cost,
        "load_share_percent": load_share_percent,
        "base_load_bid": base_load_bid,
        "system_demand": system_demand,
    }


# ============================================================
# 5. Build 24-hour load bids
#    Peak hours get higher bids
# ============================================================
def build_hourly_load_bids(system_demand: np.ndarray, base_load_bid: np.ndarray) -> np.ndarray:
    """
    Returns shape (24, 17).
    Higher total demand -> higher bid multiplier.
    """
    dmin = system_demand.min()
    dmax = system_demand.max()

    if abs(dmax - dmin) < 1e-12:
        multiplier = np.ones_like(system_demand)
    else:
        # e.g. multiplier from 0.90 to 1.15 across the day
        normalized = (system_demand - dmin) / (dmax - dmin)
        multiplier = 0.90 + 0.25 * normalized

    hourly_bids = multiplier[:, None] * base_load_bid[None, :]
    return hourly_bids


# ============================================================
# 6. Wind availability builder
#    If the user does not provide hourly wind factors,
#    create a simple default profile.
# ============================================================
def build_default_wind_profile(T: int = 24, n_wind: int = 6) -> np.ndarray:
    """
    Returns availability factors in [0,1], shape (24, 6).
    """
    hours = np.arange(T)
    profile = 0.55 + 0.20 * np.sin(2 * np.pi * (hours - 5) / 24.0)
    profile = np.clip(profile, 0.15, 0.95)

    wind_factor = np.tile(profile[:, None], (1, n_wind))

    # small zone differences
    zone_scale = np.array([1.00, 1.03, 0.98, 0.93, 0.97, 1.01], dtype=float)
    wind_factor *= zone_scale[None, :]
    wind_factor = np.clip(wind_factor, 0.0, 1.0)

    return wind_factor



def load_wind_factor_from_mat(scenario: int = 30) -> np.ndarray:
    """
    Load 24x6 wind availability factors from WindScen.mat.

    Returns
    -------
    wind_factor : np.ndarray
        Shape = (24, 6)
        Row t = hour t+1
        Column i = wind zone i+1
    """
    BASE_DIR = Path(__file__).resolve().parent
    candidates = [
        BASE_DIR / "dataset" / "WindScen.mat",
        BASE_DIR.parent / "dataset" / "WindScen.mat",
    ]

    path_wind = next((p for p in candidates if p.exists()), None)
    if path_wind is None:
        raise FileNotFoundError(
            f"Could not find WindScen.mat. Checked: {[str(p) for p in candidates]}"
        )

    wind = load_wind_scen(path_wind)

    # Build array of shape (24, 6)
    wind_factor = np.zeros((24, 6), dtype=float)

    for i in range(6):          # 6 wind zones
        for t in range(24):     # 24 hours
            wind_factor[t, i] = wind[i][t, scenario]

    return wind_factor

# ============================================================
# 7. Build Step 2 LP input (24h, with optional storage)
# ============================================================
def build_input_data_step2(
    wind_factor: np.ndarray | None = None,
    include_storage: bool = True,
    Pch_max: float = 300.0,
    Pdis_max: float = 300.0,
    E_max: float = 1200.0,
    eta_ch: float = 0.90,
    eta_dis: float = 0.95,
    e0: float = 600.0,
    cyclic_storage: bool = True,
) -> tuple[LPInputData, dict]:
    """
    Step 2:
    - 24 hours
    - copper plate
    - optional storage
    - array-based data
    """

    data = get_base_market_data()

    thermal_names = data["thermal_names"]
    wind_names = data["wind_names"]
    load_names = data["load_names"]

    Pmax = data["Pmax"]
    Pmin = data["Pmin"]
    gen_cost = data["gen_cost"]
    Wcap = data["Wcap"]
    wind_cost = data["wind_cost"]
    load_share_percent = data["load_share_percent"]
    base_load_bid = data["base_load_bid"]
    system_demand = data["system_demand"]

    T = 24
    ng = len(thermal_names)
    nw = len(wind_names)
    nd = len(load_names)

    if wind_factor is None:
        wind_factor = build_default_wind_profile(T=T, n_wind=nw)
    wind_factor = np.asarray(wind_factor, dtype=float)

    if wind_factor.shape != (T, nw):
        raise ValueError(f"wind_factor must have shape {(T, nw)}, got {wind_factor.shape}")

    Wmax_hourly = wind_factor * Wcap[None, :]
    Dmax_hourly = system_demand[:, None] * load_share_percent[None, :] / 100.0
    load_bid_hourly = build_hourly_load_bids(system_demand, base_load_bid)

    # --------------------------------------------------------
    # Variable indexing
    # --------------------------------------------------------
    var_names = []
    idx_g = np.empty((T, ng), dtype=int)
    idx_w = np.empty((T, nw), dtype=int)
    idx_d = np.empty((T, nd), dtype=int)

    counter = 0

    for t in range(T):
        for i in range(ng):
            idx_g[t, i] = counter
            var_names.append(f"g_{i+1}_t{t+1}")
            counter += 1

        for i in range(nw):
            idx_w[t, i] = counter
            var_names.append(f"w_{i+1}_t{t+1}")
            counter += 1

        for i in range(nd):
            idx_d[t, i] = counter
            var_names.append(f"d_{i+1}_t{t+1}")
            counter += 1

    if include_storage:
        idx_pch = np.empty(T, dtype=int)
        idx_pdis = np.empty(T, dtype=int)
        idx_e = np.empty(T, dtype=int)

        for t in range(T):
            idx_pch[t] = counter
            var_names.append(f"pch_t{t+1}")
            counter += 1

            idx_pdis[t] = counter
            var_names.append(f"pdis_t{t+1}")
            counter += 1

            idx_e[t] = counter
            var_names.append(f"e_t{t+1}")
            counter += 1
    else:
        idx_pch = None
        idx_pdis = None
        idx_e = None

    nv = counter

    # --------------------------------------------------------
    # Bounds
    # --------------------------------------------------------
    lb = np.zeros(nv, dtype=float)
    ub = np.full(nv, np.inf, dtype=float)

    for t in range(T):
        ub[idx_g[t, :]] = Pmax
        lb[idx_g[t, :]] = Pmin

        ub[idx_w[t, :]] = Wmax_hourly[t, :]
        ub[idx_d[t, :]] = Dmax_hourly[t, :]

    if include_storage:
        ub[idx_pch] = Pch_max
        ub[idx_pdis] = Pdis_max
        ub[idx_e] = E_max

    # --------------------------------------------------------
    # Objective
    # maximize sum(load value - generation cost)
    # storage bids/offers are zero
    # --------------------------------------------------------
    obj = np.zeros(nv, dtype=float)

    for t in range(T):
        obj[idx_g[t, :]] = -gen_cost
        obj[idx_w[t, :]] = -wind_cost
        obj[idx_d[t, :]] = load_bid_hourly[t, :]

        if include_storage:
            obj[idx_pch[t]] = 0.0
            obj[idx_pdis[t]] = 0.0
            obj[idx_e[t]] = 0.0

    # --------------------------------------------------------
    # Constraints
    # 1) balance for each hour
    # 2) storage dynamics
    # 3) optional cyclic ending condition
    # --------------------------------------------------------
    constr_names = []

    # balance equations
    for t in range(T):
        constr_names.append(f"balance_t{t+1}")

    if include_storage:
        for t in range(T):
            constr_names.append(f"storage_dyn_t{t+1}")

        if cyclic_storage:
            constr_names.append("storage_terminal_cycle")

    nc = len(constr_names)

    A = np.zeros((nc, nv), dtype=float)
    rhs = np.zeros(nc, dtype=float)
    sense = np.empty(nc, dtype=object)

    row = 0

    # --------------------------------------------------------
    # Hourly balance
    # sum(g) + sum(w) + pdis - pch - sum(d) = 0
    # --------------------------------------------------------
    for t in range(T):
        A[row, idx_g[t, :]] = 1.0
        A[row, idx_w[t, :]] = 1.0
        A[row, idx_d[t, :]] = -1.0

        if include_storage:
            A[row, idx_pdis[t]] = 1.0
            A[row, idx_pch[t]] = -1.0

        rhs[row] = 0.0
        sense[row] = GRB.EQUAL
        row += 1

    # --------------------------------------------------------
    # Storage dynamics
    # e_t = e_{t-1} + eta_ch * pch_t - pdis_t / eta_dis
    # For t=1: e_1 = e0 + ...
    # --------------------------------------------------------
    if include_storage:
        for t in range(T):
            A[row, idx_e[t]] = 1.0
            A[row, idx_pch[t]] = -eta_ch
            A[row, idx_pdis[t]] = 1.0 / eta_dis

            if t == 0:
                rhs[row] = e0
            else:
                A[row, idx_e[t - 1]] = -1.0
                rhs[row] = 0.0

            sense[row] = GRB.EQUAL
            row += 1

        if cyclic_storage:
            # e_T = e0
            A[row, idx_e[T - 1]] = 1.0
            rhs[row] = e0
            sense[row] = GRB.EQUAL
            row += 1

    input_data = LPInputData(
        var_names=var_names,
        constr_names=constr_names,
        obj=obj,
        A=A,
        rhs=rhs,
        sense=sense,
        lb=lb,
        ub=ub,
        objective_sense=GRB.MAXIMIZE,
        model_name="market_clearing_step2_multi_hour",
    )

    data_pack = {
        "T": T,
        "ng": ng,
        "nw": nw,
        "nd": nd,
        "thermal_names": thermal_names,
        "wind_names": wind_names,
        "load_names": load_names,
        "Pmax": Pmax,
        "Pmin": Pmin,
        "gen_cost": gen_cost,
        "Wcap": Wcap,
        "wind_cost": wind_cost,
        "system_demand": system_demand,
        "load_share_percent": load_share_percent,
        "base_load_bid": base_load_bid,
        "load_bid_hourly": load_bid_hourly,
        "Wmax_hourly": Wmax_hourly,
        "Dmax_hourly": Dmax_hourly,
        "include_storage": include_storage,
        "storage_params": {
            "Pch_max": Pch_max,
            "Pdis_max": Pdis_max,
            "E_max": E_max,
            "eta_ch": eta_ch,
            "eta_dis": eta_dis,
            "e0": e0,
            "cyclic_storage": cyclic_storage,
        },
        "index": {
            "g": idx_g,
            "w": idx_w,
            "d": idx_d,
            "pch": idx_pch,
            "pdis": idx_pdis,
            "e": idx_e,
        },
    }

    return input_data, data_pack


# ============================================================
# 8. Post-processing
# ============================================================
def extract_market_results(model: LPOptimizationProblem, data_pack: dict) -> dict:
    T = data_pack["T"]
    ng = data_pack["ng"]
    nw = data_pack["nw"]
    nd = data_pack["nd"]

    idx = data_pack["index"]
    gen_cost = data_pack["gen_cost"]
    load_bid_hourly = data_pack["load_bid_hourly"]

    x = model.results.x
    duals = model.results.duals
    constr_names = model.results.constr_names

    # reshape primal variables
    g = np.zeros((T, ng), dtype=float)
    w = np.zeros((T, nw), dtype=float)
    d = np.zeros((T, nd), dtype=float)

    for t in range(T):
        g[t, :] = x[idx["g"][t, :]]
        w[t, :] = x[idx["w"][t, :]]
        d[t, :] = x[idx["d"][t, :]]

    if data_pack["include_storage"]:
        pch = x[idx["pch"]]
        pdis = x[idx["pdis"]]
        e = x[idx["e"]]
    else:
        pch = np.zeros(T, dtype=float)
        pdis = np.zeros(T, dtype=float)
        e = np.zeros(T, dtype=float)

    # MCP from balance duals
    balance_duals = np.zeros(T, dtype=float)
    for t in range(T):
        cname = f"balance_t{t+1}"
        row = constr_names.index(cname)
        balance_duals[t] = duals[row]

    # Because balance was written as:
    #   sum(g)+sum(w)+pdis-pch-sum(d)=0
    # and objective is max welfare,
    # market price is negative dual
    mcp = -balance_duals

    # costs and welfare
    hourly_operating_cost = np.sum(g * gen_cost[None, :], axis=1)
    hourly_load_value = np.sum(d * load_bid_hourly, axis=1)
    hourly_social_welfare = hourly_load_value - hourly_operating_cost

    # profits
    thermal_profit_hourly = g * (mcp[:, None] - gen_cost[None, :])
    wind_profit_hourly = w * mcp[:, None]
    storage_profit_hourly = mcp * (pdis - pch)

    total_thermal_profit = np.sum(thermal_profit_hourly, axis=0)
    total_wind_profit = np.sum(wind_profit_hourly, axis=0)
    total_storage_profit = np.sum(storage_profit_hourly)

    return {
        "g": g,
        "w": w,
        "d": d,
        "pch": pch,
        "pdis": pdis,
        "e": e,
        "mcp": mcp,
        "hourly_operating_cost": hourly_operating_cost,
        "hourly_load_value": hourly_load_value,
        "hourly_social_welfare": hourly_social_welfare,
        "total_operating_cost": np.sum(hourly_operating_cost),
        "total_load_value": np.sum(hourly_load_value),
        "total_social_welfare": np.sum(hourly_social_welfare),
        "thermal_profit_hourly": thermal_profit_hourly,
        "wind_profit_hourly": wind_profit_hourly,
        "storage_profit_hourly": storage_profit_hourly,
        "total_thermal_profit": total_thermal_profit,
        "total_wind_profit": total_wind_profit,
        "total_storage_profit": total_storage_profit,
    }


# ============================================================
# 9. Reporting
# ============================================================
def print_step2_summary(result_pack: dict, data_pack: dict, title: str = "CASE"):
    thermal_names = data_pack["thermal_names"]
    wind_names = data_pack["wind_names"]

    print("\n============================================================")
    print(title)
    print("============================================================")

    print("\nHourly market-clearing prices:")
    for t, price in enumerate(result_pack["mcp"], start=1):
        print(f"Hour {t:02d}: MCP = {price:.4f} €/MWh")

    print("\n------------------------------------------------------------")
    print(f"Total Operating Cost: {result_pack['total_operating_cost']:.4f} €")
    print(f"Total Load Value:     {result_pack['total_load_value']:.4f} €")
    print(f"Total Social Welfare: {result_pack['total_social_welfare']:.4f} €")

    print("\n------------------------------------------------------------")
    print("Total producer profits over 24h")
    for i, name in enumerate(thermal_names):
        print(f"{name}: {result_pack['total_thermal_profit'][i]:.4f} €")

    for i, name in enumerate(wind_names):
        print(f"{name}: {result_pack['total_wind_profit'][i]:.4f} €")

    if data_pack["include_storage"]:
        print(f"Storage: {result_pack['total_storage_profit']:.4f} €")

        print("\n------------------------------------------------------------")
        print("Storage operation by hour")
        for t in range(data_pack["T"]):
            print(
                f"Hour {t+1:02d}: "
                f"pch = {result_pack['pch'][t]:8.4f} MW | "
                f"pdis = {result_pack['pdis'][t]:8.4f} MW | "
                f"e = {result_pack['e'][t]:8.4f} MWh | "
                f"profit = {result_pack['storage_profit_hourly'][t]:8.4f} €"
            )


def compare_cases(no_storage: dict, with_storage: dict):
    print("\n============================================================")
    print("PRICE COMPARISON: WITHOUT STORAGE vs WITH STORAGE")
    print("============================================================")
    print("Hour | MCP no storage | MCP with storage | Difference")
    for t in range(len(no_storage["mcp"])):
        diff = with_storage["mcp"][t] - no_storage["mcp"][t]
        print(f"{t+1:>4} | {no_storage['mcp'][t]:>14.4f} | {with_storage['mcp'][t]:>16.4f} | {diff:>10.4f}")

    print("\nPattern reminder:")
    print("- Storage tends to charge in low-price hours and discharge in high-price hours.")
    print("- This usually raises low prices and lowers peak prices.")
    print("- Therefore, storage often smooths the MCP profile over time.")


def run_sensitivity_cases1(wind_factor):
    cases = [
        {"name": "Base", "Pch_max": 300, "Pdis_max": 300, "E_max": 1200},
        {"name": "Storage_400", "Pch_max": 300, "Pdis_max": 300, "E_max": 400},
        {"name": "Storage_800", "Pch_max": 300, "Pdis_max": 300, "E_max": 800},
        {"name": "Storage_1600", "Pch_max": 300, "Pdis_max": 300, "E_max": 1600},
        {"name": "Storage_2000", "Pch_max": 300, "Pdis_max": 300, "E_max": 2000},
        {"name": "Storage_2400", "Pch_max": 300, "Pdis_max": 300, "E_max": 2400},
        # {"name": "Energy-limited", "Pch_max": 150, "Pdis_max": 150, "E_max": 1200},
        # {"name": "Power-limited", "Pch_max": 600, "Pdis_max": 600, "E_max": 1200},
    ]

    outputs = []

    for case in cases:
        input_data, pack = build_input_data_step2(
            wind_factor=wind_factor,
            include_storage=True,
            Pch_max=case["Pch_max"],
            Pdis_max=case["Pdis_max"],
            E_max=case["E_max"],
            eta_ch=0.90,
            eta_dis=0.95,
            e0=case["E_max"] / 2,
            cyclic_storage=True
        )
        model = LPOptimizationProblem(input_data)
        model.run()
        result = extract_market_results(model, pack)

        outputs.append({
            "name": case["name"],
            "total_operating_cost": result["total_operating_cost"],
            "total_social_welfare": result["total_social_welfare"],
            "total_storage_profit": result["total_storage_profit"],
            "avg_mcp": np.mean(result["mcp"]),
            "max_mcp": np.max(result["mcp"]),
            "min_mcp": np.min(result["mcp"]),
        })

    return outputs

def run_sensitivity_cases2(wind_factor):
    # cases = [
    #     {"name": "Storage 400MW", "Pch_max": 300, "Pdis_max": 300, "E_max": 400},
    #     {"name": "Storage 800MW", "Pch_max": 300, "Pdis_max": 300, "E_max": 800},
    #     {"name": "Storage 1200MW", "Pch_max": 300, "Pdis_max": 300, "E_max": 1200},
    #     {"name": "Storage 1600MW", "Pch_max": 300, "Pdis_max": 300, "E_max": 1600},
    #     {"name": "Storage 2000MW", "Pch_max": 300, "Pdis_max": 300, "E_max": 2000},
    #     {"name": "Storage 2400MW", "Pch_max": 300, "Pdis_max": 300, "E_max": 2400},
    # ]

    cases = [
        {"name": "hourly changing 100MW", "Pch_max": 100, "Pdis_max": 100, "E_max": 1200},
        {"name": "hourly changing 200MW", "Pch_max": 200, "Pdis_max": 200, "E_max": 1200},
        {"name": "hourly changing 300MW", "Pch_max": 300, "Pdis_max": 300, "E_max": 1200},
        {"name": "hourly changing 400MW", "Pch_max": 400, "Pdis_max": 400, "E_max": 1200},
        {"name": "hourly changing 500MW", "Pch_max": 500, "Pdis_max": 500, "E_max": 1200},
        {"name": "hourly changing 600MW", "Pch_max": 600, "Pdis_max": 600, "E_max": 1200},
        {"name": "hourly changing 700MW", "Pch_max": 700, "Pdis_max": 700, "E_max": 1200},
        {"name": "hourly changing 800MW", "Pch_max": 800, "Pdis_max": 800, "E_max": 1200},
    ]

    outputs = []

    for case in cases:
        input_data, pack = build_input_data_step2(
            wind_factor=wind_factor,
            include_storage=True,
            Pch_max=case["Pch_max"],
            Pdis_max=case["Pdis_max"],
            E_max=case["E_max"],
            eta_ch=0.90,
            eta_dis=0.95,
            e0=case["E_max"] / 2,
            cyclic_storage=True
        )

        model = LPOptimizationProblem(input_data)
        model.run()
        result = extract_market_results(model, pack)

        outputs.append({
            "name": case["name"],
            "Pch_max": case["Pch_max"],
            "Pdis_max": case["Pdis_max"],
            "E_max": case["E_max"],

            # summary values
            "total_operating_cost": result["total_operating_cost"],
            "total_social_welfare": result["total_social_welfare"],
            "total_storage_profit": result["total_storage_profit"],
            "avg_mcp": np.mean(result["mcp"]),
            "max_mcp": np.max(result["mcp"]),
            "min_mcp": np.min(result["mcp"]),

            # full hourly results
            "hourly_results": result
        })

    return outputs

# ============================================================
# 10. Example main
# ============================================================
if __name__ == "__main__":
    # Example hourly wind availability
    wind_factor = load_wind_factor_from_mat(scenario=30) #build_default_wind_profile()

    # ----------------------------
    # Case 1: Without storage
    # ----------------------------
    input_no_storage, pack_no_storage = build_input_data_step2(
        wind_factor=wind_factor,
        include_storage=False
    )

    model_no_storage = LPOptimizationProblem(input_no_storage)
    model_no_storage.run()

    result_no_storage = extract_market_results(model_no_storage, pack_no_storage)
    print_step2_summary(result_no_storage, pack_no_storage, title="CASE 1: WITHOUT STORAGE")

    # ----------------------------
    # Case 2: With storage
    # ----------------------------
    input_with_storage, pack_with_storage = build_input_data_step2(
        wind_factor=wind_factor,
        include_storage=True,
        Pch_max=300.0, # Maximum charging power every hour
        Pdis_max=300.0, # Maximum discharging power every hour
        E_max=1200.0, # Maximum energy storage capacity
        eta_ch=0.90, # Charging efficiency
        eta_dis=0.95, # Discharging efficiency
        e0=600.0, # Initial energy storage level
        cyclic_storage=True
    )

    model_with_storage = LPOptimizationProblem(input_with_storage)
    model_with_storage.run()

    result_with_storage = extract_market_results(model_with_storage, pack_with_storage)
    print_step2_summary(result_with_storage, pack_with_storage, title="CASE 2: WITH STORAGE")
    plot_storage_operation_compact(result_with_storage)
    plot_storage_price_and_profit(result_with_storage)
    plot_cost_profit_contributions(result_with_storage, pack_with_storage)
    # ----------------------------
    # Comparison
    # ----------------------------
    compare_cases(result_no_storage, result_with_storage)
    plot_mcp_comparison(result_no_storage, result_with_storage)
    plot_system_comparison(result_no_storage, result_with_storage)
    #def print_assignment_results(result_pack: dict, data_pack: dict, case_name: str):
    print_assignment_results(result_no_storage, pack_no_storage, case_name="Without Storage")
    print_assignment_results(result_with_storage, pack_with_storage, case_name="With Storage")

    sensitivity_outputs = run_sensitivity_cases2(wind_factor)
    print_sensitivity_results(sensitivity_outputs)
    plot_storage_hourly(sensitivity_outputs)


