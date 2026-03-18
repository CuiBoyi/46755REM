
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from pathlib import Path
import matplotlib.pyplot as plt
from utili import load_wind_scen


# ============================================================
# 1. Small helper container
# ============================================================
class Expando:
    pass


# ============================================================
# 2. LP input data container
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
# 3. Generic LP wrapper
# ============================================================
class LPOptimizationProblem:
    def __init__(self, input_data: LPInputData):
        self.data = input_data
        self.results = Expando()
        self._build_model()

    def _build_model(self):
        self.model = gp.Model(self.data.model_name)
        self.model.Params.OutputFlag = 0

        self.variables = []
        for i, name in enumerate(self.data.var_names):
            var = self.model.addVar(
                lb=self.data.lb[i],
                ub=self.data.ub[i],
                name=name
            )
            self.variables.append(var)

        obj_expr = gp.quicksum(
            self.data.obj[j] * self.variables[j]
            for j in range(len(self.variables))
        )
        self.model.setObjective(obj_expr, self.data.objective_sense)

        self.constraints = []
        n_constr = len(self.data.constr_names)
        n_var = len(self.data.var_names)

        for i in range(n_constr):
            expr = gp.quicksum(
                self.data.A[i, j] * self.variables[j]
                for j in range(n_var)
                if abs(self.data.A[i, j]) > 1e-12
            )
            c = self.model.addLConstr(
                expr,
                self.data.sense[i],
                self.data.rhs[i],
                name=self.data.constr_names[i]
            )
            self.constraints.append(c)

        self.model.update()

    def run(self):
        self.model.optimize()
        if self.model.status != GRB.OPTIMAL:
            raise RuntimeError(f"Optimization failed. Status = {self.model.status}")

        self.results.objective_value = self.model.ObjVal
        self.results.x = np.array([v.X for v in self.model.getVars()], dtype=float)
        self.results.var_names = [v.VarName for v in self.model.getVars()]
        self.results.duals = np.array([c.Pi for c in self.model.getConstrs()], dtype=float)
        self.results.constr_names = [c.ConstrName for c in self.model.getConstrs()]


# ============================================================
# 4. Base market data
# ============================================================
def get_base_market_data():
    thermal_names = [f"g{i}" for i in range(1, 13)]
    wind_names = [f"w{i}" for i in range(1, 7)]
    load_names = [f"d{i}" for i in range(1, 18)]

    Pmax = np.array([152, 152, 350, 591, 60, 155, 155, 400, 400, 300, 310, 350], dtype=float)
    Pmin = np.zeros(12, dtype=float)
    R_plus = np.array([40, 40, 70, 180, 60, 30, 30, 0, 0, 0, 60, 40], dtype=float)
    R_minus = np.array([40, 40, 70, 180, 60, 30, 30, 0, 0, 0, 60, 40], dtype=float)

    RU = np.array([120, 120, 350, 240, 60, 155, 155, 280, 280, 300, 180, 240], dtype=float)
    RD = np.array([120, 120, 350, 240, 60, 155, 155, 280, 280, 300, 180, 240], dtype=float)

    gen_cost = np.array([13.32, 13.32, 20.70, 20.93, 26.11, 10.52, 10.52, 6.02, 5.47, 0.00, 10.52, 10.89], dtype=float)
    C_u = np.array([15, 15, 10, 8, 7, 16, 16, 0, 0, 0, 17, 16], dtype=float)
    C_d = np.array([14, 14, 9, 7, 5, 14, 14, 0, 0, 0, 16, 14], dtype=float)
    C_plus = np.array([15, 15, 24, 25, 28, 16, 16, 0, 0, 0, 14, 16], dtype=float)
    C_minus = np.array([11, 11, 16, 17, 23, 7, 7, 0, 0, 0, 8, 8], dtype=float)


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
        "RU": RU,
        "RD": RD,
        "R_plus": R_plus,
        "R_minus": R_minus,
        "gen_cost": gen_cost,
        "Wcap": Wcap,
        "wind_cost": wind_cost,
        "load_share_percent": load_share_percent,
        "base_load_bid": base_load_bid,
        "system_demand": system_demand,
        "C_u": C_u,
        "C_d": C_d,
        "C_plus": C_plus,
        "C_minus": C_minus
    }


# ============================================================
# 5. Hourly load bids
# ============================================================
def build_hourly_load_bids(system_demand: np.ndarray, base_load_bid: np.ndarray) -> np.ndarray:
    dmin = system_demand.min()
    dmax = system_demand.max()

    if abs(dmax - dmin) < 1e-12:
        multiplier = np.ones_like(system_demand)
    else:
        normalized = (system_demand - dmin) / (dmax - dmin)
        multiplier = 0.90 + 0.25 * normalized

    return multiplier[:, None] * base_load_bid[None, :]


# ============================================================
# 6. Load wind factor from .mat
# ============================================================
def load_wind_factor_from_mat(scenario: int = 30) -> np.ndarray:
    base_dir = Path(__file__).resolve().parent
    candidates = [
        base_dir / "dataset" / "WindScen.mat",
        base_dir.parent / "dataset" / "WindScen.mat",
    ]

    path_wind = next((p for p in candidates if p.exists()), None)
    if path_wind is None:
        raise FileNotFoundError(
            f"Could not find WindScen.mat. Checked: {[str(p) for p in candidates]}"
        )

    wind = load_wind_scen(path_wind)
    wind_factor = np.zeros((24, 6), dtype=float)

    for i in range(6):
        for t in range(24):
            wind_factor[t, i] = wind[i][t, scenario]

    return wind_factor


# ============================================================
# 7. Step 1 benchmark: single-hour day-ahead problem
# ============================================================
def build_input_data_step1_single_hour(hour: int, wind_factor_hour: np.ndarray) -> tuple[LPInputData, dict]:
    data = get_base_market_data()

    thermal_names = data["thermal_names"]
    wind_names = data["wind_names"]
    load_names = data["load_names"]

    Pmax = data["Pmax"]
    Pmin = data["Pmin"]
    R_plus = data["R_plus"]
    R_minus = data["R_minus"]
    RU = data["RU"]
    RD = data["RD"]
    gen_cost = data["gen_cost"]
    Wcap = data["Wcap"]
    wind_cost = data["wind_cost"]
    load_share_percent = data["load_share_percent"]
    base_load_bid = data["base_load_bid"]
    system_demand = data["system_demand"]
    C_u = data["C_u"]
    C_d = data["C_d"]
    C_plus = data["C_plus"]
    C_minus = data["C_minus"]

    t = hour - 1
    load_bid_hourly = build_hourly_load_bids(system_demand, base_load_bid)
    bid = load_bid_hourly[t, :]
    Dmax = system_demand[t] * load_share_percent / 100.0
    Wmax = wind_factor_hour * Wcap

    ng = len(thermal_names)
    nw = len(wind_names)
    nd = len(load_names)

    idx_g = np.arange(0, ng)
    idx_w = np.arange(ng, ng + nw)
    idx_d = np.arange(ng + nw, ng + nw + nd)

    var_names = thermal_names + wind_names + load_names
    nv = len(var_names)

    lb = np.zeros(nv)
    ub = np.full(nv, np.inf)

    lb[idx_g] = Pmin
    ub[idx_g] = Pmax
    ub[idx_w] = Wmax
    ub[idx_d] = Dmax

    obj = np.zeros(nv)
    obj[idx_g] = -gen_cost
    obj[idx_w] = -wind_cost
    obj[idx_d] = bid

    constr_names = ["balance"]
    A = np.zeros((1, nv))
    rhs = np.array([0.0], dtype=float)
    sense = np.array([GRB.EQUAL], dtype=object)

    A[0, idx_g] = 1.0
    A[0, idx_w] = 1.0
    A[0, idx_d] = -1.0

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
        model_name=f"step1_day_ahead_hour_{hour}",
    )

    pack = {
        "hour": hour,
        "thermal_names": thermal_names,
        "wind_names": wind_names,
        "load_names": load_names,
        "Pmax": Pmax,
        "Pmin": Pmin,
        "gen_cost": gen_cost,
        "Wcap": Wcap,
        "wind_cost": wind_cost,
        "Dmax": Dmax,
        "Wmax": Wmax,
        "load_bid": bid,
        "system_demand_hour": system_demand[t],
        "index": {"g": idx_g, "w": idx_w, "d": idx_d},
    }
    return input_data, pack


def extract_step1_results(model: LPOptimizationProblem, pack: dict) -> dict:
    idx = pack["index"]
    x = model.results.x
    duals = model.results.duals

    g = x[idx["g"]]
    w = x[idx["w"]]
    d = x[idx["d"]]

    lambda_da = -duals[0]

    gen_cost = pack["gen_cost"]
    load_bid = pack["load_bid"]

    thermal_profit = g * (lambda_da - gen_cost)
    wind_profit = w * lambda_da
    load_utility = d * load_bid

    return {
        "g_da": g,
        "w_da": w,
        "d_da": d,
        "lambda_da": lambda_da,
        "thermal_profit_da": thermal_profit,
        "wind_profit_da": wind_profit,
        "total_operating_cost": np.sum(g * gen_cost),
        "total_load_value": np.sum(d * load_bid),
        "total_social_welfare": np.sum(d * load_bid) - np.sum(g * gen_cost),
        "load_utility": load_utility,
    }


# ============================================================
# 8. Step 6: reserve market
# ============================================================
def build_input_data_reserve_market(
    hour: int,
    flexible_units: list[str],
    upward_reserve_fraction: float = 0.15,
    downward_reserve_fraction: float = 0.10,
    upward_offer_multiplier: float = 1.00,
    downward_offer_multiplier: float = 1.00,
) -> tuple[LPInputData, dict]:
    data = get_base_market_data()

    thermal_names = data["thermal_names"]
    Pmax = data["Pmax"]
    Pmin = data["Pmin"]
    R_plus = data["R_plus"]
    R_minus = data["R_minus"]
    RU = data["RU"]
    RD = data["RD"]

    gen_cost = data["gen_cost"]
    C_u = data["C_u"]
    C_d = data["C_d"]
    system_demand = data["system_demand"]

    t = hour - 1
    demand_hour = system_demand[t]

    R_up = upward_reserve_fraction * demand_hour
    R_down = downward_reserve_fraction * demand_hour

    nf = len(flexible_units)

    idx_rup = np.arange(0, nf)
    idx_rdown = np.arange(nf, 2 * nf)

    var_names = [f"r_up_{name}" for name in flexible_units] + [f"r_down_{name}" for name in flexible_units]
    nv = len(var_names)

    lb = np.zeros(nv)
    ub = np.zeros(nv)

    up_offer = np.zeros(nf)
    down_offer = np.zeros(nf)
    reserve_capacity = np.zeros(nf)

    for k, name in enumerate(flexible_units):
        i = thermal_names.index(name)
        up_offer[k] = upward_offer_multiplier * C_u[i]
        down_offer[k] = downward_offer_multiplier * C_d[i]

        reserve_capacity[k] = Pmax[i] - Pmin[i]

        # ub[idx_rup[k]] = reserve_capacity[k]
        # ub[idx_rdown[k]] = reserve_capacity[k]
        ub[idx_rup[k]] = R_plus[i]
        ub[idx_rdown[k]] = R_minus[i]

    obj = np.zeros(nv)
    obj[idx_rup] = up_offer
    obj[idx_rdown] = down_offer

    # 2 system requirements + nf coupling constraints
    constr_names = ["upward_reserve_requirement", "downward_reserve_requirement"]
    for name in flexible_units:
        constr_names.append(f"cap_coupling_{name}")

    nc = 2 + nf
    A = np.zeros((nc, nv))
    rhs = np.zeros(nc)
    sense = np.empty(nc, dtype=object)

    # upward reserve requirement
    A[0, idx_rup] = 1.0
    rhs[0] = R_up
    sense[0] = GRB.EQUAL

    # downward reserve requirement
    A[1, idx_rdown] = 1.0
    rhs[1] = R_down
    sense[1] = GRB.EQUAL

    # coupling: r_up_i + r_down_i <= Pmax_i - Pmin_i
    for k in range(nf):
        row = 2 + k
        A[row, idx_rup[k]] = 1.0
        A[row, idx_rdown[k]] = 1.0
        rhs[row] = reserve_capacity[k]
        sense[row] = GRB.LESS_EQUAL

    input_data = LPInputData(
        var_names=var_names,
        constr_names=constr_names,
        obj=obj,
        A=A,
        rhs=rhs,
        sense=sense,
        lb=lb,
        ub=ub,
        objective_sense=GRB.MINIMIZE,
        model_name=f"step6_reserve_market_hour_{hour}",
    )

    pack = {
        "hour": hour,
        "flexible_units": flexible_units,
        "thermal_names": thermal_names,
        "Pmax": Pmax,
        "Pmin": Pmin,
        "gen_cost": gen_cost,
        "demand_hour": demand_hour,
        "R_up": R_up,
        "R_down": R_down,
        "up_offer": up_offer,
        "down_offer": down_offer,
        "reserve_capacity": reserve_capacity,
        "index": {
            "r_up": idx_rup,
            "r_down": idx_rdown,
        }
    }
    return input_data, pack


def extract_reserve_results(model: LPOptimizationProblem, pack_reserve: dict) -> dict:
    idx = pack_reserve["index"]
    x = model.results.x
    duals = model.results.duals

    r_up = x[idx["r_up"]]
    r_down = x[idx["r_down"]]

    # For minimization with equality constraints, duals are the reserve prices
    lambda_r_up = duals[0]
    lambda_r_down = duals[1]

    return {
        "r_up": r_up,
        "r_down": r_down,
        "lambda_r_up": lambda_r_up,
        "lambda_r_down": lambda_r_down,
        "total_upward_reserve": np.sum(r_up),
        "total_downward_reserve": np.sum(r_down),
    }


# ============================================================
# 9. Step 6: day-ahead market after reserve market
# ============================================================
def build_input_data_step6_day_ahead(
    hour: int,
    wind_factor_hour: np.ndarray,
    pack_reserve: dict,
    result_reserve: dict,
) -> tuple[LPInputData, dict]:
    data = get_base_market_data()

    thermal_names = data["thermal_names"]
    wind_names = data["wind_names"]
    load_names = data["load_names"]

    Pmax = data["Pmax"].copy()
    Pmin = data["Pmin"].copy()
    gen_cost = data["gen_cost"]
    Wcap = data["Wcap"]
    wind_cost = data["wind_cost"]
    load_share_percent = data["load_share_percent"]
    base_load_bid = data["base_load_bid"]
    system_demand = data["system_demand"]

    flexible_units = pack_reserve["flexible_units"]
    r_up = result_reserve["r_up"]
    r_down = result_reserve["r_down"]

    t = hour - 1
    load_bid_hourly = build_hourly_load_bids(system_demand, base_load_bid)
    bid = load_bid_hourly[t, :]
    Dmax = system_demand[t] * load_share_percent / 100.0
    Wmax = wind_factor_hour * Wcap

    ng = len(thermal_names)
    nw = len(wind_names)
    nd = len(load_names)

    idx_g = np.arange(0, ng)
    idx_w = np.arange(ng, ng + nw)
    idx_d = np.arange(ng + nw, ng + nw + nd)

    var_names = thermal_names + wind_names + load_names
    nv = len(var_names)

    lb = np.zeros(nv)
    ub = np.full(nv, np.inf)

    # Default bounds
    lb[idx_g] = Pmin
    ub[idx_g] = Pmax

    # Adjust bounds for reserve providers
    reserve_up_map = np.zeros(ng)
    reserve_down_map = np.zeros(ng)

    for k, name in enumerate(flexible_units):
        i = thermal_names.index(name)
        reserve_up_map[i] = r_up[k]
        reserve_down_map[i] = r_down[k]

    # Sequential coupling:
    # g_i + r_up_i <= Pmax_i
    # g_i - r_down_i >= Pmin_i  -> g_i >= Pmin_i + r_down_i
    ub[idx_g] = Pmax - reserve_up_map
    lb[idx_g] = Pmin + reserve_down_map

    # Feasibility check
    for i, name in enumerate(thermal_names):
        if lb[idx_g[i]] > ub[idx_g[i]] + 1e-9:
            raise ValueError(
                f"Infeasible DA bounds for {name}: lb={lb[idx_g[i]]:.4f}, ub={ub[idx_g[i]]:.4f}"
            )

    ub[idx_w] = Wmax
    ub[idx_d] = Dmax

    obj = np.zeros(nv)
    obj[idx_g] = -gen_cost
    obj[idx_w] = -wind_cost
    obj[idx_d] = bid

    constr_names = ["balance"]
    A = np.zeros((1, nv))
    rhs = np.array([0.0], dtype=float)
    sense = np.array([GRB.EQUAL], dtype=object)

    A[0, idx_g] = 1.0
    A[0, idx_w] = 1.0
    A[0, idx_d] = -1.0

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
        model_name=f"step6_day_ahead_after_reserve_hour_{hour}",
    )

    pack = {
        "hour": hour,
        "thermal_names": thermal_names,
        "wind_names": wind_names,
        "load_names": load_names,
        "Pmax": Pmax,
        "Pmin": Pmin,
        "gen_cost": gen_cost,
        "Wcap": Wcap,
        "wind_cost": wind_cost,
        "Dmax": Dmax,
        "Wmax": Wmax,
        "load_bid": bid,
        "system_demand_hour": system_demand[t],
        "reserve_up_map": reserve_up_map,
        "reserve_down_map": reserve_down_map,
        "index": {"g": idx_g, "w": idx_w, "d": idx_d},
    }
    return input_data, pack


# ============================================================
# 10. Printing helpers
# ============================================================
def print_day_ahead_summary(title: str, pack_da: dict, result_da: dict):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(f"Hour: {pack_da['hour']}")
    print(f"Day-ahead price: {result_da['lambda_da']:.4f} €/MWh")
    print(f"Total operating cost: {result_da['total_operating_cost']:.4f} €")
    print(f"Total load value:     {result_da['total_load_value']:.4f} €")
    print(f"Total social welfare: {result_da['total_social_welfare']:.4f} €")

    print("\nConventional dispatch:")
    for name, val in zip(pack_da["thermal_names"], result_da["g_da"]):
        print(f"{name:>4}: {val:10.4f} MW")

    print("\nWind dispatch:")
    for name, val in zip(pack_da["wind_names"], result_da["w_da"]):
        print(f"{name:>4}: {val:10.4f} MW")

    print("\nDemand served:")
    for name, val in zip(pack_da["load_names"], result_da["d_da"]):
        print(f"{name:>4}: {val:10.4f} MW")


def print_reserve_summary(pack_reserve: dict, result_reserve: dict):
    print("\n" + "=" * 80)
    print("RESERVE MARKET RESULTS")
    print("=" * 80)
    print(f"Hour: {pack_reserve['hour']}")
    print(f"Demand in this hour: {pack_reserve['demand_hour']:.4f} MW")
    print(f"Upward reserve requirement:   {pack_reserve['R_up']:.4f} MW")
    print(f"Downward reserve requirement: {pack_reserve['R_down']:.4f} MW")
    print(f"Upward reserve price:         {result_reserve['lambda_r_up']:.4f} €/MW")
    print(f"Downward reserve price:       {result_reserve['lambda_r_down']:.4f} €/MW")

    print("\nReserve awards:")
    print(f"{'Unit':>6} | {'R_up (MW)':>12} | {'R_down (MW)':>12} | {'Up offer':>10} | {'Down offer':>12}")
    print("-" * 66)

    for k, name in enumerate(pack_reserve["flexible_units"]):
        print(
            f"{name:>6} | "
            f"{result_reserve['r_up'][k]:12.4f} | "
            f"{result_reserve['r_down'][k]:12.4f} | "
            f"{pack_reserve['up_offer'][k]:10.4f} | "
            f"{pack_reserve['down_offer'][k]:12.4f}"
        )


def print_reserve_coupled_bounds(pack_step6_da: dict):
    print("\n" + "=" * 80)
    print("DAY-AHEAD BOUNDS AFTER RESERVE PROCUREMENT")
    print("=" * 80)
    print(f"{'Unit':>6} | {'R_up':>10} | {'R_down':>10} | {'DA lower bound':>15} | {'DA upper bound':>15}")
    print("-" * 68)

    thermal_names = pack_step6_da["thermal_names"]
    Pmax = pack_step6_da["Pmax"]
    Pmin = pack_step6_da["Pmin"]
    reserve_up_map = pack_step6_da["reserve_up_map"]
    reserve_down_map = pack_step6_da["reserve_down_map"]

    for i, name in enumerate(thermal_names):
        lb_new = Pmin[i] + reserve_down_map[i]
        ub_new = Pmax[i] - reserve_up_map[i]
        print(
            f"{name:>6} | "
            f"{reserve_up_map[i]:10.4f} | "
            f"{reserve_down_map[i]:10.4f} | "
            f"{lb_new:15.4f} | "
            f"{ub_new:15.4f}"
        )


def print_price_comparison(result_benchmark: dict, result_step6_da: dict):
    print("\n" + "=" * 80)
    print("PRICE COMPARISON")
    print("=" * 80)
    print(f"Benchmark DA price without reserve market: {result_benchmark['lambda_da']:.4f} €/MWh")
    print(f"DA price with sequential reserve market:   {result_step6_da['lambda_da']:.4f} €/MWh")
    print(f"Price change:                              {result_step6_da['lambda_da'] - result_benchmark['lambda_da']:.4f} €/MWh")


def print_dispatch_comparison(pack_benchmark: dict, result_benchmark: dict, result_step6_da: dict):
    print("\n" + "=" * 80)
    print("CONVENTIONAL DISPATCH COMPARISON")
    print("=" * 80)
    print(f"{'Unit':>6} | {'Without reserve':>16} | {'With reserve':>14} | {'Difference':>12}")
    print("-" * 62)

    for i, name in enumerate(pack_benchmark["thermal_names"]):
        g0 = result_benchmark["g_da"][i]
        g1 = result_step6_da["g_da"][i]
        print(f"{name:>6} | {g0:16.4f} | {g1:14.4f} | {g1 - g0:12.4f}")


# ============================================================
# 11. Plot helper
# ============================================================
def plot_step6_results(
    pack_reserve: dict,
    result_reserve: dict,
    pack_benchmark: dict,
    result_benchmark: dict,
    pack_step6_da: dict,
    result_step6_da: dict,
):
    flexible_units = pack_reserve["flexible_units"]
    thermal_names = pack_benchmark["thermal_names"]

    reserve_up_map = np.zeros(len(thermal_names))
    reserve_down_map = np.zeros(len(thermal_names))
    for k, name in enumerate(flexible_units):
        i = thermal_names.index(name)
        reserve_up_map[i] = result_reserve["r_up"][k]
        reserve_down_map[i] = result_reserve["r_down"][k]

    x = np.arange(len(thermal_names))
    width = 0.35

    plt.figure(figsize=(13, 6))
    plt.bar(x - width / 2, result_benchmark["g_da"], width=width, alpha=0.8, label="DA dispatch without reserve")
    plt.bar(x + width / 2, result_step6_da["g_da"], width=width, alpha=0.8, label="DA dispatch with reserve")
    plt.xticks(x, thermal_names, rotation=45)
    plt.ylabel("MW")
    plt.title("Conventional day-ahead dispatch comparison")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.figure(figsize=(13, 6))
    plt.bar(x - width / 2, reserve_up_map, width=width, alpha=0.8, label="Upward reserve")
    plt.bar(x + width / 2, reserve_down_map, width=width, alpha=0.8, label="Downward reserve")
    plt.xticks(x, thermal_names, rotation=45)
    plt.ylabel("MW")
    plt.title("Reserve awards by unit")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.figure(figsize=(8, 5))
    prices = [
        result_reserve["lambda_r_up"],
        result_reserve["lambda_r_down"],
        result_benchmark["lambda_da"],
        result_step6_da["lambda_da"],
    ]
    labels = ["Reserve up", "Reserve down", "DA w/o reserve", "DA with reserve"]
    plt.bar(labels, prices, alpha=0.8)
    plt.ylabel("Price")
    plt.title("Price comparison")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.show()


# ============================================================
# 12. Main
# ============================================================
if __name__ == "__main__":
    HOUR = 18
    SCENARIO = 30

    FLEXIBLE_UNITS = ["g1", "g2", "g3", "g4", "g5", "g6", "g7", "g11", "g12"]

    UPWARD_RESERVE_FRACTION = 0.15
    DOWNWARD_RESERVE_FRACTION = 0.10

    # These are assumptions because the assignment screenshot does not specify reserve offer prices
    UPWARD_OFFER_MULTIPLIER = 1.00
    DOWNWARD_OFFER_MULTIPLIER = 1.00

    # ----------------------------
    # Load wind data
    # ----------------------------
    wind_factor = load_wind_factor_from_mat(scenario=SCENARIO)
    wind_factor_hour = wind_factor[HOUR - 1, :]

    # ----------------------------
    # Benchmark: Step 1 without reserve market
    # ----------------------------
    input_benchmark, pack_benchmark = build_input_data_step1_single_hour(
        hour=HOUR,
        wind_factor_hour=wind_factor_hour
    )
    model_benchmark = LPOptimizationProblem(input_benchmark)
    model_benchmark.run()
    result_benchmark = extract_step1_results(model_benchmark, pack_benchmark)

    # ----------------------------
    # Reserve market first
    # ----------------------------
    input_reserve, pack_reserve = build_input_data_reserve_market(
        hour=HOUR,
        flexible_units=FLEXIBLE_UNITS,
        upward_reserve_fraction=UPWARD_RESERVE_FRACTION,
        downward_reserve_fraction=DOWNWARD_RESERVE_FRACTION,
        upward_offer_multiplier=UPWARD_OFFER_MULTIPLIER,
        downward_offer_multiplier=DOWNWARD_OFFER_MULTIPLIER,
    )
    model_reserve = LPOptimizationProblem(input_reserve)
    model_reserve.run()
    result_reserve = extract_reserve_results(model_reserve, pack_reserve)

    # ----------------------------
    # Day-ahead market after reserve market
    # ----------------------------
    input_step6_da, pack_step6_da = build_input_data_step6_day_ahead(
        hour=HOUR,
        wind_factor_hour=wind_factor_hour,
        pack_reserve=pack_reserve,
        result_reserve=result_reserve,
    )
    model_step6_da = LPOptimizationProblem(input_step6_da)
    model_step6_da.run()
    result_step6_da = extract_step1_results(model_step6_da, pack_step6_da)

    # ----------------------------
    # Print results
    # ----------------------------
    print_day_ahead_summary(
        "BENCHMARK DAY-AHEAD MARKET WITHOUT RESERVE",
        pack_benchmark,
        result_benchmark
    )

    print_reserve_summary(pack_reserve, result_reserve)
    print_reserve_coupled_bounds(pack_step6_da)

    print_day_ahead_summary(
        "DAY-AHEAD MARKET AFTER SEQUENTIAL RESERVE PROCUREMENT",
        pack_step6_da,
        result_step6_da
    )

    print_price_comparison(result_benchmark, result_step6_da)
    print_dispatch_comparison(pack_benchmark, result_benchmark, result_step6_da)

    # ----------------------------
    # Plot
    # ----------------------------
    plot_step6_results(
        pack_reserve=pack_reserve,
        result_reserve=result_reserve,
        pack_benchmark=pack_benchmark,
        result_benchmark=result_benchmark,
        pack_step6_da=pack_step6_da,
        result_step6_da=result_step6_da,
    )