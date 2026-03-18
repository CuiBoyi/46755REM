
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from pathlib import Path
from utili import load_wind_scen
from plot import plot_dispatch_and_profit_step5


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
    wind_factor = np.zeros((24, 6), dtype=float)

    for i in range(6):
        for t in range(24):
            wind_factor[t, i] = wind[i][t, scenario]

    return wind_factor


# ============================================================
# 7. Step 1: single-hour day-ahead problem
# ============================================================
def build_input_data_step1_single_hour(hour: int, wind_factor_hour: np.ndarray) -> tuple[LPInputData, dict]:
    """
    hour: 1..24
    wind_factor_hour: shape (6,)
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

    t = hour - 1
    load_bid_hourly = build_hourly_load_bids(system_demand, base_load_bid)
    bid = load_bid_hourly[t, :]
    Dmax = system_demand[t] * load_share_percent / 100.0
    Wmax = wind_factor_hour * Wcap

    ng = len(thermal_names)
    nw = len(wind_names)
    nd = len(load_names)

    var_names = []
    idx_g = np.arange(0, ng)
    idx_w = np.arange(ng, ng + nw)
    idx_d = np.arange(ng + nw, ng + nw + nd)

    for name in thermal_names:
        var_names.append(name)
    for name in wind_names:
        var_names.append(name)
    for name in load_names:
        var_names.append(name)

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
    nc = 1
    A = np.zeros((nc, nv))
    rhs = np.zeros(nc)
    sense = np.empty(nc, dtype=object)

    A[0, idx_g] = 1.0
    A[0, idx_w] = 1.0
    A[0, idx_d] = -1.0
    rhs[0] = 0.0
    sense[0] = GRB.EQUAL

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
        "index": {
            "g": idx_g,
            "w": idx_w,
            "d": idx_d,
        }
    }
    return input_data, pack


def extract_step1_results(model: LPOptimizationProblem, pack: dict) -> dict:
    idx = pack["index"]
    x = model.results.x
    duals = model.results.duals

    g = x[idx["g"]]
    w = x[idx["w"]]
    d = x[idx["d"]]

    # For max welfare with balance written as sum(supply)-sum(demand)=0
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
# 8. Build actual real-time deviations
# ============================================================
def build_real_time_realization(
    result_da: dict,
    thermal_names: list[str],
    wind_names: list[str],
    outage_unit: str = "g9",
    wind_down_units: list[str] = None,
    wind_up_units: list[str] = None,
    wind_down_rate: float = 0.15,
    wind_up_rate: float = 0.10,
):
    if wind_down_units is None:
        wind_down_units = ["w1", "w2", "w3"]
    if wind_up_units is None:
        wind_up_units = ["w4", "w5", "w6"]

    g_da = result_da["g_da"].copy()
    w_da = result_da["w_da"].copy()

    g_actual_forced = g_da.copy()
    w_actual = w_da.copy()

    # generator outage
    if outage_unit in thermal_names:
        ig = thermal_names.index(outage_unit)
        g_actual_forced[ig] = 0.0

    # wind deviations
    for name in wind_down_units:
        jw = wind_names.index(name)
        w_actual[jw] = (1.0 - wind_down_rate) * w_da[jw]

    for name in wind_up_units:
        jw = wind_names.index(name)
        w_actual[jw] = (1.0 + wind_up_rate) * w_da[jw]

    delta_g_forced = g_actual_forced - g_da
    delta_w = w_actual - w_da

    total_imbalance = np.sum(delta_g_forced) + np.sum(delta_w)

    return {
        "g_actual_forced": g_actual_forced,
        "w_actual": w_actual,
        "delta_g_forced": delta_g_forced,
        "delta_w": delta_w,
        "total_imbalance": total_imbalance,  # negative => system short
        "outage_unit": outage_unit,
        "wind_down_units": wind_down_units,
        "wind_up_units": wind_up_units,
        "wind_down_rate": wind_down_rate,
        "wind_up_rate": wind_up_rate,
    }


# ============================================================
# 9. Step 5: balancing market
# ============================================================
def build_input_data_balancing(
    pack_da: dict,
    result_da: dict,
    rt_realization: dict,
    balancing_units: list[str],
    load_curtailment_cost: float = 500.0,
) -> tuple[LPInputData, dict]:
    thermal_names = pack_da["thermal_names"]
    Pmax = pack_da["Pmax"]
    Pmin = pack_da["Pmin"]
    gen_cost = pack_da["gen_cost"]
    lambda_da = result_da["lambda_da"]
    g_da = result_da["g_da"]
    g_actual_forced = rt_realization["g_actual_forced"]
    total_imbalance = rt_realization["total_imbalance"]

    # Need balancing amount:
    # sum(up) - sum(down) + shed = - total_imbalance
    required = -total_imbalance

    nb = len(balancing_units)

    var_names = []
    idx_up = np.arange(0, nb)
    idx_down = np.arange(nb, 2 * nb)
    idx_ls = 2 * nb

    for name in balancing_units:
        var_names.append(f"up_{name}")
    for name in balancing_units:
        var_names.append(f"down_{name}")
    var_names.append("load_shed")

    nv = len(var_names)

    lb = np.zeros(nv)
    ub = np.full(nv, np.inf)

    up_price = np.zeros(nb)
    down_price = np.zeros(nb)

    for k, name in enumerate(balancing_units):
        i = thermal_names.index(name)

        # Up/down offer prices
        up_price[k] = lambda_da + 0.10 * gen_cost[i]
        down_price[k] = lambda_da - 0.15 * gen_cost[i]

        # IMPORTANT:
        # Upward headroom is based on actual forced output before balancing
        ub[idx_up[k]] = max(0.0, Pmax[i] - g_actual_forced[i])

        # Downward room is based on actual forced output before balancing
        ub[idx_down[k]] = max(0.0, g_actual_forced[i] - Pmin[i])

    ub[idx_ls] = GRB.INFINITY #ls means load shedding

    # Objective:
    # min sum(up_price*up) - sum(down_price*down) + VOLL * load_shed
    obj = np.zeros(nv)
    obj[idx_up] = up_price
    obj[idx_down] = -down_price
    obj[idx_ls] = load_curtailment_cost

    constr_names = ["balancing_balance"]
    nc = 1
    A = np.zeros((nc, nv))
    rhs = np.zeros(nc)
    sense = np.empty(nc, dtype=object)

    A[0, idx_up] = 1.0
    A[0, idx_down] = -1.0
    A[0, idx_ls] = 1.0
    rhs[0] = required
    sense[0] = GRB.EQUAL

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
        model_name="step5_balancing_market",
    )

    pack = {
        "balancing_units": balancing_units,
        "up_price": up_price,
        "down_price": down_price,
        "required_balancing": required,
        "load_curtailment_cost": load_curtailment_cost,
        "index": {
            "up": idx_up,
            "down": idx_down,
            "ls": idx_ls,
        }
    }
    return input_data, pack


def extract_balancing_results(model: LPOptimizationProblem, pack_bal: dict, lambda_da: float) -> dict:
    idx = pack_bal["index"]
    x = model.results.x
    dual = model.results.duals[0]

    up = x[idx["up"]]
    down = x[idx["down"]]
    load_shed = x[idx["ls"]]

    # For minimization with equality balance, dual is directly the marginal cost
    lambda_bal = dual

    return {
        "up": up,
        "down": down,
        "load_shed": load_shed,
        "lambda_bal": lambda_bal,
        "required_balancing": pack_bal["required_balancing"],
        "system_short": pack_bal["required_balancing"] > 1e-9,
        "system_long": pack_bal["required_balancing"] < -1e-9,
        "system_balanced": abs(pack_bal["required_balancing"]) <= 1e-9,
    }


# ============================================================
# 10. Profit settlement
# ============================================================
def settlement_price_two_price(delta: float, system_short: bool, system_long: bool, lambda_da: float, lambda_bal: float) -> float:
    """
    Two-price rule used here:
    - If the system is short:
      * harmful deviation (delta < 0) settled at balancing price
      * helpful deviation (delta > 0) settled at day-ahead price
    - If the system is long:
      * harmful deviation (delta > 0) settled at balancing price
      * helpful deviation (delta < 0) settled at day-ahead price
    - If exactly balanced: settle at day-ahead price
    """
    if system_short:
        return lambda_bal if delta < 0 else lambda_da
    elif system_long:
        return lambda_bal if delta > 0 else lambda_da
    else:
        return lambda_da


def compute_total_profits(
    pack_da: dict,
    result_da: dict,
    rt_realization: dict,
    pack_bal: dict,
    result_bal: dict,
):
    thermal_names = pack_da["thermal_names"]
    wind_names = pack_da["wind_names"]
    gen_cost = pack_da["gen_cost"]

    lambda_da = result_da["lambda_da"]
    lambda_bal = result_bal["lambda_bal"]

    g_da = result_da["g_da"]
    w_da = result_da["w_da"]

    g_actual_forced = rt_realization["g_actual_forced"].copy()
    w_actual = rt_realization["w_actual"].copy()

    balancing_units = pack_bal["balancing_units"]
    up = result_bal["up"]
    down = result_bal["down"]

    # Actual conventional output after balancing activation
    g_actual = g_actual_forced.copy()
    up_map = np.zeros(len(thermal_names))
    down_map = np.zeros(len(thermal_names))

    for k, name in enumerate(balancing_units):
        i = thermal_names.index(name)
        up_map[i] = up[k]
        down_map[i] = down[k]
        g_actual[i] += up[k] - down[k]

    # Deviation from day-ahead schedule
    delta_g = g_actual - g_da
    delta_w = w_actual - w_da

    system_short = result_bal["system_short"]
    system_long = result_bal["system_long"]

    # --------------------------------------------------------
    # Conventional profits
    # DA revenue + balancing revenue - actual production cost
    # Provider balancing activation is paid at lambda_bal
    # Non-provider harmful/helpful deviation due to outage follows the settlement scheme
    # --------------------------------------------------------
    conv_profit_one = np.zeros(len(thermal_names))
    conv_profit_two = np.zeros(len(thermal_names))

    for i, name in enumerate(thermal_names):
        revenue_da = lambda_da * g_da[i]
        cost_actual = gen_cost[i] * g_actual[i]

        if name in balancing_units:
            # provider is explicitly activated in balancing market
            revenue_bal = lambda_bal * (up_map[i] - down_map[i])

            conv_profit_one[i] = revenue_da + revenue_bal - cost_actual
            print(f"{name} is a balancing provider. Up = {up_map[i]:.4f}, Down = {down_map[i]:.4f}, Balancing revenue = {revenue_bal:.4f}")
            conv_profit_two[i] = revenue_da + revenue_bal - cost_actual
        else:
            # non-provider deviation (e.g. outage unit) settled by imbalance rule
            delta = delta_g[i]

            revenue_imb_one = lambda_bal * delta
            price_two = settlement_price_two_price(delta, system_short, system_long, lambda_da, lambda_bal)
            revenue_imb_two = price_two * delta

            conv_profit_one[i] = revenue_da + revenue_imb_one - cost_actual
            conv_profit_two[i] = revenue_da + revenue_imb_two - cost_actual

    # --------------------------------------------------------
    # Wind profits
    # DA revenue + imbalance settlement - cost(=0)
    # --------------------------------------------------------
    wind_profit_one = np.zeros(len(wind_names))
    wind_profit_two = np.zeros(len(wind_names))

    for j, name in enumerate(wind_names):
        delta = delta_w[j]
        revenue_da = lambda_da * w_da[j]

        wind_profit_one[j] = revenue_da + lambda_bal * delta

        price_two = settlement_price_two_price(delta, system_short, system_long, lambda_da, lambda_bal)
        wind_profit_two[j] = revenue_da + price_two * delta

    return {
        "g_actual": g_actual,
        "w_actual": w_actual,
        "delta_g": delta_g,
        "delta_w": delta_w,
        "conv_profit_one": conv_profit_one,
        "conv_profit_two": conv_profit_two,
        "wind_profit_one": wind_profit_one,
        "wind_profit_two": wind_profit_two,
        "total_conv_profit_one": np.sum(conv_profit_one),
        "total_conv_profit_two": np.sum(conv_profit_two),
        "total_wind_profit_one": np.sum(wind_profit_one),
        "total_wind_profit_two": np.sum(wind_profit_two),
    }


# ============================================================
# 11. Printing helpers
# ============================================================
def print_day_ahead_summary(pack_da: dict, result_da: dict):
    print("\n" + "=" * 70)
    print(f"DAY-AHEAD MARKET (STEP 1) - HOUR {pack_da['hour']}")
    print("=" * 70)
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


def print_real_time_assumptions(rt_realization: dict):
    print("\n" + "=" * 70)
    print("REAL-TIME ASSUMPTIONS")
    print("=" * 70)
    print(f"Outage unit: {rt_realization['outage_unit']} (full outage)")
    print(f"Wind farms with -15% deviation: {rt_realization['wind_down_units']}")
    print(f"Wind farms with +10% deviation: {rt_realization['wind_up_units']}")
    print(f"Total system imbalance before balancing: {rt_realization['total_imbalance']:.4f} MW")
    if rt_realization['total_imbalance'] < 0:
        print("System is short before balancing.")
    elif rt_realization['total_imbalance'] > 0:
        print("System is long before balancing.")
    else:
        print("System is exactly balanced before balancing.")


def print_balancing_summary(pack_bal: dict, result_bal: dict):
    print("\n" + "=" * 70)
    print("BALANCING MARKET")
    print("=" * 70)
    print(f"Required balancing quantity: {result_bal['required_balancing']:.4f} MW")
    print(f"Balancing price: {result_bal['lambda_bal']:.4f} €/MWh")
    print(f"Load shedding: {result_bal['load_shed']:.4f} MW")

    print("\nBalancing activation:")
    for k, name in enumerate(pack_bal["balancing_units"]):
        print(
            f"{name:>4}: "
            f"up = {result_bal['up'][k]:10.4f} MW | "
            f"down = {result_bal['down'][k]:10.4f} MW | "
            f"up offer = {pack_bal['up_price'][k]:8.4f} | "
            f"down offer = {pack_bal['down_price'][k]:8.4f}"
        )


def print_profit_summary(pack_da: dict, profit_pack: dict):
    thermal_names = pack_da["thermal_names"]
    wind_names = pack_da["wind_names"]

    print("\n" + "=" * 70)
    print("TOTAL PROFITS: ONE-PRICE vs TWO-PRICE")
    print("=" * 70)

    print("\nConventional generators:")
    print(f"{'Unit':>6} | {'One-price (€)':>15} | {'Two-price (€)':>15}")
    print("-" * 44)
    for i, name in enumerate(thermal_names):
        print(f"{name:>6} | {profit_pack['conv_profit_one'][i]:15.4f} | {profit_pack['conv_profit_two'][i]:15.4f}")

    print("\nWind farms:")
    print(f"{'Unit':>6} | {'One-price (€)':>15} | {'Two-price (€)':>15}")
    print("-" * 44)
    for j, name in enumerate(wind_names):
        print(f"{name:>6} | {profit_pack['wind_profit_one'][j]:15.4f} | {profit_pack['wind_profit_two'][j]:15.4f}")

    print("\nTotals:")
    print(f"Total conventional profit (one-price): {profit_pack['total_conv_profit_one']:.4f} €")
    print(f"Total conventional profit (two-price): {profit_pack['total_conv_profit_two']:.4f} €")
    print(f"Total wind profit (one-price):         {profit_pack['total_wind_profit_one']:.4f} €")
    print(f"Total wind profit (two-price):         {profit_pack['total_wind_profit_two']:.4f} €")


def print_actual_outputs(pack_da: dict, profit_pack: dict):
    print("\n" + "=" * 70)
    print("ACTUAL REAL-TIME OUTPUTS")
    print("=" * 70)

    print("\nConventional actual outputs:")
    for name, val, dev in zip(pack_da["thermal_names"], profit_pack["g_actual"], profit_pack["delta_g"]):
        print(f"{name:>4}: actual = {val:10.4f} MW | deviation from DA = {dev:10.4f} MW")

    print("\nWind actual outputs:")
    for name, val, dev in zip(pack_da["wind_names"], profit_pack["w_actual"], profit_pack["delta_w"]):
        print(f"{name:>4}: actual = {val:10.4f} MW | deviation from DA = {dev:10.4f} MW")


# ============================================================
# 12. Main
# ============================================================
if __name__ == "__main__":
    # ----------------------------
    # User-selected / assumed settings
    # ----------------------------
    HOUR = 18
    SCENARIO = 30

    OUTAGE_UNIT = "g9"
    BALANCING_UNITS = ["g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8", "g10", "g11", "g12"]

    WIND_DOWN_UNITS = ["w1", "w2", "w3"]   # -15%
    WIND_UP_UNITS = ["w4", "w5", "w6"]     # +10%

    LOAD_CURTAILMENT_COST = 500.0

    # ----------------------------
    # Load data
    # ----------------------------
    base_data = get_base_market_data()
    wind_factor = load_wind_factor_from_mat(scenario=SCENARIO)
    wind_factor_hour = wind_factor[HOUR - 1, :]

    # ----------------------------
    # Step 1 day-ahead
    # ----------------------------
    input_da, pack_da = build_input_data_step1_single_hour(
        hour=HOUR,
        wind_factor_hour=wind_factor_hour
    )
    model_da = LPOptimizationProblem(input_da)
    model_da.run()
    result_da = extract_step1_results(model_da, pack_da)

    # ----------------------------
    # Real-time realization
    # ----------------------------
    rt_realization = build_real_time_realization(
        result_da=result_da,
        thermal_names=pack_da["thermal_names"],
        wind_names=pack_da["wind_names"],
        outage_unit=OUTAGE_UNIT,
        wind_down_units=WIND_DOWN_UNITS,
        wind_up_units=WIND_UP_UNITS,
        wind_down_rate=0.15,
        wind_up_rate=0.10,
    )

    # ----------------------------
    # Balancing market
    # ----------------------------
    input_bal, pack_bal = build_input_data_balancing(
        pack_da=pack_da,
        result_da=result_da,
        rt_realization=rt_realization,
        balancing_units=BALANCING_UNITS,
        load_curtailment_cost=LOAD_CURTAILMENT_COST,
    )
    model_bal = LPOptimizationProblem(input_bal)
    model_bal.run()
    result_bal = extract_balancing_results(
        model=model_bal,
        pack_bal=pack_bal,
        lambda_da=result_da["lambda_da"],
    )

    # ----------------------------
    # Profit calculation
    # ----------------------------
    profit_pack = compute_total_profits(
        pack_da=pack_da,
        result_da=result_da,
        rt_realization=rt_realization,
        pack_bal=pack_bal,
        result_bal=result_bal,
    )

    # ----------------------------
    # Print everything
    # ----------------------------
    print_day_ahead_summary(pack_da, result_da)
    print_real_time_assumptions(rt_realization)
    print_balancing_summary(pack_bal, result_bal)
    print_actual_outputs(pack_da, profit_pack)
    print_profit_summary(pack_da, profit_pack)

    plot_dispatch_and_profit_step5(pack_da, result_da, profit_pack)