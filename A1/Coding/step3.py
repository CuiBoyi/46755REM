

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from pathlib import Path
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
        self.model.Params.OutputFlag = 1
        self._build_variables()
        self._build_objective()
        self._build_constraints()
        self.model.update()

    def run(self):
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            self._save_results()
        else:
            raise RuntimeError(f"Optimization failed. Status = {self.model.status}")

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
# 4. Base market + network data
# ============================================================
def get_step3_data():
    # ---------- names ----------
    thermal_names = [f"g{i}" for i in range(1, 13)]
    wind_names = [f"w{i}" for i in range(1, 7)]
    load_names = [f"d{i}" for i in range(1, 18)]

    # ---------- generator data ----------
    # same order as your previous code
    Pmax = np.array([152, 152, 350, 591, 60, 155, 155, 400, 400, 300, 310, 350], dtype=float)

    # Step 1 usually uses dispatch model without commitment.
    # If you want pure Step-1 style, keep lower bounds = 0.
    Pmin = np.zeros(12, dtype=float)

    # Optional: if you want technical minimum output, uncomment this:
    # Pmin = np.array([30.4, 30.4, 75, 206.85, 12, 54.25, 54.25, 100, 100, 300, 108.5, 140], dtype=float)

    gen_cost = np.array([13.32, 13.32, 20.70, 20.93, 26.11, 10.52, 10.52, 6.02, 5.47, 0.00, 10.52, 10.89], dtype=float)

    # Table 1: node of each conventional unit
    gen_nodes = np.array([1, 2, 7, 13, 15, 15, 16, 18, 21, 22, 23, 23], dtype=int)

    # ---------- wind data ----------
    # Wind farm capacities from your original code
    Wcap = np.array([157.92, 162.83, 154.34, 126.48, 142.32, 147.68], dtype=float)
    wind_cost = np.zeros(6, dtype=float)

    # Assumed wind node mapping (important!)
    # This mapping was not shown in your screenshots.
    # Replace these nodes with the exact nodes from your assignment if provided elsewhere.
    wind_nodes = np.array([3, 5, 7, 16, 21, 23], dtype=int)

    # ---------- demand data ----------
    load_share_percent = np.array(
        [3.8, 3.4, 6.3, 2.6, 2.5, 4.8, 4.4, 6.0, 6.1, 6.8, 9.3, 6.8, 11.1, 3.5, 11.7, 6.4, 4.5],
        dtype=float,
    )

    # base bid from your previous Step 1/2 code
    base_load_bid = np.array(
        [24.445, 24.923, 21.456, 25.880, 26.000, 23.250, 23.7282, 21.815, 21.695,
         20.8586, 17.869, 20.858, 15.717, 24.804, 15.000, 21.3369, 23.608],
        dtype=float,
    )

    # Table 4: load node locations
    load_nodes = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 18, 19, 20], dtype=int)

    system_demand = np.array([
        1775.835, 1669.815, 1590.300, 1563.795, 1563.795, 1590.300,
        1961.370, 2279.430, 2517.975, 2544.480, 2544.480, 2517.975,
        2517.975, 2517.975, 2464.965, 2464.965, 2623.995, 2650.500,
        2650.500, 2544.480, 2411.955, 2199.915, 1934.865, 1669.815
    ], dtype=float)

    # ---------- network data ----------
    # Table 5: transmission lines
    line_from = np.array([
        1, 1, 1, 2, 2, 3, 3, 4, 5, 6,
        7, 8, 8, 9, 9, 10, 10,
        11, 11, 12, 12, 13, 14, 15, 15, 15, 16, 16, 17, 17, 18, 19, 20, 21
    ], dtype=int)

    line_to = np.array([
        2, 3, 5, 4, 6, 9, 24, 9, 10, 10,
        8, 9, 10, 11, 12, 11, 12,
        13, 14, 13, 23, 23, 16, 16, 21, 24, 17, 19, 18, 22, 21, 20, 23, 22
    ], dtype=int)

    reactance = np.array([
        0.0146, 0.2253, 0.0907, 0.1356, 0.2050, 0.1271, 0.0840, 0.1110, 0.0940, 0.0642,
        0.0652, 0.1762, 0.1762, 0.0840, 0.0840, 0.0840, 0.0840,
        0.0488, 0.0426, 0.0488, 0.0985, 0.0884, 0.0594, 0.0172, 0.0249, 0.0529, 0.0263,
        0.0234, 0.0143, 0.1069, 0.0132, 0.0203, 0.0112, 0.0692
    ], dtype=float)

    line_cap = np.array([
        175, 175, 350, 175, 175, 175, 400, 175, 350, 175,
        350, 175, 175, 400, 400, 400, 400,
        500, 500, 500, 500, 500, 500, 500, 1000, 500, 500,
        500, 500, 500, 1000, 1000, 1000, 500
    ], dtype=float)

    n_nodes = 24
    ref_node = 1

    return {
        "thermal_names": thermal_names,
        "wind_names": wind_names,
        "load_names": load_names,
        "Pmax": Pmax,
        "Pmin": Pmin,
        "gen_cost": gen_cost,
        "gen_nodes": gen_nodes,
        "Wcap": Wcap,
        "wind_cost": wind_cost,
        "wind_nodes": wind_nodes,
        "load_share_percent": load_share_percent,
        "base_load_bid": base_load_bid,
        "load_nodes": load_nodes,
        "system_demand": system_demand,
        "line_from": line_from,
        "line_to": line_to,
        "reactance": reactance,
        "line_cap": line_cap,
        "n_nodes": n_nodes,
        "ref_node": ref_node,
    }


# ============================================================
# 5. Hourly bids
# ============================================================
def build_hourly_load_bids(system_demand: np.ndarray, base_load_bid: np.ndarray) -> np.ndarray:
    dmin = system_demand.min()
    dmax = system_demand.max()

    if abs(dmax - dmin) < 1e-12:
        multiplier = np.ones_like(system_demand)
    else:
        normalized = (system_demand - dmin) / (dmax - dmin)
        multiplier = 0.90 + 0.25 * normalized

    hourly_bids = multiplier[:, None] * base_load_bid[None, :]
    return hourly_bids


# ============================================================
# 6. Wind profile
# ============================================================
def build_default_wind_profile(T: int = 24, n_wind: int = 6) -> np.ndarray:
    hours = np.arange(T)
    profile = 0.55 + 0.20 * np.sin(2 * np.pi * (hours - 5) / 24.0)
    profile = np.clip(profile, 0.15, 0.95)

    wind_factor = np.tile(profile[:, None], (1, n_wind))
    zone_scale = np.array([1.00, 1.03, 0.98, 0.93, 0.97, 1.01], dtype=float)
    wind_factor *= zone_scale[None, :]
    wind_factor = np.clip(wind_factor, 0.0, 1.0)
    return wind_factor


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
# 7. Build Step 3 LP input (1 hour, no storage, with network)
# ============================================================
def build_input_data_step3_one_hour(
    hour: int = 18,
    wind_factor: np.ndarray | None = None,
    line_capacity_scale: float = 1.0,
    modified_lines: dict | None = None,
    use_flow_variables: bool = True,
) -> tuple[LPInputData, dict]:
    """
    Build 1-hour nodal market clearing model with DC network constraints.

    Parameters
    ----------
    hour : int
        1..24
    wind_factor : np.ndarray or None
        Shape (24,6)
    line_capacity_scale : float
        Scale all line capacities
    modified_lines : dict or None
        Example: {24: 0.5, 30: 1.2}
        meaning capacity of line index 24 is multiplied by 0.5, etc.
    use_flow_variables : bool
        True = include line flow variables explicitly
    """
    if not (1 <= hour <= 24):
        raise ValueError("hour must be between 1 and 24")

    data = get_step3_data()

    thermal_names = data["thermal_names"]
    wind_names = data["wind_names"]
    load_names = data["load_names"]

    Pmax = data["Pmax"]
    Pmin = data["Pmin"]
    gen_cost = data["gen_cost"]
    gen_nodes = data["gen_nodes"]

    Wcap = data["Wcap"]
    wind_cost = data["wind_cost"]
    wind_nodes = data["wind_nodes"]

    load_share_percent = data["load_share_percent"]
    base_load_bid = data["base_load_bid"]
    load_nodes = data["load_nodes"]

    system_demand = data["system_demand"]

    line_from = data["line_from"].copy()
    line_to = data["line_to"].copy()
    reactance = data["reactance"].copy()
    line_cap = data["line_cap"].copy()

    n_nodes = data["n_nodes"]
    ref_node = data["ref_node"]

    ng = len(thermal_names)
    nw = len(wind_names)
    nd = len(load_names)
    nl = len(line_from)

    # ----- wind -----
    if wind_factor is None:
        wind_factor = build_default_wind_profile(T=24, n_wind=nw)
    wind_factor = np.asarray(wind_factor, dtype=float)

    if wind_factor.shape != (24, nw):
        raise ValueError(f"wind_factor must have shape (24,{nw}), got {wind_factor.shape}")

    h = hour - 1  # zero-based hour index

    Wmax = wind_factor[h, :] * Wcap
    Dmax = system_demand[h] * load_share_percent / 100.0
    load_bid_hourly = build_hourly_load_bids(system_demand, base_load_bid)
    load_bid = load_bid_hourly[h, :]

    # ----- line capacity modification -----
    line_cap *= line_capacity_scale
    if modified_lines is not None:
        for ell, factor in modified_lines.items():
            if not (0 <= ell < nl):
                raise ValueError(f"modified_lines contains invalid line index {ell}")
            line_cap[ell] *= factor

    # ========================================================
    # Variable indexing
    # ========================================================
    var_names = []
    counter = 0

    idx_g = np.empty(ng, dtype=int)
    idx_w = np.empty(nw, dtype=int)
    idx_d = np.empty(nd, dtype=int)
    idx_theta = np.empty(n_nodes, dtype=int)

    for i in range(ng):
        idx_g[i] = counter
        var_names.append(f"g_{i+1}")
        counter += 1

    for i in range(nw):
        idx_w[i] = counter
        var_names.append(f"w_{i+1}")
        counter += 1

    for i in range(nd):
        idx_d[i] = counter
        var_names.append(f"d_{i+1}")
        counter += 1

    for n in range(n_nodes):
        idx_theta[n] = counter
        var_names.append(f"theta_{n+1}")
        counter += 1

    if use_flow_variables:
        idx_f = np.empty(nl, dtype=int)
        for ell in range(nl):
            idx_f[ell] = counter
            var_names.append(f"f_{line_from[ell]}_{line_to[ell]}")
            counter += 1
    else:
        idx_f = None

    nv = counter

    # ========================================================
    # Bounds
    # ========================================================
    lb = np.full(nv, -np.inf, dtype=float)
    ub = np.full(nv,  np.inf, dtype=float)

    lb[idx_g] = Pmin
    ub[idx_g] = Pmax

    lb[idx_w] = 0.0
    ub[idx_w] = Wmax

    lb[idx_d] = 0.0
    ub[idx_d] = Dmax

    # voltage angles
    # Usually left free, except reference node fixed by equality constraint.
    # Optional numerical bounds:
    lb[idx_theta] = -GRB.INFINITY
    ub[idx_theta] =  GRB.INFINITY

    if use_flow_variables:
        lb[idx_f] = -line_cap
        ub[idx_f] =  line_cap

    # ========================================================
    # Objective
    # maximize sum(load value - generation cost)
    # ========================================================
    obj = np.zeros(nv, dtype=float)
    obj[idx_g] = -gen_cost
    obj[idx_w] = -wind_cost
    obj[idx_d] = load_bid
    # theta and flow have zero objective

    # ========================================================
    # Constraints
    # 1) nodal balance (24 nodes)
    # 2) DC flow equations (for each line)
    # 3) reference angle
    # 4) line limits if flows are not explicit
    # ========================================================
    constr_names = []

    # nodal balance
    for n in range(1, n_nodes + 1):
        constr_names.append(f"balance_node{n}")

    # flow equations
    if use_flow_variables:
        for ell in range(nl):
            constr_names.append(f"flowdef_l{ell+1}_{line_from[ell]}_{line_to[ell]}")

    # reference angle
    constr_names.append(f"theta_ref_node{ref_node}")

    # if no explicit flow variables, need direct angle-based line limit constraints
    if not use_flow_variables:
        for ell in range(nl):
            constr_names.append(f"line_ub_l{ell+1}_{line_from[ell]}_{line_to[ell]}")
            constr_names.append(f"line_lb_l{ell+1}_{line_from[ell]}_{line_to[ell]}")

    nc = len(constr_names)

    A = np.zeros((nc, nv), dtype=float)
    rhs = np.zeros(nc, dtype=float)
    sense = np.empty(nc, dtype=object)

    row = 0

    # --------------------------------------------------------
    # 1) Nodal balance:
    # generation + wind - demand - net export = 0
    #
    # Using explicit line flows:
    # at node n:
    # sum g_n + sum w_n - sum d_n
    #   - sum_{lines outgoing from n} f_l
    #   + sum_{lines incoming to n} f_l = 0
    # --------------------------------------------------------
    for n in range(1, n_nodes + 1):
        # conventional generation at node n
        for i in range(ng):
            if gen_nodes[i] == n:
                A[row, idx_g[i]] = 1.0

        # wind at node n
        for i in range(nw):
            if wind_nodes[i] == n:
                A[row, idx_w[i]] = 1.0

        # load at node n
        for j in range(nd):
            if load_nodes[j] == n:
                A[row, idx_d[j]] = -1.0

        if use_flow_variables:
            for ell in range(nl):
                if line_from[ell] == n:
                    A[row, idx_f[ell]] += -1.0
                elif line_to[ell] == n:
                    A[row, idx_f[ell]] += 1.0
        else:
            # substitute f = (theta_i - theta_j)/x directly
            for ell in range(nl):
                i_node = line_from[ell]
                j_node = line_to[ell]
                b = 1.0 / reactance[ell]

                if n == i_node:
                    A[row, idx_theta[i_node - 1]] += -b
                    A[row, idx_theta[j_node - 1]] +=  b
                elif n == j_node:
                    A[row, idx_theta[i_node - 1]] +=  b
                    A[row, idx_theta[j_node - 1]] += -b

        rhs[row] = 0.0
        sense[row] = GRB.EQUAL
        row += 1

    # --------------------------------------------------------
    # 2) Flow definitions:
    # f_l - (theta_i - theta_j)/x = 0
    # --------------------------------------------------------
    if use_flow_variables:
        for ell in range(nl):
            i_node = line_from[ell]
            j_node = line_to[ell]
            b = 1.0 / reactance[ell]

            A[row, idx_f[ell]] = 1.0
            A[row, idx_theta[i_node - 1]] = -b
            A[row, idx_theta[j_node - 1]] =  b

            rhs[row] = 0.0
            sense[row] = GRB.EQUAL
            row += 1

    # --------------------------------------------------------
    # 3) Reference angle
    # theta_ref = 0
    # --------------------------------------------------------
    A[row, idx_theta[ref_node - 1]] = 1.0
    rhs[row] = 0.0
    sense[row] = GRB.EQUAL
    row += 1

    # --------------------------------------------------------
    # 4) Direct line limits if no explicit f variables
    # (theta_i - theta_j)/x <= Fmax
    # -(theta_i - theta_j)/x <= Fmax
    # --------------------------------------------------------
    if not use_flow_variables:
        for ell in range(nl):
            i_node = line_from[ell]
            j_node = line_to[ell]
            b = 1.0 / reactance[ell]
            Fmax = line_cap[ell]

            # upper bound
            A[row, idx_theta[i_node - 1]] =  b
            A[row, idx_theta[j_node - 1]] = -b
            rhs[row] = Fmax
            sense[row] = GRB.LESS_EQUAL
            row += 1

            # lower bound
            A[row, idx_theta[i_node - 1]] = -b
            A[row, idx_theta[j_node - 1]] =  b
            rhs[row] = Fmax
            sense[row] = GRB.LESS_EQUAL
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
        model_name=f"step3_nodal_market_hour_{hour}",
    )

    data_pack = {
        "hour": hour,
        "ng": ng,
        "nw": nw,
        "nd": nd,
        "nl": nl,
        "n_nodes": n_nodes,
        "ref_node": ref_node,
        "thermal_names": thermal_names,
        "wind_names": wind_names,
        "load_names": load_names,
        "Pmax": Pmax,
        "Pmin": Pmin,
        "gen_cost": gen_cost,
        "gen_nodes": gen_nodes,
        "Wcap": Wcap,
        "wind_cost": wind_cost,
        "wind_nodes": wind_nodes,
        "load_share_percent": load_share_percent,
        "base_load_bid": base_load_bid,
        "load_nodes": load_nodes,
        "system_demand": system_demand,
        "load_bid": load_bid,
        "Dmax": Dmax,
        "Wmax": Wmax,
        "line_from": line_from,
        "line_to": line_to,
        "reactance": reactance,
        "line_cap": line_cap,
        "use_flow_variables": use_flow_variables,
        "index": {
            "g": idx_g,
            "w": idx_w,
            "d": idx_d,
            "theta": idx_theta,
            "f": idx_f,
        },
    }

    return input_data, data_pack


# ============================================================
# 8. Post-processing
# ============================================================
def extract_step3_results(model: LPOptimizationProblem, data_pack: dict) -> dict:
    idx = data_pack["index"]
    ng = data_pack["ng"]
    nw = data_pack["nw"]
    nd = data_pack["nd"]
    nl = data_pack["nl"]
    n_nodes = data_pack["n_nodes"]

    gen_cost = data_pack["gen_cost"]
    load_bid = data_pack["load_bid"]
    thermal_names = data_pack["thermal_names"]
    wind_names = data_pack["wind_names"]
    load_names = data_pack["load_names"]
    line_from = data_pack["line_from"]
    line_to = data_pack["line_to"]
    reactance = data_pack["reactance"]
    use_flow_variables = data_pack["use_flow_variables"]

    x = model.results.x
    duals = model.results.duals
    constr_names = model.results.constr_names

    g = x[idx["g"]]
    w = x[idx["w"]]
    d = x[idx["d"]]
    theta = x[idx["theta"]]

    if use_flow_variables:
        f = x[idx["f"]]
    else:
        f = np.zeros(nl, dtype=float)
        for ell in range(nl):
            i_node = line_from[ell]
            j_node = line_to[ell]
            f[ell] = (theta[i_node - 1] - theta[j_node - 1]) / reactance[ell]

    # nodal prices from node-balance duals
    nodal_price = np.zeros(n_nodes, dtype=float)
    for n in range(1, n_nodes + 1):
        cname = f"balance_node{n}"
        row = constr_names.index(cname)
        # same sign convention as your previous model:
        # balance written as supply - demand - export = 0 in a welfare-maximization LP
        nodal_price[n - 1] = -duals[row]

    # profits
    thermal_profit = np.zeros(ng, dtype=float)
    for i in range(ng):
        node = data_pack["gen_nodes"][i]
        thermal_profit[i] = g[i] * (nodal_price[node - 1] - gen_cost[i])

    wind_profit = np.zeros(nw, dtype=float)
    for i in range(nw):
        node = data_pack["wind_nodes"][i]
        wind_profit[i] = w[i] * nodal_price[node - 1]

    load_utility = np.zeros(nd, dtype=float)
    for j in range(nd):
        node = data_pack["load_nodes"][j]
        load_utility[j] = d[j] * (load_bid[j] - nodal_price[node - 1])

    operating_cost = np.sum(g * gen_cost)
    load_value = np.sum(d * load_bid)
    social_welfare = load_value - operating_cost

    # congestion info
    line_cap = data_pack["line_cap"]
    line_loading = np.abs(f) / np.maximum(line_cap, 1e-12)
    congested = np.isclose(np.abs(f), line_cap, atol=1e-5)

    return {
        "g": g,
        "w": w,
        "d": d,
        "theta": theta,
        "f": f,
        "nodal_price": nodal_price,
        "thermal_profit": thermal_profit,
        "wind_profit": wind_profit,
        "load_utility": load_utility,
        "operating_cost": operating_cost,
        "load_value": load_value,
        "social_welfare": social_welfare,
        "line_loading": line_loading,
        "congested": congested,
        "thermal_names": thermal_names,
        "wind_names": wind_names,
        "load_names": load_names,
    }


# ============================================================
# 9. Reporting
# ============================================================
def print_step3_summary(result_pack: dict, data_pack: dict, title: str = "STEP 3 CASE"):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    print(f"\nHour analyzed: {data_pack['hour']}")
    print(f"Total operating cost: {result_pack['operating_cost']:.4f} €")
    print(f"Total load value:     {result_pack['load_value']:.4f} €")
    print(f"Total social welfare: {result_pack['social_welfare']:.4f} €")

    print("\n---------------- NODAL PRICES ----------------")
    for n, price in enumerate(result_pack["nodal_price"], start=1):
        print(f"Node {n:>2}: {price:>10.4f} €/MWh")

    print("\n---------------- CONVENTIONAL GENERATION ----------------")
    for i, name in enumerate(data_pack["thermal_names"]):
        node = data_pack["gen_nodes"][i]
        print(
            f"{name:>3} @ node {node:>2} | "
            f"dispatch = {result_pack['g'][i]:>9.4f} MW | "
            f"profit = {result_pack['thermal_profit'][i]:>10.4f} €"
        )

    print("\n---------------- WIND GENERATION ----------------")
    for i, name in enumerate(data_pack["wind_names"]):
        node = data_pack["wind_nodes"][i]
        print(
            f"{name:>3} @ node {node:>2} | "
            f"dispatch = {result_pack['w'][i]:>9.4f} MW | "
            f"profit = {result_pack['wind_profit'][i]:>10.4f} €"
        )

    print("\n---------------- LOAD SERVED ----------------")
    for j, name in enumerate(data_pack["load_names"]):
        node = data_pack["load_nodes"][j]
        print(
            f"{name:>3} @ node {node:>2} | "
            f"served = {result_pack['d'][j]:>9.4f} MW | "
            f"utility = {result_pack['load_utility'][j]:>10.4f} €"
        )

    print("\n---------------- LINE FLOWS ----------------")
    for ell in range(data_pack["nl"]):
        i_node = data_pack["line_from"][ell]
        j_node = data_pack["line_to"][ell]
        cap = data_pack["line_cap"][ell]
        flow = result_pack["f"][ell]
        loading = result_pack["line_loading"][ell] * 100.0
        congested = result_pack["congested"][ell]

        flag = " <-- congested" if congested else ""
        print(
            f"Line {ell+1:>2}: {i_node:>2} -> {j_node:>2} | "
            f"flow = {flow:>10.4f} MW | "
            f"cap = {cap:>7.2f} MW | "
            f"loading = {loading:>7.2f}%{flag}"
        )


# ============================================================
# 10. Sensitivity analysis on transmission capacity
# ============================================================
def run_line_sensitivity(
    hour: int,
    wind_factor: np.ndarray,
    target_line_index: int,
    multipliers: list[float],
):
    """
    target_line_index is zero-based.
    """
    outputs = []

    data = get_step3_data()
    i_node = data["line_from"][target_line_index]
    j_node = data["line_to"][target_line_index]
    base_cap = data["line_cap"][target_line_index]

    print("\n" + "=" * 70)
    print(f"SENSITIVITY ON LINE {target_line_index+1}: {i_node} -> {j_node}")
    print("=" * 70)

    for mult in multipliers:
        input_data, pack = build_input_data_step3_one_hour(
            hour=hour,
            wind_factor=wind_factor,
            modified_lines={target_line_index: mult},
            use_flow_variables=True,
        )
        model = LPOptimizationProblem(input_data)
        model.run()
        result = extract_step3_results(model, pack)

        outputs.append({
            "multiplier": mult,
            "line_cap": base_cap * mult,
            "social_welfare": result["social_welfare"],
            "min_nodal_price": np.min(result["nodal_price"]),
            "max_nodal_price": np.max(result["nodal_price"]),
            "price_spread": np.max(result["nodal_price"]) - np.min(result["nodal_price"]),
            "target_line_flow": result["f"][target_line_index],
            "target_line_loading": result["line_loading"][target_line_index],
        })

    print("mult | line cap | welfare | min price | max price | spread | line flow | loading")
    for out in outputs:
        print(
            f"{out['multiplier']:>4.2f} | "
            f"{out['line_cap']:>8.2f} | "
            f"{out['social_welfare']:>10.2f} | "
            f"{out['min_nodal_price']:>9.4f} | "
            f"{out['max_nodal_price']:>9.4f} | "
            f"{out['price_spread']:>8.4f} | "
            f"{out['target_line_flow']:>9.4f} | "
            f"{100*out['target_line_loading']:>7.2f}%"
        )

    return outputs


# ============================================================
# 11. Main
# ============================================================
if __name__ == "__main__":
    # choose one hour from Step 1 / Step 3
    hour = 18

    # wind profile
    try:
        wind_factor = load_wind_factor_from_mat(scenario=30)
    except Exception as e:
        print(f"Wind file not found or failed to load: {e}")
        print("Using default wind profile instead.")
        wind_factor = build_default_wind_profile(T=24, n_wind=6)

    # --------------------------------------------------------
    # Base Step 3 case
    # --------------------------------------------------------
    input_data, pack = build_input_data_step3_one_hour(
        hour=hour,
        wind_factor=wind_factor,
        line_capacity_scale=1.0,
        modified_lines=None,
        use_flow_variables=True,
    )

    model = LPOptimizationProblem(input_data)
    model.run()

    result = extract_step3_results(model, pack)
    print_step3_summary(result, pack, title="STEP 3: NODAL MARKET CLEARING WITH NETWORK CONSTRAINTS")

    # --------------------------------------------------------
    # Example sensitivity:
    # choose one important line and shrink/enlarge its capacity
    # Example here: line 25 in the list (0-based index 24), i.e. 15 -> 21
    # Change this line index if your teacher asks for another line.
    # --------------------------------------------------------
    sensitivity_outputs = run_line_sensitivity(
        hour=hour,
        wind_factor=wind_factor,
        target_line_index=24,   # line 25 in human counting: 15 -> 21
        multipliers=[0.5, 0.75, 1.0, 1.25, 1.5]
    )
    