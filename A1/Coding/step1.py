
import gurobipy as gp
from gurobipy import GRB


class Expando(object):
    pass


class LP_InputData:
    def __init__(
        self,
        VARIABLES: list[str],
        CONSTRAINTS: list[str],
        objective_coeff: dict[str, float],
        constraints_coeff: dict[str, dict[str, float]],
        constraints_rhs: dict[str, float],
        constraints_sense: dict[str, int],
        objective_sense: int,
        model_name: str
    ):
        self.VARIABLES = VARIABLES
        self.CONSTRAINTS = CONSTRAINTS
        self.objective_coeff = objective_coeff
        self.constraints_coeff = constraints_coeff
        self.constraints_rhs = constraints_rhs
        self.constraints_sense = constraints_sense
        self.objective_sense = objective_sense
        self.model_name = model_name


class LP_OptimizationProblem:
    def __init__(self, input_data: LP_InputData):
        self.data = input_data
        self.results = Expando()
        self._build_model()

    def _build_variables(self):
        self.variables = {
            v: self.model.addVar(lb=0, name=v)
            for v in self.data.VARIABLES
        }

    def _build_constraints(self):
        self.constraints = {
            c: self.model.addLConstr(
                gp.quicksum(
                    self.data.constraints_coeff[c][v] * self.variables[v]
                    for v in self.data.VARIABLES
                ),
                self.data.constraints_sense[c],
                self.data.constraints_rhs[c],
                name=c
            )
            for c in self.data.CONSTRAINTS
        }

    def _build_objective_function(self):
        objective = gp.quicksum(
            self.data.objective_coeff[v] * self.variables[v]
            for v in self.data.VARIABLES
        )
        self.model.setObjective(objective, self.data.objective_sense)

    def _build_model(self):
        self.model = gp.Model(name=self.data.model_name)
        self._build_variables()
        self._build_objective_function()
        self._build_constraints()
        self.model.update()

    def _save_results(self):
        self.results.objective_value = self.model.ObjVal
        self.results.variables = {v.VarName: v.X for v in self.model.getVars()}
        self.results.optimal_duals = {c.ConstrName: c.Pi for c in self.model.getConstrs()}

    def run(self):
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            self._save_results()
        else:
            print(f"Optimization of {self.model.ModelName} was not successful. Status = {self.model.status}")

    def display_results(self):
        print("\n------------------- RESULTS -------------------")
        print(f"Optimal objective value (Social Welfare): {self.results.objective_value:.4f}")

        print("\n--- Variables ---")
        for key, value in self.results.variables.items():
            print(f"{key}: {value:.4f}")

        print("\n--- Dual variables ---")
        for key, value in self.results.optimal_duals.items():
            print(f"{key}: {value:.4f}")


def build_input_data_for_hour(hour: int) -> tuple[LP_InputData, dict]:
    # =========================
    # 1) Sets
    # =========================
    thermal_units = [f"g{i}" for i in range(1, 13)]
    wind_units = [f"w{i}" for i in range(1, 7)]
    load_units = [f"d{i}" for i in range(1, 18)]

    VARIABLES = thermal_units + wind_units + load_units

    # =========================
    # 2) Conventional generators
    # =========================
    Pmax = {
        "g1": 152,
        "g2": 152,
        "g3": 350,
        "g4": 591,
        "g5": 60,
        "g6": 155,
        "g7": 155,
        "g8": 400,
        "g9": 400,
        "g10": 300,
        "g11": 310,
        "g12": 350
    }

    Pmin = {
        "g1": 30.4,
        "g2": 30.4,
        "g3": 75,
        "g4": 206.85,
        "g5": 12,
        "g6": 54.25,
        "g7": 54.25,
        "g8": 100,
        "g9": 100,
        "g10": 300,
        "g11": 108.5,
        "g12": 140
    }

    Pmin = {
        "g1": 0,
        "g2": 0,
        "g3": 0,
        "g4": 0,
        "g5": 0,
        "g6": 0,
        "g7": 0,
        "g8": 0,
        "g9": 0,
        "g10": 0,
        "g11": 0,
        "g12": 0
    }

    gen_cost = {
        "g1": 13.32,
        "g2": 13.32,
        "g3": 20.70,
        "g4": 20.93,
        "g5": 26.11,
        "g6": 10.52,
        "g7": 10.52,
        "g8": 6.02,
        "g9": 5.47,
        "g10": 0.00,
        "g11": 10.52,
        "g12": 10.89
    }

    # =========================
    # 3) Wind farms
    # =========================
    Wmax = {w: 200.0 for w in wind_units}
    Wmax = {
        "w1": 157.92,
        "w2": 162.83,
        "w3": 154.34,
        "w4": 126.48,
        "w5": 142.32,
        "w6": 147.68}
    wind_cost = {w: 0.0 for w in wind_units}

    # =========================
    # 4) Loads
    # =========================
    load_share_percent = {
        "d1": 3.8,
        "d2": 3.4,
        "d3": 6.3,
        "d4": 2.6,
        "d5": 2.5,
        "d6": 4.8,
        "d7": 4.4,
        "d8": 6.0,
        "d9": 6.1,
        "d10": 6.8,
        "d11": 9.3,
        "d12": 6.8,
        "d13": 11.1,
        "d14": 3.5,
        "d15": 11.7,
        "d16": 6.4,
        "d17": 4.5
    }

    load_bid_price = {
        "d1": 24.445,
        "d2": 24.923,
        "d3": 21.456,
        "d4": 25.880,
        "d5": 26.000,
        "d6": 23.25,
        "d7": 23.7282,
        "d8": 21.815,
        "d9": 21.695,
        "d10": 20.8586,
        "d11": 17.869,
        "d12": 20.858,
        "d13": 15.717,
        "d14": 24.804,
        "d15": 15.000,
        "d16": 21.3369,
        "d17": 23.608
    }

    # =========================
    # 5) Hourly system demand
    # =========================
    system_demand = {
        1: 1775.835,
        2: 1669.815,
        3: 1590.300,
        4: 1563.795,
        5: 1563.795,
        6: 1590.300,
        7: 1961.370,
        8: 2279.430,
        9: 2517.975,
        10: 2544.480,
        11: 2544.480,
        12: 2517.975,
        13: 2517.975,
        14: 2517.975,
        15: 2464.965,
        16: 2464.965,
        17: 2623.995,
        18: 2650.500,
        19: 2650.500,
        20: 2544.480,
        21: 2411.955,
        22: 2199.915,
        23: 1934.865,
        24: 1669.815
    }

    if hour not in system_demand:
        raise ValueError("hour must be an integer between 1 and 24.")

    system_load = system_demand[hour]

    # Each load upper bound for the selected hour
    Dmax = {
        d: system_load * load_share_percent[d] / 100.0
        for d in load_units
    }

    # =========================
    # 6) Objective coefficients
    # =========================
    objective_coeff = {}

    for g in thermal_units:
        objective_coeff[g] = -gen_cost[g]

    for w in wind_units:
        objective_coeff[w] = -wind_cost[w]   # = 0

    for d in load_units:
        objective_coeff[d] = load_bid_price[d]

    # =========================
    # 7) Constraints
    # =========================
    CONSTRAINTS = []

    for g in thermal_units:
        CONSTRAINTS.append(f"{g}_min")
        CONSTRAINTS.append(f"{g}_max")

    for w in wind_units:
        CONSTRAINTS.append(f"{w}_max")

    for d in load_units:
        CONSTRAINTS.append(f"{d}_max")

    CONSTRAINTS.append("balance")

    constraints_coeff = {
        c: {v: 0.0 for v in VARIABLES}
        for c in CONSTRAINTS
    }

    # Generator min/max
    for g in thermal_units:
        constraints_coeff[f"{g}_min"][g] = 1.0
        constraints_coeff[f"{g}_max"][g] = 1.0

    # Wind max
    for w in wind_units:
        constraints_coeff[f"{w}_max"][w] = 1.0

    # Demand max
    for d in load_units:
        constraints_coeff[f"{d}_max"][d] = 1.0

    # Power balance: generation + wind = demand
    for g in thermal_units:
        constraints_coeff["balance"][g] = 1.0

    for w in wind_units:
        constraints_coeff["balance"][w] = 1.0

    for d in load_units:
        constraints_coeff["balance"][d] = -1.0

    constraints_rhs = {}

    for g in thermal_units:
        constraints_rhs[f"{g}_min"] = Pmin[g]
        constraints_rhs[f"{g}_max"] = Pmax[g]

    for w in wind_units:
        constraints_rhs[f"{w}_max"] = Wmax[w]

    for d in load_units:
        constraints_rhs[f"{d}_max"] = Dmax[d]

    constraints_rhs["balance"] = 0.0

    constraints_sense = {}

    for g in thermal_units:
        constraints_sense[f"{g}_min"] = GRB.GREATER_EQUAL
        constraints_sense[f"{g}_max"] = GRB.LESS_EQUAL

    for w in wind_units:
        constraints_sense[f"{w}_max"] = GRB.LESS_EQUAL

    for d in load_units:
        constraints_sense[f"{d}_max"] = GRB.LESS_EQUAL

    constraints_sense["balance"] = GRB.EQUAL

    input_data = LP_InputData(
        VARIABLES=VARIABLES,
        CONSTRAINTS=CONSTRAINTS,
        objective_coeff=objective_coeff,
        constraints_coeff=constraints_coeff,
        constraints_rhs=constraints_rhs,
        constraints_sense=constraints_sense,
        objective_sense=GRB.MAXIMIZE,
        model_name=f"market_clearing_hour_{hour}"
    )

    data_pack = {
        "hour": hour,
        "system_load": system_load,
        "thermal_units": thermal_units,
        "wind_units": wind_units,
        "load_units": load_units,
        "Pmin": Pmin,
        "Pmax": Pmax,
        "gen_cost": gen_cost,
        "Wmax": Wmax,
        "load_bid_price": load_bid_price,
        "Dmax": Dmax
    }

    return input_data, data_pack


def print_market_results(model: LP_OptimizationProblem, data_pack: dict):
    thermal_units = data_pack["thermal_units"]
    wind_units = data_pack["wind_units"]
    load_units = data_pack["load_units"]
    gen_cost = data_pack["gen_cost"]
    load_bid_price = data_pack["load_bid_price"]
    Dmax = data_pack["Dmax"]
    Pmin = data_pack["Pmin"]
    Pmax = data_pack["Pmax"]
    hour = data_pack["hour"]
    system_load = data_pack["system_load"]

    results = model.results
    variables = results.variables
    duals = results.optimal_duals

    mcp = -duals["balance"]

    total_operating_cost = sum(
        gen_cost[g] * variables[g] for g in thermal_units
    )

    total_load_value = sum(
        load_bid_price[d] * variables[d] for d in load_units
    )

    social_welfare = results.objective_value

    print("\n============================================================")
    print("MARKET CLEARING OUTCOMES")
    print("============================================================")
    print(f"Hour: {hour}")
    print(f"System demand: {system_load:.3f} MW")
    print(f"Market-Clearing Price (MCP): {mcp:.4f} €/MWh")
    print(f"Total Operating Cost:        {total_operating_cost:.4f} €")
    print(f"Total Load Value:            {total_load_value:.4f} €")
    print(f"Total Social Welfare:        {social_welfare:.4f} €")

    print("\n============================================================")
    print("DISPATCH OF CONVENTIONAL UNITS")
    print("============================================================")
    for g in thermal_units:
        q = variables[g]
        status = ""
        tol = 1e-6
        if abs(q - Pmin[g]) < tol and abs(q - Pmax[g]) < tol:
            status = "Fixed"
        elif abs(q - Pmin[g]) < tol:
            status = "At Pmin"
        elif abs(q - Pmax[g]) < tol:
            status = "At Pmax"
        else:
            status = "Marginal/Internal"
        print(f"{g}: dispatch = {q:.4f} MW | cost = {gen_cost[g]:.4f} €/MWh | {status}")

    print("\n============================================================")
    print("DISPATCH OF WIND FARMS")
    print("============================================================")
    for w in wind_units:
        q = variables[w]
        status = "At Wmax" if abs(q - 200.0) < 1e-6 else "Internal"
        print(f"{w}: dispatch = {q:.4f} MW | cost = 0.0000 €/MWh | {status}")

    print("\n============================================================")
    print("LOAD ACCEPTANCE")
    print("============================================================")
    for d in load_units:
        q = variables[d]
        bid = load_bid_price[d]
        dmax = Dmax[d]
        if abs(q) < 1e-6:
            status = "Rejected"
        elif abs(q - dmax) < 1e-6:
            status = "Accepted (Full)"
        else:
            status = "Accepted (Partial/Marginal)"
        print(f"{d}: consumption = {q:.4f} MW / {dmax:.4f} MW | bid = {bid:.4f} €/MWh | {status}")

    print("\n============================================================")
    print("PRODUCER PROFITS")
    print("============================================================")
    for g in thermal_units:
        q = variables[g]
        profit = q * (mcp - gen_cost[g])
        print(f"{g}: profit = {profit:.4f} €")

    for w in wind_units:
        q = variables[w]
        profit = q * mcp
        print(f"{w}: profit = {profit:.4f} €")

    print("\n============================================================")
    print("DEMAND UTILITIES")
    print("============================================================")
    for d in load_units:
        q = variables[d]
        utility = q * (load_bid_price[d] - mcp)
        print(f"{d}: utility = {utility:.4f} €")

    print("\n============================================================")
    print("KKT CONDITIONS VERIFICATION (Economic Interpretation)")
    print("============================================================")
    print("For generators:")
    print("  - If Pmin < g < Pmax, then the unit is marginal and MCP should equal its cost.")
    print("  - If g = Pmax, then usually MCP >= cost.")
    print("  - If g = Pmin, then the lower bound is binding.")
    print("\nFor demands:")
    print("  - If 0 < d < Dmax, then the load is marginal and bid price should equal MCP.")
    print("  - If d = Dmax, then usually bid price >= MCP.")
    print("  - If d = 0, then usually bid price <= MCP.")

    print("\n--- Generator-side KKT interpretation ---")
    tol = 1e-6
    for g in thermal_units:
        q = variables[g]
        c = gen_cost[g]
        if (Pmin[g] + tol) < q < (Pmax[g] - tol):
            status = "Marginal/Internal"
            relation = f"MCP ≈ cost -> {mcp:.4f} vs {c:.4f}"
        elif abs(q - Pmax[g]) < tol:
            status = "At upper bound"
            relation = f"MCP >= cost expected -> {mcp:.4f} vs {c:.4f}"
        elif abs(q - Pmin[g]) < tol:
            status = "At lower bound"
            relation = f"Lower bound binding -> MCP may be <=/>=/= cost depending on duals"
        else:
            status = "Other"
            relation = "Check numerically"
        print(f"{g}: {status} | dispatch = {q:.4f} MW | {relation}")

    print("\n--- Demand-side KKT interpretation ---")
    for d in load_units:
        q = variables[d]
        bid = load_bid_price[d]
        if tol < q < (Dmax[d] - tol):
            status = "Marginal/Internal"
            relation = f"Bid ≈ MCP -> {bid:.4f} vs {mcp:.4f}"
        elif abs(q - Dmax[d]) < tol:
            status = "At upper bound"
            relation = f"Bid >= MCP expected -> {bid:.4f} vs {mcp:.4f}"
        elif abs(q) < tol:
            status = "Rejected"
            relation = f"Bid <= MCP expected -> {bid:.4f} vs {mcp:.4f}"
        else:
            status = "Other"
            relation = "Check numerically"
        print(f"{d}: {status} | consumption = {q:.4f} MW | {relation}")


if __name__ == "__main__":
    hour = 18

    input_data, data_pack = build_input_data_for_hour(hour)

    model = LP_OptimizationProblem(input_data)
    model.run()
    model.display_results()

    if hasattr(model.results, "objective_value"):
        print_market_results(model, data_pack)