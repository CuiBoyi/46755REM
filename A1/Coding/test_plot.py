
# Test the plot function
import numpy as np
# from plot import plot_supply_demand
import matplotlib.pyplot as plt
from utili import load_wind_scen, build_load_bids
from pathlib import Path

#-----load wind -----
BASE_DIR = Path(__file__).resolve().parent
candidates = [
    BASE_DIR / "dataset" / "WindScen.mat",
    BASE_DIR.parent / "dataset" / "WindScen.mat",
]

path_wind = next((p for p in candidates if p.exists()), None)
print(f"Looking for WindScen.mat in: {[str(p) for p in candidates]}")
if path_wind is None:
    raise FileNotFoundError(
        f"Could not find WindScen.mat. Checked: {[str(p) for p in candidates]}"
    )

wind = load_wind_scen(path_wind)  # pass explicit file path
print(f"check the index for zone=1, time =18, scenario = 0: {wind[0][18, 0]}")  # check the value at time=18, scenario=0
# print(p)  # removed: p is undefined
# ...existing code...


def plot_supply_demand(demand, supply):
    """
    Plot stepwise supply and demand curves.

    Parameters
    ----------
    demand : np.ndarray
        Column 0 = quantity, Column 5 = price
    supply : np.ndarray
        Column 0 = quantity, Column 5 = price
    """

    # sort demand by descending price
    demand_sorted = demand[np.argsort(-demand[:, 4])]
    # sort supply by ascending price
    supply_sorted = supply[np.argsort(supply[:, 4])]

    # cumulative quantities
    demand_cum = np.concatenate((np.array([0]), np.cumsum(demand_sorted[:, 0])[:-1]))
    supply_cum = np.concatenate((np.array([0]), np.cumsum(supply_sorted[:, 0])[:-1]))

    plt.figure(figsize=(10, 6))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"

    # demand curve with final point at price = 0
    demand_x = np.concatenate((demand_cum, [np.sum(demand_sorted[:, 0])]))
    demand_y = np.concatenate((demand_sorted[:, 4], [0]))

    plt.step(demand_x, demand_y, where="post", label="Demand")

    # supply curve
    plt.step(supply_cum, supply_sorted[:, 4], where="post", label="Supply")
    demand_total = np.sum(demand_sorted[:, 0])

    # plot the intersection point (MPC)
    for i in range(len(supply_cum)):
        if supply_cum[i] >= demand_cum[-1]:  # if supply exceeds total demand
            plt.plot(demand_total, supply_sorted[i, 4], "ro", label="MPC")
            break
    mcp_quantity = demand_total
    mcp_price = supply_sorted[i, 4]
    plt.annotate(
        f"MCP\n({mcp_quantity:.0f}, {mcp_price:.2f})",
        (mcp_quantity, mcp_price),
        xytext=(15, -30),          # 向右上偏移
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->")
    )
#     plt.fill_between(
#     demand_x,
#     demand_y,
#     step="post",
#     alpha=0.2,
#     color="blue",
#     label="Demand Area"
# )
#     mask = supply_cum < demand_total
#     plt.fill_between(
#     supply_cum[mask],
#     supply_sorted[:,4][mask],
#     step="post",
#     alpha=0.2,
#     color="orange",
#     label="Supply Area"
# )
    plt.xlabel("Cumulative Quantity (MW)")
    plt.ylabel("Price (€/MWh)")
    plt.title("Supply and Demand Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


supply = np.zeros([12,5])  # default, factor, pmax, pmin, cost
supply[:,2] = [152, 152, 350, 591, 60, 155, 155, 400, 400, 300, 310, 350] # Pmax
supply[:,3] = [30.4, 30.4, 75, 206.85, 12, 54.25, 54.25, 100, 100, 300, 108.5, 140] # Pmin
supply[:,4] = [13.32, 13.32, 20.70, 20.93, 26.11, 10.52, 10.52, 6.02, 5.47, 0.00, 10.52, 10.89] # gen_cost
supply[:,0] = supply[:,2]  # quantity = Pmax at first, will adjust for wind later



wind_units = np.zeros((6, 5))  # default cap, factor, max, pmin, price
wind_units[:,1] = [wind[i][17,30] for i in range(6)] # capacity factor at time=18, scenario=30 for zones 1-6
wind_units[:,2] = np.ones(6) * 200.0  # max capacity
wind_units[:,3] = np.ones(6) * 0.0  # min capacity
wind_units[:,0] = wind_units[:,1] * wind_units[:,2]  # capacity factor * max capacity = expected generation
wind_units[:,4] = np.zeros(6)  # price, set to 0 for now
print(wind_units)

supply_total = np.vstack((supply, wind_units))

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


# initialize demand and supply arrays
# demand: [quantity, price]
demand_total_values = system_demand[18]  # get load bid prices for time=18
demand = np.zeros((17, 5)) # default, factor, pmax, pmin, cost

demand[:, 1] = [3.8, 3.4, 6.3, 2.6, 2.5, 4.8, 4.4, 6.0, 6.1, 6.8, 9.3, 6.8, 11.1, 3.5, 11.7, 6.4, 4.5] # load share percent
demand[:, 0] = demand[:, 1] * demand_total_values / 100  # quantity
demand[:, 4] = build_load_bids(demand_total_values, demand[:, 1])  # load bid price

print("Demand array:")
print(demand)
print("Supply array:")
print(supply_total)

plot_supply_demand(demand, supply_total)