
# Define the plot function

from turtle import width

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# supply curve and demand curve
def plot_supply_demand(demand, supply):
    """
    Plots the supply and demand curves to find MPC

    Parameters:
    demand np.array: An array of 2demand values, 0:  quantity, 1: price
    supply np.array: An array of 2supply values, 0:  quantity, 1: price
    """
    plt.plot(demand[:, 0], demand[:, 1], label='Demand')
    plt.plot(supply[:, 0], supply[:, 1], label='Supply')
    plt.xlabel('Quantity')
    plt.ylabel('Price')
    plt.title('Supply and Demand Curves') 
    plt.legend()
    plt.show()

def plot_storage_operation_compact(result_pack: dict, title: str = "Storage Operation Over 24 Hours"):
    pch = np.asarray(result_pack["pch"], dtype=float)
    pdis = np.asarray(result_pack["pdis"], dtype=float)
    e = np.asarray(result_pack["e"], dtype=float)

    hours = np.arange(1, len(pch) + 1)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot charge as negative, discharge as positive
    ax1.bar(hours, -pch, width=0.6, label="Charging power [MW]")
    ax1.bar(hours, pdis, width=0.6, label="Discharging power [MW]")
    ax1.axhline(0, linewidth=1)
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Power [MW]")
    ax1.set_xticks(hours)
    ax1.grid(True, axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(hours, e, marker="o", linewidth=2, label="Battery energy [MWh]")
    ax2.set_ylabel("Stored energy [MWh]")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_storage_price_and_profit(result_pack: dict, title: str = "Hourly MCP and Storage Profit"):
    import matplotlib.pyplot as plt
    import numpy as np

    mcp = np.asarray(result_pack["mcp"], dtype=float)
    storage_profit = np.asarray(result_pack["storage_profit_hourly"], dtype=float)

    hours = np.arange(1, len(mcp) + 1)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # MCP on left axis
    ax1.plot(hours, mcp, marker="o", linewidth=2, label="MCP [$/MWh]")
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("MCP [$/MWh]")
    ax1.set_xticks(hours)
    ax1.grid(True, axis="both", alpha=0.3)

    # Storage profit on right axis
    ax2 = ax1.twinx()
    ax2.bar(hours, storage_profit, alpha=0.5, label="Storage profit [$]")
    ax2.set_ylabel("Storage profit [$]")

    # Combined legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_cost_profit_contributions(result_pack: dict, data_pack: dict):
    import numpy as np
    import matplotlib.pyplot as plt

    g = result_pack["g"]
    w = result_pack["w"]
    mcp = result_pack["mcp"]

    gen_cost = data_pack["gen_cost"]
    thermal_names = data_pack["thermal_names"]
    wind_names = data_pack["wind_names"]

    T, ng = g.shape
    nw = w.shape[1]

    hours = np.arange(1, T + 1)

    # Operating cost per generator
    cost = g * gen_cost[None, :]

    # Profit per generator
    thermal_profit = g * (mcp[:, None] - gen_cost[None, :])
    wind_profit = w * mcp[:, None]

    fig, ax = plt.subplots(figsize=(14, 7))

    # ---------- COST (above axis) ----------
    bottom_cost = np.zeros(T)

    for i in range(ng):
        ax.bar(
            hours,
            cost[:, i],
            bottom=bottom_cost,
            label=f"{thermal_names[i]} cost",
        )
        bottom_cost += cost[:, i]

    # ---------- PROFIT (below axis) ----------
    bottom_profit = np.zeros(T)

    for i in range(ng):
        ax.bar(
            hours,
            -thermal_profit[:, i],
            bottom=-bottom_profit,
            label=f"{thermal_names[i]} profit",
            alpha=0.6,
        )
        bottom_profit += thermal_profit[:, i]

    for i in range(nw):
        ax.bar(
            hours,
            -wind_profit[:, i],
            bottom=-bottom_profit,
            label=f"{wind_names[i]} profit",
            alpha=0.6,
        )
        bottom_profit += wind_profit[:, i]

    ax.axhline(0, color="black", linewidth=1)

    ax.set_xlabel("Hour")
    ax.set_ylabel("$")
    ax.set_title("Operating Cost (above) and Profit (below) by Unit")
    ax.set_xticks(hours)

    ax.grid(True, axis="y", alpha=0.3)

    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()

def plot_mcp_comparison(no_storage: dict, with_storage: dict):
    import matplotlib.pyplot as plt
    import numpy as np

    hours = np.arange(1, len(no_storage["mcp"]) + 1)

    plt.figure(figsize=(12, 5))
    plt.plot(hours, no_storage["mcp"], marker="o", linewidth=2, label="Without storage")
    plt.plot(hours, with_storage["mcp"], marker="s", linewidth=2, label="With storage")
    plt.xlabel("Hour")
    plt.ylabel("MCP [$/MWh]")
    plt.title("Hourly Market-Clearing Price Comparison")
    plt.xticks(hours)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_system_comparison(no_storage: dict, with_storage: dict):
    import numpy as np
    import matplotlib.pyplot as plt

    hours = np.arange(1, len(no_storage["mcp"]) + 1)

    # Hourly total profit of all producers
    no_profit_hourly = (
        np.sum(no_storage["thermal_profit_hourly"], axis=1)
        + np.sum(no_storage["wind_profit_hourly"], axis=1)
    )

    with_profit_hourly = (
        np.sum(with_storage["thermal_profit_hourly"], axis=1)
        + np.sum(with_storage["wind_profit_hourly"], axis=1)
        + with_storage["storage_profit_hourly"]
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax1, ax2, ax3, ax4 = axes.flatten()

    # --------------------------------------------------
    # 1. MCP
    # --------------------------------------------------
    ax1.plot(hours, no_storage["mcp"], marker="o", linewidth=2, label="Without storage")
    ax1.plot(hours, with_storage["mcp"], marker="s", linewidth=2, label="With storage")
    ax1.set_title("Hourly MCP")
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("$/MWh")
    ax1.set_xticks(hours)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # --------------------------------------------------
    # 2. Operating cost
    # --------------------------------------------------
    ax2.plot(hours, no_storage["hourly_operating_cost"]/1000, marker="o", linewidth=2, label="Without storage")
    ax2.plot(hours, with_storage["hourly_operating_cost"]/1000, marker="s", linewidth=2, label="With storage")
    ax2.set_title("Hourly Operating Cost")
    ax2.set_xlabel("Hour")
    ax2.set_ylabel("Cost [k$]")
    ax2.set_xticks(hours)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # --------------------------------------------------
    # 3. Total profit
    # --------------------------------------------------
    ax3.plot(hours, no_profit_hourly/1000, marker="o", linewidth=2, label="Without storage")
    ax3.plot(hours, with_profit_hourly/1000, marker="s", linewidth=2, label="With storage")
    ax3.set_title("Hourly Total Producer Profit")
    ax3.set_xlabel("Hour")
    ax3.set_ylabel("Profit [k$]")
    ax3.set_xticks(hours)
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # --------------------------------------------------
    # 4. Social welfare
    # --------------------------------------------------
    ax4.plot(hours, no_storage["hourly_social_welfare"]/1000, marker="o", linewidth=2, label="Without storage")
    ax4.plot(hours, with_storage["hourly_social_welfare"]/1000, marker="s", linewidth=2, label="With storage")
    ax4.set_title("Hourly Social Welfare")
    ax4.set_xlabel("Hour")
    ax4.set_ylabel("Welfare [k$]")
    ax4.set_xticks(hours)
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # 24h totals
    total_text = (
        "24h totals\n"
        f"No storage:  cost={no_storage['total_operating_cost']/1000:.1f} k$, "
        f"profit={np.sum(no_profit_hourly)/1000:.1f} k$, "
        f"welfare={no_storage['total_social_welfare']/1000:.1f} k$\n"
        f"With storage: cost={with_storage['total_operating_cost']/1000:.1f} k$, "
        f"profit={np.sum(with_profit_hourly)/1000:.1f} k$, "
        f"welfare={with_storage['total_social_welfare']/1000:.1f} k$"
    )

    fig.text(0.5, 0.01, total_text, ha="center", va="bottom", fontsize=15)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()

def print_assignment_results(result_pack: dict, data_pack: dict, case_name: str):
    print("\n" + "=" * 70)
    print(case_name)
    print("=" * 70)

    print("\nHourly market-clearing prices (€/MWh):")
    for t, p in enumerate(result_pack["mcp"], start=1):
        print(f"Hour {t:02d}: {p:8.4f}")

    print("\nSystem totals over 24h:")
    print(f"Total operating cost   = {result_pack['total_operating_cost']:.4f} €")
    print(f"Total load value       = {result_pack['total_load_value']:.4f} €")
    print(f"Total social welfare   = {result_pack['total_social_welfare']:.4f} €")

    print("\nTotal profit of each conventional unit:")
    for name, val in zip(data_pack["thermal_names"], result_pack["total_thermal_profit"]):
        print(f"{name}: {val:10.4f} €")

    print("\nTotal profit of each wind farm:")
    for name, val in zip(data_pack["wind_names"], result_pack["total_wind_profit"]):
        print(f"{name}: {val:10.4f} €")

    if data_pack["include_storage"]:
        print(f"\nTotal storage profit   = {result_pack['total_storage_profit']:.4f} €")

def print_sensitivity_results(outputs):
    print("\n" + "=" * 90)
    print("SENSITIVITY ANALYSIS")
    print("=" * 90)
    print(f"{'Case':20s} {'OpCost (€)':>14s} {'Welfare (€)':>14s} {'StorageProfit (€)':>18s} {'Avg MCP':>10s} {'Min MCP':>10s} {'Max MCP':>10s}")
    for row in outputs:
        print(
            f"{row['name']:20s} "
            f"{row['total_operating_cost']:14.2f} "
            f"{row['total_social_welfare']:14.2f} "
            f"{row['total_storage_profit']:18.2f} "
            f"{row['avg_mcp']:10.2f} "
            f"{row['min_mcp']:10.2f} "
            f"{row['max_mcp']:10.2f}"
        )

def plot_storage_hourly(outputs):
    hours = np.arange(1, len(outputs[0]["hourly_results"]["storage_profit_hourly"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()

    for case in outputs:
        r = case["hourly_results"]
        label = case["name"]

        ax1.plot(hours, r["mcp"], marker="o", linewidth=1.8, label=label)
        ax2.plot(hours, r["hourly_operating_cost"], marker="o", linewidth=1.8, label=label)
        ax3.plot(hours, r["e"], marker="o", linewidth=1.8, label=label)
        ax4.plot(hours, r["hourly_social_welfare"], marker="o", linewidth=1.8, label=label)

    ax1.set_title("Hourly Market-Clearing Price")
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("€/MWh")
    ax1.set_xticks(hours)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.set_title("Hourly Operating Cost")
    ax2.set_xlabel("Hour")
    ax2.set_ylabel("€")
    ax2.set_xticks(hours)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    ax3.set_title("Hourly State of Charge")
    ax3.set_xlabel("Hour")
    ax3.set_ylabel("MWh")
    ax3.set_xticks(hours)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)

    ax4.set_title("Hourly Social Welfare")
    ax4.set_xlabel("Hour")
    ax4.set_ylabel("€")
    ax4.set_xticks(hours)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=8)

    # 24h totals
    total_text = "24h totals\n"

    for case in outputs:
        r = case["hourly_results"]

        total_profit = (
            np.sum(r["thermal_profit_hourly"])
            + np.sum(r["wind_profit_hourly"])
            + np.sum(r["storage_profit_hourly"])
        )

        total_text += (
            f"{case['name']}: "
            f"cost={r['total_operating_cost']/1000:.1f} k€, "
            f"profit={total_profit/1000:.1f} k€, "
            f"welfare={r['total_social_welfare']/1000:.1f} k€\n"
        )

    fig.text(0.5, 0.01, total_text, ha="center", va="bottom", fontsize=15)

    plt.tight_layout(rect=[0, 0.15, 1, 1])
    plt.show()

import matplotlib.pyplot as plt


def plot_dispatch_and_profit_step5(pack_da: dict, result_da: dict, profit_pack: dict):
    """
    Plot:
    - x-axis: unit names (conventional + wind)
    - left y-axis: generation output (bar)
        * day-ahead dispatch
        * actual real-time output
    - right y-axis: profit (line)
        * day-ahead profit
        * one-price total profit
        * two-price total profit
    """

    thermal_names = pack_da["thermal_names"]
    wind_names = pack_da["wind_names"]
    unit_names = thermal_names + wind_names

    # ----------------------------
    # Generation data
    # ----------------------------
    da_output = np.concatenate([result_da["g_da"], result_da["w_da"]])
    actual_output = np.concatenate([profit_pack["g_actual"], profit_pack["w_actual"]])

    # ----------------------------
    # Profit data
    # DA-only profit from day-ahead market
    # ----------------------------
    da_profit = np.concatenate([result_da["thermal_profit_da"], result_da["wind_profit_da"]])

    # Total profit under one-price and two-price settlement
    one_price_profit = np.concatenate([profit_pack["conv_profit_one"], profit_pack["wind_profit_one"]])
    two_price_profit = np.concatenate([profit_pack["conv_profit_two"], profit_pack["wind_profit_two"]])

    # ----------------------------
    # Plot settings
    # ----------------------------
    x = np.arange(len(unit_names))
    width = 0.36

    fig, ax1 = plt.subplots(figsize=(16, 7))

    # Left axis: output bars
    #bars1 = ax1.bar(x - width/2, da_output, width, label="Day-ahead output")
    #bars2 = ax1.bar(x + width/2, actual_output, width, label="Actual real-time output")
    bars1 = ax1.bar(x - width/2, da_output, width,
                color="#005FF8",
                alpha=0.75,
                edgecolor="black",
                linewidth=0.5,
                label="Day-ahead")

    bars2 = ax1.bar(x + width/2, actual_output, width,
                color="#FF5E00",
                alpha=0.75,
                edgecolor="black",
                linewidth=0.5,
                label="Real-time")

    ax1.set_xlabel("Generating units")
    ax1.set_ylabel("Power output [MW]")
    ax1.set_xticks(x)
    ax1.set_xticklabels(unit_names, rotation=45)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.set_ylim(0, 1200)
    ax1.set_yticks(np.arange(0, 1200, 200))

    # Right axis: profit lines
    ax2 = ax1.twinx()
    line1, = ax2.plot(x, da_profit,color="#005FF8", marker="s",
         markersize=8,
         markerfacecolor="white",
         markeredgecolor="black", linewidth=1, label="Day-ahead profit")
    line2, = ax2.plot(x, one_price_profit, color="#FF5E00", marker="x", markersize=8,
         markerfacecolor="white",
         markeredgecolor="black",linewidth=1, label="One-price profit")
    line3, = ax2.plot(x, two_price_profit, color="#FF5E00", marker="+", markersize=8,
         markerfacecolor="white",
         markeredgecolor="black",linewidth=1, label="Two-price profit")

    ax2.set_ylabel("Profit [€]")
    ax2.set_ylim(-3500, 3500)  # Adjust as needed
    ax2.set_yticks(np.arange(-3500, 3500, 1000))


    # Vertical separator between conventional and wind
    split_pos = len(thermal_names) - 0.5
    #ax1.axvline(split_pos, linestyle="--", alpha=0.7)

    # Optional text labels
    ymax_left = max(np.max(da_output), np.max(actual_output))
    #ax1.text(len(thermal_names)/2 - 0.5, ymax_left * 1.03, "Conventional", ha="center", va="bottom")
    #ax1.text(len(thermal_names) + len(wind_names)/2 - 0.5, ymax_left * 1.03, "Wind", ha="center", va="bottom")

    # Combined legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", ncol=2)
    ax2.axhline(y=0, linestyle="--", linewidth=1)

    # =========================
    # inset for wind profits
    # =========================
    axins = inset_axes(ax2, width="32%", height="32%", bbox_to_anchor=(0.075, -0.7, 0.9, 1.3),
    bbox_transform=ax2.transAxes,)

    # wind index range
    n_th = len(thermal_names)
    wind_x = x[n_th:]

    # only plot wind profits in inset
    axins.plot(wind_x, da_profit[n_th:], color="#005FF8", marker="s",markersize=8,
         markerfacecolor="white",
         markeredgecolor="black", linewidth=1.5, label="DA profit")
    axins.plot(wind_x, one_price_profit[n_th:], color="#FF5E00", marker="x", markersize=8,
         markerfacecolor="white",
         markeredgecolor="black",linewidth=1.5, label="One-price")
    axins.plot(wind_x, two_price_profit[n_th:], color="#FF5E00", marker="+", markersize=8,
         markerfacecolor="white",
         markeredgecolor="black",linewidth=1.5, label="Two-price")

    # zero line
    axins.axhline(0, linestyle="--", linewidth=0.8)

    # x ticks only for wind units
    axins.set_xticks(wind_x)
    axins.set_xticklabels(unit_names[n_th:], rotation=45, fontsize=8)

    # zoom y-range around wind profits
    wind_profit_min = min(
        np.min(da_profit[n_th:]),
        np.min(one_price_profit[n_th:]),
        np.min(two_price_profit[n_th:])
    )
    wind_profit_max = max(
        np.max(da_profit[n_th:]),
        np.max(one_price_profit[n_th:]),
        np.max(two_price_profit[n_th:])
    )

    margin = 0.08 * (wind_profit_max - wind_profit_min)
    axins.set_ylim(wind_profit_min - margin, wind_profit_max + margin)

    axins.set_title("Zoom: wind profits", fontsize=9)
    axins.grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle("Day-ahead vs Real-time Output and Profit")
    #plt.title("Day-ahead vs Real-time Output and Profit by Unit")
    plt.tight_layout()
    plt.show()