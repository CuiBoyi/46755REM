# %%
import matplotlib.pyplot as plt
import pandas as pd
import pypsa

country = 'PRT'
# %%


year = 2030
url = f"https://raw.githubusercontent.com/PyPSA/technology-data/v0.11.0/outputs/costs_{year}.csv"
# Read the cost data into a pandas DataFrame, using the first two columns as the index
costs = pd.read_csv(url, index_col=[0, 1])

# Convert costs from per kW to per MW for consistency
# 将成本从每千瓦转换为每兆瓦以保持一致性
costs.loc[costs.unit.str.contains("/kW"), "value"] *= 1e3
# Update the unit description to reflect the conversion
# 更新单位描述以反映转换
costs.unit = costs.unit.str.replace("/kW", "/MW")

# Define default values for missing cost parameters
# 定义缺失成本参数的默认值
defaults = {
    "FOM": 0,  # Fixed Operation and Maintenance costs
    "VOM": 0,  # Variable Operation and Maintenance costs
    "efficiency": 1,  # Default efficiency (100%)
    "fuel": 0,  # Default fuel cost
    "investment": 0,  # Default investment cost
    "lifetime": 25,  # Default lifetime in years
    "discount rate": 0.07,  # Default discount rate
}
# Fill missing values in the cost data with the defaults
costs = costs.value.unstack().fillna(defaults)

# Assign fuel costs for OCGT (Open Cycle Gas Turbine) and CCGT (Combined Cycle Gas Turbine)
costs.at["OCGT", "fuel"] = costs.at["gas", "fuel"]
costs.at["CCGT", "fuel"] = costs.at["gas", "fuel"]

def annuity(r, n):
    return r / (1.0 - 1.0 / (1.0 + r) ** n)

annuity(0.07, 20)

# Based on this, we can calculate the marginal generation costs (€/MWh):
# 基于此，我们可以计算边际发电成本（€/MWh）：
costs["marginal_cost"] = costs["VOM"] + costs["fuel"] / costs["efficiency"]

# 可变运维成本 (Variable Operation and Maintenance ~ VOM)


# Annualised investment costs (capital_cost in PyPSA terms, €/MW/a):
# 以及年化投资成本（PyPSA术语中的capital_cost，€/MW/a）：
annuity = costs.apply(lambda x: annuity(x["discount rate"], x["lifetime"]), axis=1)
costs["capital_cost"] = (annuity + costs["FOM"] / 100) * costs["investment"]

# Now we can for example read the capital and marginal cost of onshore wind and solar, or the emissions factors of the carrier gas used in and OCGT
# 现在我们可以例如读取陆上风电和太阳能的资本和边际成本，或者OCGT中使用的载体气体的排放因子

costs.at["onwind", "capital_cost"] #EUR/MW/a
print("Onshore Wind Capital Cost:", costs.at["onwind", "capital_cost"])

costs.at["solar", "capital_cost"] #EUR/MW/a
print("Solar Capital Cost:", costs.at["solar", "capital_cost"])






# %%

# Retrieving time series data

data_solar = pd.read_csv('../../Problem_data/pv_optimal.csv',sep=';')
data_solar.index = pd.DatetimeIndex(data_solar['utc_time'])

data_wind = pd.read_csv('../../Problem_data/onshore_wind_1979-2017.csv',sep=';')
data_wind.index = pd.DatetimeIndex(data_wind['utc_time'])

data_el = pd.read_csv('../../Problem_data/electricity_demand.csv',sep=';')
data_el.index = pd.DatetimeIndex(data_el['utc_time'])

data_solar.head()

country = 'PRT'


# Joint capacity and dispatch optimization
# For building the model, we start again by initialising an empty network, adding the snapshots, and the electricity bus.
# 初始化网络、设定时间维度（8760小时）并建立核心能量节点。
n = pypsa.Network() #初始化网络容器
# 构建时间序列 (Time Series)
hours_in_2015 = pd.date_range('2015-01-01 00:00Z',
                              '2015-12-31 23:00Z',
                              freq='h')

# 设定模型的“时间切片” (Snapshots)
n.set_snapshots(hours_in_2015.values)

# 添加核心节点 (Bus)
n.add("Bus",
      "electricity")
# 查看时间切片 (Snapshots)
n.snapshots

# We add all the technologies we are going to include as carriers.
# 我们将要包含的所有技术作为载体添加。
carriers = [
    "onwind",
    "solar",
    "OCGT",
    "battery storage",
]

n.add(
    "Carrier",
    carriers,
    color=["dodgerblue", "gold", "indianred", "brown"],
)


# Next, we add the demand time series to the model.
# 接下来，我们将需求时间序列添加到模型中。
# add load to the bus
n.add("Load",
      "demand",
      bus="electricity",
      p_set=data_el[country].values)

# Let’s have a check whether the data was read-in correctly.
# 检查数据是否正确读取。
n.loads_t.p_set.plot(figsize=(6, 2), ylabel="MW")

# %%

# We add now the generators and set up their capacities to be extendable so that they can be optimized together with the dispatch time series. For the wind and solar generator, we need to indicate the capacity factor or maximum power per unit ‘p_max_pu’
# 现在我们添加发电机，并将其容量设置为可扩展，以便它们可以与调度时间序列一起优化。
# 对于风能和太阳能发电机，我们需要指示每单位的容量因子或最大功率“p_max_pu”


# 
n.add(
    "Generator",
    "OCGT",
    bus="electricity",
    carrier="OCGT",
    capital_cost=costs.at["OCGT", "capital_cost"],
    marginal_cost=costs.at["OCGT", "marginal_cost"],
    efficiency=costs.at["OCGT", "efficiency"],
    p_nom_extendable=True,
)

CF_wind = data_wind[country][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in n.snapshots]]
n.add(
        "Generator",
        "onwind",
        bus="electricity",
        carrier="onwind",
        p_max_pu=CF_wind.values,
        capital_cost=costs.at["onwind", "capital_cost"],
        marginal_cost=costs.at["onwind", "marginal_cost"],
        efficiency=costs.at["onwind", "efficiency"],
        p_nom_extendable=True,
    )

CF_solar = data_solar[country][[hour.strftime("%Y-%m-%dT%H:%M:%SZ") for hour in n.snapshots]]
n.add(
        "Generator",
        "solar",
        bus="electricity",
        carrier="solar",
        p_max_pu= CF_solar.values,
        capital_cost=costs.at["solar", "capital_cost"],
        marginal_cost=costs.at["solar", "marginal_cost"],
        efficiency=costs.at["solar", "efficiency"],
        p_nom_extendable=True,
    )

n.generators_t.p_max_pu.loc["2015-01"].plot(figsize=(6, 2), ylabel="CF")





# %%
# We add the battery storage, assuming a fixed energy-to-power ratio of 2 hours, i.e. if fully charged, the battery can discharge at full capacity for 2 hours.
# For the capital cost, we have to factor in both the capacity and energy cost of the storage.
# We include the charging and discharging efficiencies we enforce a cyclic state-of-charge condition, i.e. the state of charge at the beginning of the optimisation period must equal the final state of charge.
# 我们添加电池储能，假设固定的能量与功率比为2小时，即如果完全充电，电池可以以满容量放电2小时。
# 对于资本成本，我们必须考虑储能的容量和能量成本。
# 我们包括充电和放电效率，并强制执行循环状态的条件，即优化期开始时的状态必须等于最终状态。


n.add(
    "StorageUnit",
    "battery storage",
    bus="electricity",
    carrier="battery storage",
    max_hours=2,
    capital_cost=costs.at["battery inverter", "capital_cost"]
    + 2 * costs.at["battery storage", "capital_cost"],
    efficiency_store=costs.at["battery inverter", "efficiency"],
    efficiency_dispatch=costs.at["battery inverter", "efficiency"],
    p_nom_extendable=True,
    cyclic_state_of_charge=True,
)



# %% Model run

n.optimize(solver_name="highs")




# %%
# Now, we can look at the results and evaluate the total system cost (in billion Euros per year)
# 现在，我们可以查看结果并评估总系统成本（以每年十亿欧元计）
n.objective / 1e9

# %%

# Calculate the revenues collected by the OCGT plant throughout the year and show that their sum is equal to its costs.
# 计算OCGT工厂全年收取的收入，并显示它们的总和等于其成本。

# To calculate the revenues collected by every technology, we multiply the energy generated in every hour by the electricity price in that hour and sum for the entire year.
# 要计算每项技术收取的收入，我们将每小时产生的能量乘以该小时的电价，并对全年进行求和。
n.generators_t.p.multiply(n.buses_t.marginal_price.to_numpy()).sum().div(1e6) # EUR -> MEUR

# The market revenues correspond to the total cost for OCGT, which we can also read using the statistics module:
# 市场收入对应于OCGT的总成本，我们也可以使用统计模块读取：
(n.statistics.capex() + n.statistics.opex()).div(1e6) # EUR -> MEUR

