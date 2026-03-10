# %%

import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB


# %%
# ==========================================
# 1. 获取数据与参数设定
# ==========================================
# 获取 PyPSA 成本数据
year = 2030
url = f"https://raw.githubusercontent.com/PyPSA/technology-data/v0.11.0/outputs/costs_{year}.csv"
costs = pd.read_csv(url, index_col=[0, 1])
costs.loc[costs.unit.str.contains("/kW"), "value"] *= 1e3
costs.unit = costs.unit.str.replace("/kW", "/MW")
defaults = {"FOM": 0, "VOM": 0, "efficiency": 1, "fuel": 0, "investment": 0, "lifetime": 25, "discount rate": 0.07}
costs = costs.value.unstack().fillna(defaults)
costs.at["OCGT", "fuel"] = costs.at["gas", "fuel"]
costs["marginal_cost"] = costs["VOM"] + costs["fuel"] / costs["efficiency"]

# 读取负荷切片 (中午 12:00)
data_day1 = pd.read_csv('data_day1.csv')
demand_factor = data_day1.loc[12, 'electricity'] # 0.910

# 发电机参数 [边际成本, 容量]
mc_gen = {'Wind': 0.0, 'Solar': 0.0, 'OCGT': costs.at["OCGT", "marginal_cost"]}
cap_gen = {'Wind': 300.0, 'Solar': 200.0, 'OCGT': 600.0}

# 需求侧参数 [报价, 最大需求量]
bid_dem = {'D1': 150.0, 'D2': 120.0}
max_dem = {'D1': 400.0 * demand_factor, 'D2': 500.0 * demand_factor}

# ==========================================
# 2. 构建 Gurobi 模型
# ==========================================
m = gp.Model("Market_Clearing_Step1_KKT")
m.setParam('OutputFlag', 0) # 关闭 Gurobi 默认求解日志以便让输出更整洁

# 定义变量 (注意：这里只限制 lb=0，上限留给显式约束以便提取对偶变量)
g = {i: m.addVar(lb=0, name=f"g_{i}") for i in mc_gen}
d = {j: m.addVar(lb=0, name=f"d_{j}") for j in bid_dem}

# 目标函数：最大化社会福利
welfare = gp.quicksum(bid_dem[j] * d[j] for j in bid_dem) - gp.quicksum(mc_gen[i] * g[i] for i in mc_gen)
m.setObjective(welfare, GRB.MAXIMIZE)

# 核心约束 1：功率平衡 (Demand - Generation == 0)
# 这样设定的对偶变量直接就是正值的市场出清价格 (MCP)
power_balance = m.addConstr(gp.quicksum(d[j] for j in bid_dem) - gp.quicksum(g[i] for i in mc_gen) == 0, name="Power_Balance")

# 核心约束 2：容量上限 (为了提取 mu_max)
cap_g_constr = {i: m.addConstr(g[i] <= cap_gen[i], name=f"Cap_g_{i}") for i in mc_gen}
max_d_constr = {j: m.addConstr(d[j] <= max_dem[j], name=f"Max_d_{j}") for j in bid_dem}

# 求解
m.optimize()

# ==========================================
# 3. 提取结果与 KKT 验证打印
# ==========================================
if m.status == GRB.OPTIMAL:
    MCP = power_balance.Pi # 提取功率平衡约束的影子价格 (统一出清价格)
    
    print("="*60)
    print(f"MARKET CLEARING OUTCOMES")
    print("="*60)
    print(f"Market-Clearing Price (MCP): {MCP:.2f} €/MWh")
    print(f"Total Social Welfare:        {m.ObjVal:.2f} €\n")
    
    print("="*60)
    print("KKT CONDITIONS VERIFICATION (Stationarity & Complementary Slackness)")
    print("="*60)
    
    # 发电侧 KKT 验证
    print("--- PRODUCERS (Generators) ---")
    for i in mc_gen:
        mc = mc_gen[i]
        disp = g[i].X
        mu_max = cap_g_constr[i].Pi
        # 根据平稳性条件计算下限对偶变量 mu_min: MCP - MC = mu_max - mu_min
        mu_min = max(0, mu_max + mc - MCP) 
        
        status = "Marginal" if 0 < disp < cap_gen[i] else ("Accepted (Full)" if disp == cap_gen[i] else "Rejected")
        
        print(f"\nGenerator: {i} [{status}]")
        print(f"  Bid Price (MC): {mc:.2f} €/MWh  |  Dispatch: {disp:.2f} MW / {cap_gen[i]:.2f} MW")
        print(f"  Dual Variables: mu_max = {mu_max:.2f}, mu_min = {mu_min:.2f}")
        print(f"  Stationarity Check: MCP ({MCP:.2f}) == MC ({mc:.2f}) + mu_max ({mu_max:.2f}) - mu_min ({mu_min:.2f}) -> {abs(MCP - (mc + mu_max - mu_min)) < 1e-5}")
        print(f"  Complementary Slackness:")
        print(f"    mu_max * (P_max - P) = {mu_max:.2f} * {cap_gen[i] - disp:.2f} = {mu_max * (cap_gen[i] - disp):.2f}")
        print(f"    mu_min * P = {mu_min:.2f} * {disp:.2f} = {mu_min * disp:.2f}")

    # 需求侧 KKT 验证
    print("\n--- CONSUMERS (Demands) ---")
    for j in bid_dem:
        bid = bid_dem[j]
        cons = d[j].X
        nu_max = max_d_constr[j].Pi
        # 根据平稳性条件计算下限对偶变量 nu_min: Bid - MCP = nu_max - nu_min

        nu_min = max(0, nu_max + MCP - bid)
        
        status = "Marginal" if 0 < cons < max_dem[j] else ("Accepted (Full)" if cons == max_dem[j] else "Rejected")
        
        print(f"\nDemand: {j} [{status}]")
        print(f"  Bid Price: {bid:.2f} €/MWh  |  Consumption: {cons:.2f} MW / {max_dem[j]:.2f} MW")
        print(f"  Dual Variables: nu_max = {nu_max:.2f}, nu_min = {nu_min:.2f}")
        print(f"  nu_min * D = {nu_min:.2f} * {cons:.2f} = {nu_min * cons:.2f}")