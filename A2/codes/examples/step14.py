"""
Risk-averse offering strategy analysis — CVaR formulation (Lecture 9).

Produces five figures saved to outputs/:
  ra_fig1_frontier.png       – E[profit] vs CVaR frontier, both schemes
  ra_fig2_offers.png         – Hourly DA offers for each beta, both schemes
  ra_fig3_distributions.png  – Out-of-sample profit distributions per beta
  ra_fig4_fold_robustness.png– Sensitivity of CVaR offers to fold selection
  ra_fig5_std_and_min.png    – Profit std-dev and worst-case vs beta

Solver strategy:
  One-price CVaR  → Gurobi (small LP: 24+1+n_scen variables)
  Two-price CVaR  → scipy HiGHS (LP with over/under linearisation variables
                     exceeds Gurobi restricted-license size limit)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from scipy.optimize import linprog
from scipy.sparse import lil_matrix, csr_matrix

# ── import shared helpers from step1 ─────────────────────────────────────────
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from step1 import (
    load_ready_data,
    generate_imbalance_scenarios,
    generate_balancing_prices,
    build_joint_scenarios,
    build_equal_probabilities,
    scenario_subset,
    complement_indices,
    build_cross_validation_folds,
    scenario_arrays,
    one_price_profits_from_offer,
    two_price_profits_from_offer,
    empirical_profit_cvar,
    solve_one_price_cvar_gurobi,      # still use Gurobi for one-price (small LP)
    N_WIND, N_PRICE, N_IMB, N_HOURS,
    P_DEFICIT, RANDOM_SEED, CAPACITY_MW,
    CVAR_ALPHA,
    CV_N_FOLDS, CV_TOTAL_SCENARIOS, CV_IN_SAMPLE_SIZE,
    BASE_DIR,
    solve_one_price,
    solve_two_price,
)

# ── configuration ─────────────────────────────────────────────────────────────
BETAS   = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4, 6, 8, 10.0]
SCHEMES = ["One-price", "Two-price"]
OUT     = BASE_DIR / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

BETA_COLORS = plt.cm.plasma(np.linspace(0.1, 0.85, len(BETAS)))
EUR_K = 1e3


# ═════════════════════════════════════════════════════════════════════════════
# Two-price CVaR LP via scipy / HiGHS
# ═════════════════════════════════════════════════════════════════════════════

def _solve_two_price_cvar_scipy(joint_scenarios, beta,
                                alpha=CVAR_ALPHA, capacity=CAPACITY_MW):
    """
    LP formulation of:
        max  E[profit] + beta*(zeta - 1/(1-alpha)*E[eta])

    Variables  (column order):
        q[T]          day-ahead offers
        over[S,T]     positive deviations  (wind - q)+
        under[S,T]    negative deviations  (q - wind)+
        zeta          VaR level
        eta[S]        CVaR shortfall  (>= 0)

    Two-price settlement:
        SI=1 (deficit):  over settled at DA,    under settled at BP
        SI=0 (surplus):  over settled at BP,    under settled at DA
    """
    arrays = scenario_arrays(joint_scenarios)
    wind = arrays["wind"]          # (S, T)
    da   = arrays["da"]
    bp   = arrays["bp"]
    si   = arrays["si"]

    S, T = wind.shape
    probs = np.full(S, 1.0 / S)

    # settlement prices per deviation direction
    settle_ov  = np.where(si == 1, da,  bp)   # (S, T)  over revenue rate
    settle_un  = np.where(si == 1, bp,  da)   # (S, T)  under cost rate

    # ---- variable index helpers ----------------------------------------
    nq  = T
    nov = S * T
    nun = S * T
    nz  = 1
    ne  = S
    NV  = nq + nov + nun + nz + ne

    iq  = slice(0,   nq)
    iov = lambda s, t: nq + s*T + t
    iun = lambda s, t: nq + nov + s*T + t
    iz  = nq + nov + nun
    ie  = lambda s:    nq + nov + nun + nz + s

    # ---- objective (scipy minimises, so negate) -------------------------
    c = np.zeros(NV)
    # E[DA_t * q_t]  →  c[q_t] = -Σ_s probs[s]*DA[s,t]
    c[iq] = -probs @ da                         # shape (T,)
    # E[settle_ov * over]
    for s in range(S):
        for t in range(T):
            c[iov(s, t)] = -probs[s] * settle_ov[s, t]
    # E[settle_un * under]  (cost, so positive in minimisation)
    for s in range(S):
        for t in range(T):
            c[iun(s, t)] = +probs[s] * settle_un[s, t]
    # beta * zeta
    c[iz] = -beta
    # -beta/(1-alpha) * E[eta]
    for s in range(S):
        c[ie(s)] = beta / (1.0 - alpha) * probs[s]

    # ---- equality: wind[s,t] - q[t] = over[s,t] - under[s,t]
    #      => q[t] + over[s,t] - under[s,t] = wind[s,t]  -----------
    n_eq = S * T
    Aeq  = lil_matrix((n_eq, NV))
    beq  = np.zeros(n_eq)
    for s in range(S):
        for t in range(T):
            row = s * T + t
            Aeq[row, t]           =  1.0   # q[t]
            Aeq[row, iov(s, t)]   =  1.0   # +over[s,t]
            Aeq[row, iun(s, t)]   = -1.0   # -under[s,t]
            beq[row]              = wind[s, t]

    # ---- inequality: zeta - eta[s] - profit[s] <= 0  (for each s) ------
    # profit[s] = Σ_t [DA*q + settle_ov*over - settle_un*under]
    n_ub = S
    Aub  = lil_matrix((n_ub, NV))
    bub  = np.zeros(n_ub)
    for s in range(S):
        Aub[s, iz]    =  1.0          # zeta
        Aub[s, ie(s)] = -1.0          # -eta[s]
        for t in range(T):
            Aub[s, t]           -= da[s, t]            # -DA*q
            Aub[s, iov(s, t)]   -= settle_ov[s, t]    # -settle_ov*over
            Aub[s, iun(s, t)]   += settle_un[s, t]    # +settle_un*under

    # ---- variable bounds ------------------------------------------------
    # Relaxed bounds on over/under (equality constraints enforce the
    # physical balance; objective naturally sets one to zero at optimality).
    bounds = (
        [(0.0, float(capacity))] * T +         # q
        [(0.0, float(capacity))] * (S * T) +   # over
        [(0.0, float(capacity))] * (S * T) +   # under
        [(None, None)] +                        # zeta (unbounded)
        [(0.0, None)] * S                       # eta >= 0
    )

    res = linprog(c, A_ub=csr_matrix(Aub), b_ub=bub,
                  A_eq=csr_matrix(Aeq), b_eq=beq,
                  bounds=bounds,
                  method="highs")

    if res.status not in (0, 1):
        raise RuntimeError(f"Two-price CVaR scipy LP failed: {res.message}")

    return np.clip(res.x[iq], 0.0, capacity)


# ═════════════════════════════════════════════════════════════════════════════
# Unified solve / evaluate wrappers
# ═════════════════════════════════════════════════════════════════════════════

def _solve_risk_averse(joint_scenarios, beta, scheme):
    """Return offer_mw and in-sample profits for the given scheme and beta."""
    t0   = perf_counter()
    probs = build_equal_probabilities(len(joint_scenarios))

    if beta == 0.0:
        if scheme == "One-price":
            res = solve_one_price(joint_scenarios, probs, CAPACITY_MW)
        else:
            res = solve_two_price(joint_scenarios, probs, CAPACITY_MW)
        offer = res["offer_mw"]
    elif scheme == "One-price":
        offer, _ = solve_one_price_cvar_gurobi(joint_scenarios, beta, CVAR_ALPHA)
    else:
        offer = _solve_two_price_cvar_scipy(joint_scenarios, beta, CVAR_ALPHA)

    # compute in-sample profits
    arr     = scenario_arrays(joint_scenarios)
    if scheme == "One-price":
        profits = one_price_profits_from_offer(arr, offer)
    else:
        profits = two_price_profits_from_offer(arr, offer)

    return {
        "offer_mw":         offer,
        "scenario_profits": profits,
        "expected_profit":  float(np.mean(profits)),
        "cvar":             empirical_profit_cvar(profits, CVAR_ALPHA),
        "solve_time":       perf_counter() - t0,
    }


def _eval_out_of_sample(joint_scenarios, offer_mw, scheme):
    arr = scenario_arrays(joint_scenarios)
    if scheme == "One-price":
        profits = one_price_profits_from_offer(arr, offer_mw)
    else:
        profits = two_price_profits_from_offer(arr, offer_mw)
    return {
        "expected_profit": float(np.mean(profits)),
        "cvar":            empirical_profit_cvar(profits, CVAR_ALPHA),
        "profit_std":      float(np.std(profits, ddof=0)),
        "profit_min":      float(np.min(profits)),
        "profit_max":      float(np.max(profits)),
        "scenario_profits": profits,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Scenario construction
# ═════════════════════════════════════════════════════════════════════════════

def build_scenarios():
    wind, da = load_ready_data()
    si  = generate_imbalance_scenarios(n_imb=N_IMB, n_hours=N_HOURS,
                                       p_deficit=P_DEFICIT, seed=RANDOM_SEED)
    bp  = generate_balancing_prices(da, si)
    return build_joint_scenarios(wind, da, si, bp)


# ═════════════════════════════════════════════════════════════════════════════
# Core analysis: risk-averse frontier
# ═════════════════════════════════════════════════════════════════════════════

def run_cvar_frontier(joint):
    folds   = build_cross_validation_folds(joint)
    all_idx = np.sort(np.concatenate(folds))
    in_idx  = folds[0]
    out_idx = complement_indices(all_idx, in_idx)
    in_scen  = scenario_subset(joint, in_idx)
    out_scen = scenario_subset(joint, out_idx)

    print(f"\nFrontier: {len(in_scen)} in-sample  |  {len(out_scen)} out-of-sample")

    records, offer_records, all_profits = [], [], {}

    for scheme in SCHEMES:
        for beta in BETAS:
            print(f"  {scheme:12s}  beta={beta:>5.2f} … ", end="", flush=True)
            res      = _solve_risk_averse(in_scen, beta, scheme)
            out_eval = _eval_out_of_sample(out_scen, res["offer_mw"], scheme)
            print(f"in E[π]={res['expected_profit']/EUR_K:.1f}k  "
                  f"CVaR={res['cvar']/EUR_K:.1f}k  "
                  f"out E[π]={out_eval['expected_profit']/EUR_K:.1f}k  "
                  f"({res['solve_time']*1e3:.0f} ms)")

            records.append({
                "Scheme":            scheme,
                "Beta":              beta,
                "In_Expected_EUR":   res["expected_profit"],
                "In_CVaR_EUR":       res["cvar"],
                "In_Std_EUR":        float(np.std(res["scenario_profits"])),
                "Out_Expected_EUR":  out_eval["expected_profit"],
                "Out_CVaR_EUR":      out_eval["cvar"],
                "Out_Std_EUR":       out_eval["profit_std"],
                "Out_Min_EUR":       out_eval["profit_min"],
                "Out_Max_EUR":       out_eval["profit_max"],
                "Offer_MW":          res["offer_mw"],
            })
            all_profits[(scheme, beta)] = out_eval["scenario_profits"]

            for h, q in enumerate(res["offer_mw"], start=1):
                offer_records.append({"Scheme": scheme, "Beta": beta,
                                      "Hour": h, "DA_Offer_MW": q})

    return (pd.DataFrame(records),
            pd.DataFrame(offer_records),
            all_profits,
            in_scen, out_scen)


# ═════════════════════════════════════════════════════════════════════════════
# Fold robustness at beta = 1.0
# ═════════════════════════════════════════════════════════════════════════════

def run_fold_robustness(joint, beta=1.0):
    folds   = build_cross_validation_folds(joint)
    all_idx = np.sort(np.concatenate(folds))
    records, offer_records = [], []

    print(f"\nFold robustness at beta = {beta}")
    for fold_id, in_idx in enumerate(folds, start=1):
        out_idx  = complement_indices(all_idx, in_idx)
        in_scen  = scenario_subset(joint, in_idx)
        out_scen = scenario_subset(joint, out_idx)

        for scheme in SCHEMES:
            res      = _solve_risk_averse(in_scen, beta, scheme)
            out_eval = _eval_out_of_sample(out_scen, res["offer_mw"], scheme)
            print(f"  Fold {fold_id}  {scheme:12s}  "
                  f"E[π]={out_eval['expected_profit']/EUR_K:.1f}k  "
                  f"CVaR={out_eval['cvar']/EUR_K:.1f}k")
            records.append({
                "Fold":             fold_id,
                "Scheme":           scheme,
                "In_Expected_EUR":  res["expected_profit"],
                "In_CVaR_EUR":      res["cvar"],
                "Out_Expected_EUR": out_eval["expected_profit"],
                "Out_CVaR_EUR":     out_eval["cvar"],
                "Out_Std_EUR":      out_eval["profit_std"],
                "Mean_Offer_MW":    float(np.mean(res["offer_mw"])),
                "Offer_MW":         res["offer_mw"],
            })
            for h, q in enumerate(res["offer_mw"], start=1):
                offer_records.append({"Fold": fold_id, "Scheme": scheme,
                                      "Hour": h, "DA_Offer_MW": q})

    return pd.DataFrame(records), pd.DataFrame(offer_records)


# ═════════════════════════════════════════════════════════════════════════════
# Figure 1 – Risk-averse frontier (E[profit] vs CVaR)
# ═════════════════════════════════════════════════════════════════════════════

def fig1_frontier(frontier, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, scheme in zip(axes, SCHEMES):
        sub = frontier[frontier["Scheme"] == scheme].sort_values("Beta")

        ax.plot(sub["In_CVaR_EUR"] / EUR_K, sub["In_Expected_EUR"] / EUR_K,
                "o-", color="steelblue", lw=2.0, ms=9)

        for _, row in sub.iterrows():
            ax.annotate(
                f"β={row['Beta']:g}",
                (row["In_CVaR_EUR"] / EUR_K,
                 row["In_Expected_EUR"] / EUR_K),
                textcoords="offset points", xytext=(5, 4),
                fontsize=9, color="steelblue")

        ax.set_title(scheme, fontsize=13, fontweight="bold")
        ax.set_xlabel("CVaR of daily profit (k EUR)", fontsize=11)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Expected daily profit (k EUR)", fontsize=11)
    fig.suptitle(
        f"Risk-averse frontier: E[profit] vs CVaR  (α = {CVAR_ALPHA:.2f}, in-sample)",
        fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"  Saved {path.name}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 2 – Hourly DA offers for each beta
# ═════════════════════════════════════════════════════════════════════════════

def fig2_offers(offers_df, path):
    hours = np.arange(1, N_HOURS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, scheme in zip(axes, SCHEMES):
        sub = offers_df[offers_df["Scheme"] == scheme]
        for i, beta in enumerate(BETAS):
            row = sub[sub["Beta"] == beta].sort_values("Hour")
            lw  = 2.2 if beta in (0.0, 10.0) else 1.2
            ls  = "-" if beta == 0.0 else ("--" if beta == 10.0 else "-")
            al  = 1.0 if beta in (0.0, 10.0) else 0.6
            ax.step(hours, row["DA_Offer_MW"].values, where="mid",
                    color=BETA_COLORS[i], lw=lw, ls=ls,
                    alpha=al, label=f"β={beta:g}")

        ax.axhline(CAPACITY_MW, color="black", ls=":", lw=0.9,
                   alpha=0.5, label=f"Cap ({int(CAPACITY_MW)} MW)")
        ax.set_xlim(1, N_HOURS)
        ax.set_ylim(-10, CAPACITY_MW * 1.05)
        ax.set_xlabel("Hour of day", fontsize=11)
        ax.set_ylabel("DA offer (MW)", fontsize=11)
        ax.set_title(scheme, fontsize=12, fontweight="bold")
        ax.set_xticks(range(1, N_HOURS + 1, 2))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7.5, ncol=2, loc="upper right")

    fig.suptitle("Hourly day-ahead offers by risk-aversion level (β)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"  Saved {path.name}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 3 – Profit distributions (box plots)
# ═════════════════════════════════════════════════════════════════════════════

def fig3_distributions(all_profits, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, scheme in zip(axes, SCHEMES):
        data   = [all_profits[(scheme, b)] / EUR_K for b in BETAS]
        labels = [f"β={b:g}" for b in BETAS]
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                        medianprops=dict(color="black", lw=2),
                        flierprops=dict(marker=".", ms=3, alpha=0.4),
                        whiskerprops=dict(lw=1.2),
                        capprops=dict(lw=1.2))
        for patch, color in zip(bp["boxes"], BETA_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        # mark CVaR (10th percentile, lower tail) for each beta
        for i, beta in enumerate(BETAS):
            profits = all_profits[(scheme, beta)]
            cvar_val = empirical_profit_cvar(profits, CVAR_ALPHA) / EUR_K
            ax.plot(i + 1, cvar_val, "v", color="black", ms=7, zorder=5)

        ax.set_xlabel("Risk-aversion level (β)", fontsize=11)
        ax.set_ylabel("Out-of-sample daily profit (k EUR)", fontsize=11)
        ax.set_title(scheme, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"Profit distribution vs risk aversion  (α = {CVAR_ALPHA:.2f},"
        f" out-of-sample)  ▼ = CVaR",
        fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"  Saved {path.name}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 4 – Fold robustness
# ═════════════════════════════════════════════════════════════════════════════

def fig4_fold_robustness(robustness_df, rob_offers_df, beta, path):
    hours       = np.arange(1, N_HOURS + 1)
    fold_colors = plt.cm.tab10(np.linspace(0, 0.8, CV_N_FOLDS))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ((ax_op_offer, ax_tp_offer),
     (ax_op_perf,  ax_tp_perf)) = axes

    for scheme, ax_offer, ax_perf in [
        ("One-price", ax_op_offer, ax_op_perf),
        ("Two-price", ax_tp_offer, ax_tp_perf),
    ]:
        sub_r = robustness_df[robustness_df["Scheme"] == scheme]
        sub_o = rob_offers_df[rob_offers_df["Scheme"] == scheme]

        # offer profiles
        for i, (_, row) in enumerate(sub_r.iterrows()):
            fold = row["Fold"]
            odf  = sub_o[sub_o["Fold"] == fold].sort_values("Hour")
            ax_offer.step(hours, odf["DA_Offer_MW"].values, where="mid",
                          color=fold_colors[i], lw=1.3, alpha=0.85,
                          label=f"Fold {fold}")

        ax_offer.axhline(CAPACITY_MW, color="black", ls=":", lw=0.9, alpha=0.5)
        ax_offer.set_xlim(1, N_HOURS)
        ax_offer.set_ylim(-10, CAPACITY_MW * 1.06)
        ax_offer.set_title(f"{scheme} – offers across folds (β={beta:g})",
                           fontsize=11, fontweight="bold")
        ax_offer.set_xlabel("Hour of day")
        ax_offer.set_ylabel("DA offer (MW)")
        ax_offer.set_xticks(range(1, N_HOURS + 1, 2))
        ax_offer.legend(fontsize=8, ncol=4)
        ax_offer.grid(True, alpha=0.3)

        # performance scatter
        xs = sub_r["Out_CVaR_EUR"].values / EUR_K
        ys = sub_r["Out_Expected_EUR"].values / EUR_K
        for i, (x, y, fold) in enumerate(
                zip(xs, ys, sub_r["Fold"].values)):
            ax_perf.scatter(x, y, color=fold_colors[i], s=80, zorder=3,
                            label=f"Fold {fold}")
            ax_perf.annotate(str(fold), (x, y),
                             textcoords="offset points", xytext=(4, 3),
                             fontsize=8)

        ax_perf.axvline(xs.mean(), color="gray", ls="--", lw=1, alpha=0.7)
        ax_perf.axhline(ys.mean(), color="gray", ls=":",  lw=1, alpha=0.7)
        ax_perf.set_title(
            f"{scheme} – out-of-sample performance per fold (β={beta:g})",
            fontsize=11, fontweight="bold")
        ax_perf.set_xlabel("Out-of-sample CVaR (k EUR)")
        ax_perf.set_ylabel("Out-of-sample E[profit] (k EUR)")
        ax_perf.legend(fontsize=8, ncol=4)
        ax_perf.grid(True, alpha=0.3)

    fig.suptitle(f"Fold robustness at β = {beta}  "
                 f"(α = {CVAR_ALPHA:.2f}, {CV_N_FOLDS} folds)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"  Saved {path.name}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 5 – Profit std-dev and worst-case vs beta
# ═════════════════════════════════════════════════════════════════════════════

def fig5_std_and_min(frontier, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, scheme in zip(axes, SCHEMES):
        sub   = frontier[frontier["Scheme"] == scheme].sort_values("Beta")
        betas = sub["Beta"].values

        ax2 = ax.twinx()
        l1, = ax.plot(betas, sub["Out_Std_EUR"] / EUR_K, "o-",
                      color="steelblue", lw=2, ms=7,
                      label="Std-dev (left)")
        l2, = ax2.plot(betas, sub["Out_Min_EUR"] / EUR_K, "s--",
                       color="tomato",    lw=2, ms=7,
                       label="Min profit (right)")

        ax.set_xlabel("β (risk-aversion weight)", fontsize=11)
        ax.set_ylabel("Profit std-dev (k EUR)", fontsize=11, color="steelblue")
        ax2.set_ylabel("Worst-case profit (k EUR)", fontsize=11, color="tomato")
        ax.tick_params(axis="y", colors="steelblue")
        ax2.tick_params(axis="y", colors="tomato")
        ax.set_title(scheme, fontsize=12, fontweight="bold")
        ax.set_xscale("symlog", linthresh=0.1)
        #ax.set_xticks(betas)
        ax.set_xticklabels([f"{b:.1f}" for b in betas])
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.grid(True, alpha=0.3)
        ax.legend([l1, l2], [l.get_label() for l in [l1, l2]], fontsize=9)

    fig.suptitle("Profit variability and downside risk vs β",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"  Saved {path.name}")


# ═════════════════════════════════════════════════════════════════════════════
# Numerical summary
# ═════════════════════════════════════════════════════════════════════════════

def print_summary(frontier, robustness_df):
    SEP = "=" * 76

    print(f"\n{SEP}")
    print(f"RISK-AVERSE FRONTIER  (out-of-sample, α = {CVAR_ALPHA:.2f})")
    print(SEP)
    cols = ["Scheme", "Beta",
            "In_Expected_EUR", "In_CVaR_EUR",
            "Out_Expected_EUR", "Out_CVaR_EUR",
            "Out_Std_EUR", "Out_Min_EUR"]
    fmt = frontier[cols].copy()
    for c in cols[2:]:
        fmt[c] = (frontier[c] / EUR_K).map("{:.2f}k".format)
    fmt.columns = ["Scheme", "β",
                   "In E[π]", "In CVaR",
                   "Out E[π]", "Out CVaR",
                   "Out Std", "Out Min"]
    print(fmt.to_string(index=False))

    print(f"\n{SEP}")
    print("RISK-AVERSION COST  (beta=10 vs beta=0, out-of-sample)")
    print(SEP)
    for scheme in SCHEMES:
        sub  = frontier[frontier["Scheme"] == scheme].set_index("Beta")
        d_ep = (sub.loc[0.0, "Out_Expected_EUR"] -
                sub.loc[10.0, "Out_Expected_EUR"]) / EUR_K
        d_cv = (sub.loc[10.0, "Out_CVaR_EUR"] -
                sub.loc[0.0,  "Out_CVaR_EUR"]) / EUR_K
        pct  = d_ep / (sub.loc[0.0, "Out_Expected_EUR"] / EUR_K) * 100
        print(f"  {scheme}: E[profit] drops {d_ep:.2f}k EUR ({pct:.2f}%),  "
              f"CVaR improves {d_cv:.2f}k EUR")

    print(f"\n{SEP}")
    print(f"FOLD ROBUSTNESS  (β = 1.0, {CV_N_FOLDS} folds)")
    print(SEP)
    for scheme in SCHEMES:
        sub = robustness_df[robustness_df["Scheme"] == scheme]
        ep  = sub["Out_Expected_EUR"].values / EUR_K
        cv  = sub["Out_CVaR_EUR"].values    / EUR_K
        mo  = sub["Mean_Offer_MW"].values
        print(f"\n  {scheme}")
        print(f"    Out E[profit]  : {ep.mean():.2f}k ± {ep.std(ddof=1):.2f}k  "
              f"range [{ep.min():.2f}k, {ep.max():.2f}k]")
        print(f"    Out CVaR       : {cv.mean():.2f}k ± {cv.std(ddof=1):.2f}k  "
              f"range [{cv.min():.2f}k, {cv.max():.2f}k]")
        print(f"    Mean offer     : {mo.mean():.1f} ± {mo.std(ddof=1):.1f} MW  "
              f"range [{mo.min():.1f}, {mo.max():.1f}] MW")

        # Coefficient of variation
        print(f"    CoV E[profit]  : {ep.std(ddof=1)/ep.mean()*100:.1f}%  "
              f"CoV CVaR: {cv.std(ddof=1)/cv.mean()*100:.1f}%")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("Building scenarios …")
    joint = build_scenarios()
    print(f"  Total joint scenarios: {len(joint)}")

    print("\n--- Frontier analysis ---")
    frontier, offers_df, all_profits, in_scen, out_scen = \
        run_cvar_frontier(joint)

    print("\n--- Fold robustness ---")
    robustness_df, rob_offers_df = run_fold_robustness(joint, beta=1.0)

    # ── save CSVs ─────────────────────────────────────────────────────────
    save_cols = [c for c in frontier.columns if c != "Offer_MW"]
    frontier[save_cols].to_csv(
        BASE_DIR / "risk_averse_cvar_frontier_fresh.csv", index=False)
    offers_df.to_csv(
        BASE_DIR / "risk_averse_cvar_offers_fresh.csv", index=False)
    rob_save = [c for c in robustness_df.columns if c != "Offer_MW"]
    robustness_df[rob_save].to_csv(
        BASE_DIR / "risk_averse_fold_robustness_fresh.csv", index=False)

    # ── figures ───────────────────────────────────────────────────────────
    print("\nGenerating figures …")
    fig1_frontier(frontier,       OUT / "ra_fig1_frontier.png")
    fig2_offers(offers_df,        OUT / "ra_fig2_offers.png")
    fig3_distributions(all_profits, OUT / "ra_fig3_distributions.png")
    fig4_fold_robustness(robustness_df, rob_offers_df, beta=1.0,
                         path=OUT / "ra_fig4_fold_robustness.png")
    fig5_std_and_min(frontier,    OUT / "ra_fig5_std_and_min.png")

    print_summary(frontier, robustness_df)
    print(f"\nDone.  Figures saved to: {OUT}")


if __name__ == "__main__":
    main()
