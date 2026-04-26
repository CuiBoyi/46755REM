"""
Step 2 – Participation in Ancillary Service Markets (DK2 FCR-D UP)

FCR-D UP (Upward Frequency Containment Reserve – Disturbance):
  A flexible load (0–600 kW) provides upward reserve by reducing its
  consumption on demand. If the bid is b kW, it must be deliverable at
  every minute of the bidding hour.

Deliverability condition for scenario s:
    min_{t=1..60} c_s(t) >= b

P90 (Energinet) requirement: at least 90 % of scenarios must be deliverable.

Tasks
-----
2.1  In-sample optimisation:  ALSO-X (MILP) and CVaR (LP)
2.2  Out-of-sample verification (no optimisation)
2.3  Energinet perspective: sensitivity to reliability threshold (80–100 %)

Outputs (saved to outputs/):
  s2_fig1_profiles.png        sample load profiles
  s2_fig2_available_dist.png  min-consumption distribution + bid marks
  s2_fig3_oos_verification.png out-of-sample shortfall distributions
  s2_fig4_sensitivity.png     bid and shortfall vs reliability threshold
"""

import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gurobipy as gp
from gurobipy import GRB
from scipy.optimize import linprog

# ── configuration ─────────────────────────────────────────────────────────────
SEED        = 42
N_TOTAL     = 300
N_IN        = 100
N_OUT       = 200
T_MIN       = 60          # minutes per bidding hour
LO_KW       = 220.0       # minimum load (kW)
HI_KW       = 600.0       # maximum load (kW)
MAX_STEP    = 35.0        # max minute-to-minute change (kW)
ALPHA       = 0.90        # P90 reliability requirement

# Task 2.3 threshold sweep
THRESHOLDS  = np.round(np.arange(0.80, 1.001, 0.01), 3)

BASE_DIR = Path(__file__).resolve().parents[1]
OUT = BASE_DIR / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# Load profile generation
# ═════════════════════════════════════════════════════════════════════════════

def generate_profiles(n=N_TOTAL, T=T_MIN, lo=LO_KW, hi=HI_KW,
                      max_step=MAX_STEP, seed=SEED):
    """
    Realistic clipped random-walk load profiles.

    Start:  Uniform[lo + 80, hi - 50]  (well inside bounds, ~350–550 kW)
    Steps:  Normal(0, 12 kW) clipped to [−max_step, +max_step]
            → typical variation ≈ 10 kW/min, hard cap at 35 kW  (spec met)
    Bounds: clipped to [lo, hi] = [220, 600] kW

    This produces non-degenerate min-consumptions (≈ 250–450 kW range) so
    that the P90 bid is a meaningful intermediate value, not just the floor.
    """
    rng = np.random.default_rng(seed)
    c = np.empty((n, T))
    c[:, 0] = rng.uniform(lo + 80, hi - 50, size=n)   # start 300–550 kW
    for t in range(1, T):
        steps = np.clip(rng.normal(0.0, 12.0, size=n), -max_step, max_step)
        c[:, t] = np.clip(c[:, t - 1] + steps, lo, hi)
    return c


# ═════════════════════════════════════════════════════════════════════════════
# Task 2.1 – ALSO-X  (chance-constraint MILP via Big-M)
# ═════════════════════════════════════════════════════════════════════════════

def solve_alsox(available, p=ALPHA):
    """
    Maximise b subject to P(min_t c(t) >= b) >= p.

    MILP formulation (ALSO-X):
        max  b
        s.t. b  <=  avail_s + M*(1 - z_s)   for all s
             sum(z_s) >= ceil(p * N)
             z_s in {0,1},  0 <= b <= HI_KW

    z_s = 1  iff scenario s is required to be deliverable.
    M = HI_KW  (safe big-M: b <= HI_KW and avail_s >= LO_KW > 0).
    """
    N   = len(available)
    req = int(np.ceil(p * N))     # minimum number of satisfied scenarios
    M   = float(HI_KW)

    m = gp.Model("FCRD_ALSOX")
    m.Params.OutputFlag = 0

    b = m.addVar(lb=0.0, ub=M, name="b")
    z = m.addVars(N, vtype=GRB.BINARY, name="z")

    for s in range(N):
        m.addConstr(b <= available[s] + M * (1.0 - z[s]),
                    name=f"delivery_{s}")
    m.addConstr(gp.quicksum(z[s] for s in range(N)) >= req,
                name="reliability")

    m.setObjective(b, GRB.MAXIMIZE)
    m.optimize()

    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"ALSO-X: Gurobi status {m.Status}")

    b_star = float(b.X)
    z_star = np.array([float(z[s].X) for s in range(N)])
    return b_star, z_star


# ═════════════════════════════════════════════════════════════════════════════
# Task 2.1 – CVaR (LP approximation of chance constraint)
# ═════════════════════════════════════════════════════════════════════════════

def solve_cvar(available, alpha=ALPHA):
    """
    Maximise b subject to CVaR_alpha(b - avail_s) <= 0.

    CVaR_alpha(X) <= 0  is a convex (sufficient) approximation of the
    P(X > 0) <= 1-alpha chance constraint.  It requires the EXPECTED
    shortfall in the worst (1-alpha) fraction to be non-positive.

    LP (variables: b, zeta, eta_0,...,eta_{N-1}):
        max   b
        s.t.  zeta + 1/((1-alpha)*N) * sum(eta_s) <= 0       [CVaR <= 0]
              b - avail_s - zeta  <=  eta_s         for all s [shortfall]
              eta_s >= 0
              0 <= b <= HI_KW,  zeta free
    """
    N     = len(available)
    NV    = 2 + N          # b, zeta, eta_0..eta_{N-1}

    # Objective: min -b
    c_obj       = np.zeros(NV)
    c_obj[0]    = -1.0

    # Constraints A_ub x <= b_ub
    # (1)  zeta + sum(eta)/(( 1-alpha)*N) <= 0
    # (2s) b - zeta - eta_s <= avail_s    for s=0..N-1
    A = np.zeros((1 + N, NV))
    rhs = np.zeros(1 + N)

    A[0, 1]  = 1.0                            # zeta
    A[0, 2:] = 1.0 / ((1.0 - alpha) * N)     # eta coefficients

    for s in range(N):
        A[1 + s, 0]     =  1.0    # b
        A[1 + s, 1]     = -1.0    # -zeta
        A[1 + s, 2 + s] = -1.0    # -eta_s
        rhs[1 + s]      = available[s]

    bounds = [(0.0, float(HI_KW)),  # b
              (None, None)]         # zeta
    bounds += [(0.0, None)] * N     # eta >= 0

    res = linprog(c_obj, A_ub=A, b_ub=rhs, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(f"CVaR LP: {res.message}")

    b_star   = float(res.x[0])
    zeta_star = float(res.x[1])
    cvar_val  = zeta_star + np.sum(np.maximum(0.0, res.x[0] - available - zeta_star)) \
                / ((1.0 - alpha) * N)
    return b_star, zeta_star, cvar_val


# ═════════════════════════════════════════════════════════════════════════════
# Out-of-sample evaluation
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_oos(available_oos, b):
    """Compute delivery probability and shortfall statistics."""
    shortfall = np.maximum(0.0, b - available_oos)
    return {
        "bid_kW":           b,
        "prob_delivery":    float(np.mean(shortfall == 0.0)),
        "prob_shortfall":   float(np.mean(shortfall > 0.0)),
        "exp_shortfall_kW": float(np.mean(shortfall)),
        "max_shortfall_kW": float(np.max(shortfall)),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Figure 1 – Sample load profiles
# ═════════════════════════════════════════════════════════════════════════════

def fig1_profiles(profiles_in, profiles_oos, path, n_show=12):
    from matplotlib.patches import Patch

    minutes = np.arange(1, T_MIN + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    # Stacked percentile bands: outer → inner, cumulative fill gives fade effect
    # alpha tuned so centre reaches ~0.45 opacity and edges stay ~0.04
    band_specs = [
        (5,  95, 0.04),
        (10, 90, 0.07),
        (20, 80, 0.10),
        (30, 70, 0.13),
        (40, 60, 0.16),
    ]

    for ax, profiles, cmap_name, label in [
        (axes[0], profiles_in,  "Blues",   "In-sample"),
        (axes[1], profiles_oos, "Oranges", "Out-of-sample"),
    ]:
        color_fill = plt.cm.get_cmap(cmap_name)(0.55)
        color_mean = plt.cm.get_cmap(cmap_name)(0.88)
        color_line = plt.cm.get_cmap(cmap_name)(0.50)

        # Pre-compute all needed percentiles
        pct = {p: np.percentile(profiles, p, axis=0)
               for p in [5, 10, 20, 30, 40, 60, 70, 80, 90, 95]}

        # Paint bands outer → inner (cumulative stacking produces fade)
        for lo_p, hi_p, a in band_specs:
            ax.fill_between(minutes, pct[lo_p], pct[hi_p],
                            color=color_fill, alpha=a, linewidth=0)

        # A few individual profile traces (very faded, texture only)
        idx = np.linspace(0, len(profiles) - 1, n_show, dtype=int)
        for i in idx:
            ax.plot(minutes, profiles[i], color=color_line,
                    lw=0.65, alpha=0.18, zorder=2)

        # Mean curve — bold, on top
        mean_curve = profiles.mean(axis=0)
        h_mean, = ax.plot(minutes, mean_curve, color=color_mean,
                          lw=2.5, zorder=5)

        # Physical bounds
        ax.axhline(LO_KW, color="black", ls="--", lw=1.1)
        ax.axhline(HI_KW, color="black", ls=":",  lw=1.1)

        ax.set_xlabel("Minute of hour", fontsize=11)
        ax.set_ylabel("Consumption (kW)", fontsize=11)
        ax.set_title(f"{label}  (N = {len(profiles)})",
                     fontsize=12, fontweight="bold")
        ax.set_xlim(1, T_MIN)
        ax.set_ylim(LO_KW - 20, HI_KW + 20)
        ax.legend(handles=[
            h_mean,
            Patch(facecolor=color_fill, alpha=0.45, label="40–60th percentile"),
            Patch(facecolor=color_fill, alpha=0.25,  label="20–80th percentile"),
            Patch(facecolor=color_fill, alpha=0.10,  label="5–95th percentile"),
            plt.Line2D([0], [0], color="black", ls="--", lw=1.1,
                       label=f"Min floor ({int(LO_KW)} kW)"),
            plt.Line2D([0], [0], color="black", ls=":",  lw=1.1,
                       label=f"Max cap ({int(HI_KW)} kW)"),
        ], labels=[
            "Mean",
            "40–60th percentile",
            "20–80th percentile",
            "5–95th percentile",
            f"Min floor ({int(LO_KW)} kW)",
            f"Max cap ({int(HI_KW)} kW)",
        ], fontsize=8.5, loc="upper right")
        ax.grid(True, alpha=0.25)

    fig.suptitle("FCR-D UP flexible load profiles — probability bands + mean",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"  Saved {path.name}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 2 – Distribution of available headroom (min consumption)
# ═════════════════════════════════════════════════════════════════════════════

def fig2_available(available_in, available_oos, b_alsox, b_cvar, path):
    fig, ax = plt.subplots(figsize=(9, 4.5))

    bins = np.linspace(LO_KW, HI_KW, 40)
    ax.hist(available_in,  bins=bins, alpha=0.6, color="steelblue",
            label=f"In-sample  (N={N_IN})",  density=True)
    ax.hist(available_oos, bins=bins, alpha=0.5, color="orange",
            label=f"Out-of-sample (N={N_OUT})", density=True)

    ax.axvline(b_alsox, color="darkblue",   lw=2.2, ls="-",
               label=f"ALSO-X bid = {b_alsox:.1f} kW")
    ax.axvline(b_cvar,  color="crimson",    lw=2.2, ls="--",
               label=f"CVaR  bid  = {b_cvar:.1f} kW")

    ax.set_xlabel("Min-consumption (headroom) per scenario (kW)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Distribution of available FCR-D UP headroom  "
                 f"(P90 = {int(ALPHA*100)}%)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"  Saved {path.name}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 3 – Out-of-sample verification
# ═════════════════════════════════════════════════════════════════════════════

def fig3_oos_verification(available_oos, b_alsox, b_cvar, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)

    for ax, b, label, color in [
        (axes[0], b_alsox, "ALSO-X", "steelblue"),
        (axes[1], b_cvar,  "CVaR",   "crimson"),
    ]:
        shortfalls = np.maximum(0.0, b - available_oos)
        n_short = int(np.sum(shortfalls > 0))
        n_ok    = N_OUT - n_short

        # Histogram of shortfalls (including zero)
        ax.hist(shortfalls[shortfalls > 0], bins=20, color=color,
                alpha=0.75, label=f"Shortfall > 0  (n={n_short})")
        ax.axvline(0, color="black", lw=1.2, ls="--")

        pct_ok = 100.0 * n_ok / N_OUT
        ax.set_title(f"{label}  —  bid = {b:.1f} kW\n"
                     f"Delivered in {n_ok}/{N_OUT} scenarios ({pct_ok:.1f}%)",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Reserve shortfall (kW)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        if n_short == 0:
            ax.text(0.5, 0.5, "No shortfalls", transform=ax.transAxes,
                    ha="center", va="center", fontsize=14, color=color,
                    fontweight="bold")

    fig.suptitle(f"Out-of-sample P90 verification  (N_oos = {N_OUT})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"  Saved {path.name}")


# ═════════════════════════════════════════════════════════════════════════════
# Figure 4 – Task 2.3 sensitivity to reliability threshold
# ═════════════════════════════════════════════════════════════════════════════

def fig4_sensitivity(sens_df, path):
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax2 = ax1.twinx()

    pct = sens_df["Threshold"] * 100

    l1, = ax1.plot(pct, sens_df["Bid_kW"], "o-",
                   color="steelblue", lw=2.2, ms=6,
                   label="Optimal bid (left)")
    l2, = ax2.plot(pct, sens_df["OOS_Exp_Shortfall_kW"], "s--",
                   color="tomato",    lw=2.2, ms=6,
                   label="OOS exp. shortfall (right)")
    l3, = ax2.plot(pct, sens_df["OOS_Prob_Shortfall"] * 100, "^:",
                   color="darkorange", lw=1.8, ms=6,
                   label="OOS shortfall probability % (right)")

    # Mark P90
    ax1.axvline(90, color="gray", ls="--", lw=1.2, alpha=0.7,
                label="P90 reference")

    ax1.set_xlabel("Reliability requirement threshold (%)", fontsize=11)
    ax1.set_ylabel("Optimal reserve bid (kW)", fontsize=11, color="steelblue")
    ax2.set_ylabel("Out-of-sample shortfall metric", fontsize=11, color="tomato")
    ax1.tick_params(axis="y", colors="steelblue")
    ax2.tick_params(axis="y", colors="tomato")

    lines = [l1, l2, l3]
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=9, loc="center left")

    ax1.set_title("Energinet perspective: bid and shortfall vs reliability threshold",
                  fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"  Saved {path.name}")


# ═════════════════════════════════════════════════════════════════════════════
# Print summary
# ═════════════════════════════════════════════════════════════════════════════

def print_summary(available_in, available_oos,
                  b_alsox, b_cvar, cvar_val,
                  oos_alsox, oos_cvar):
    SEP = "=" * 68
    print(f"\n{SEP}")
    print("SCENARIO STATISTICS")
    print(SEP)
    print(f"  In-sample  min headroom : "
          f"{available_in.min():.1f} – {available_in.max():.1f} kW  "
          f"  mean={available_in.mean():.1f} kW")
    print(f"  Out-of-sample min head. : "
          f"{available_oos.min():.1f} – {available_oos.max():.1f} kW  "
          f"  mean={available_oos.mean():.1f} kW")
    print(f"  P90 in-sample headroom  : "
          f"{np.percentile(available_in, 10):.1f} kW  "
          f"(10th percentile)")

    print(f"\n{SEP}")
    print("TASK 2.1 — IN-SAMPLE OPTIMAL BIDS (P90 constraint)")
    print(SEP)
    print(f"  ALSO-X (MILP)  bid = {b_alsox:.2f} kW")
    print(f"  CVaR   (LP)    bid = {b_cvar:.2f} kW  "
          f"  in-sample CVaR = {cvar_val:.2f} kW")
    print(f"\n  Analytical check: 10th percentile of in-sample available  "
          f"= {np.percentile(available_in, 10):.2f} kW")
    print(f"  Analytical CVaR check: mean of 10 smallest available        "
          f"= {np.sort(available_in)[:int(np.ceil(0.10*N_IN))].mean():.2f} kW")

    print(f"\n{SEP}")
    print("TASK 2.2 — OUT-OF-SAMPLE VERIFICATION (N_oos = 200)")
    print(SEP)
    for label, res in [("ALSO-X", oos_alsox), ("CVaR", oos_cvar)]:
        print(f"\n  {label}  (bid = {res['bid_kW']:.2f} kW)")
        print(f"    Delivery probability  : {res['prob_delivery']*100:.1f}%  "
              f"(Energinet P90 = 90.0%)")
        print(f"    Shortfall probability : {res['prob_shortfall']*100:.1f}%")
        print(f"    Expected shortfall    : {res['exp_shortfall_kW']:.2f} kW")
        print(f"    Max shortfall         : {res['max_shortfall_kW']:.2f} kW")
        p90_met = "✓  MET" if res["prob_delivery"] >= ALPHA else "✗  NOT MET"
        print(f"    P90 requirement OOS   : {p90_met}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    # ── generate profiles ─────────────────────────────────────────────────
    print("Generating load profiles …")
    all_profiles = generate_profiles()
    profiles_in  = all_profiles[:N_IN]
    profiles_oos = all_profiles[N_IN:]

    # Verify constraints
    assert profiles_in.shape  == (N_IN,  T_MIN)
    assert profiles_oos.shape == (N_OUT, T_MIN)
    assert np.all(profiles_in  >= LO_KW - 1e-9) and np.all(profiles_in  <= HI_KW + 1e-9)
    assert np.all(profiles_oos >= LO_KW - 1e-9) and np.all(profiles_oos <= HI_KW + 1e-9)
    diffs_in  = np.abs(np.diff(profiles_in,  axis=1))
    diffs_oos = np.abs(np.diff(profiles_oos, axis=1))
    assert diffs_in.max()  <= MAX_STEP + 1e-9, f"Step constraint violated in-sample: {diffs_in.max():.2f}"
    assert diffs_oos.max() <= MAX_STEP + 1e-9, f"Step constraint violated OOS: {diffs_oos.max():.2f}"
    print(f"  Profiles OK — in-sample max step: {diffs_in.max():.2f} kW, "
          f"OOS max step: {diffs_oos.max():.2f} kW")

    # Available headroom per scenario = min consumption over 60 minutes
    available_in  = profiles_in.min(axis=1)
    available_oos = profiles_oos.min(axis=1)

    # ── Task 2.1 ──────────────────────────────────────────────────────────
    print("\n--- Task 2.1: In-sample optimisation (P90) ---")
    print("  Solving ALSO-X … ", end="", flush=True)
    b_alsox, z_alsox = solve_alsox(available_in, p=ALPHA)
    print(f"bid = {b_alsox:.2f} kW")

    print("  Solving CVaR   … ", end="", flush=True)
    b_cvar, zeta_cvar, cvar_val = solve_cvar(available_in, alpha=ALPHA)
    print(f"bid = {b_cvar:.2f} kW")

    # ── Task 2.2 ──────────────────────────────────────────────────────────
    print("\n--- Task 2.2: Out-of-sample verification ---")
    oos_alsox = evaluate_oos(available_oos, b_alsox)
    oos_cvar  = evaluate_oos(available_oos, b_cvar)
    print(f"  ALSO-X: delivery {oos_alsox['prob_delivery']*100:.1f}%  "
          f"exp shortfall {oos_alsox['exp_shortfall_kW']:.2f} kW")
    print(f"  CVaR:   delivery {oos_cvar['prob_delivery']*100:.1f}%  "
          f"exp shortfall {oos_cvar['exp_shortfall_kW']:.2f} kW")

    # ── Task 2.3 ──────────────────────────────────────────────────────────
    print(f"\n--- Task 2.3: Sensitivity to threshold ({THRESHOLDS[0]:.0%} – "
          f"{THRESHOLDS[-1]:.0%}) ---")
    sens_records = []
    for p in THRESHOLDS:
        b_p, _ = solve_alsox(available_in, p=p)
        sf      = np.maximum(0.0, b_p - available_oos)
        sens_records.append({
            "Threshold":           p,
            "Bid_kW":              b_p,
            "OOS_Prob_Delivery":   float(np.mean(sf == 0)),
            "OOS_Prob_Shortfall":  float(np.mean(sf > 0)),
            "OOS_Exp_Shortfall_kW": float(np.mean(sf)),
            "OOS_Max_Shortfall_kW": float(np.max(sf)),
        })
        print(f"  p={p:.2f}  bid={b_p:.1f} kW  "
              f"OOS delivery={np.mean(sf==0)*100:.1f}%  "
              f"exp shortfall={np.mean(sf):.2f} kW")

    sens_df = pd.DataFrame(sens_records)

    # ── Save CSVs ─────────────────────────────────────────────────────────
    sens_df.to_csv(BASE_DIR / "step2_sensitivity.csv", index=False)
    pd.DataFrame({
        "Minute": np.arange(1, T_MIN + 1),
        **{f"in_{i+1}": profiles_in[i] for i in range(min(5, N_IN))},
        **{f"oos_{i+1}": profiles_oos[i] for i in range(min(5, N_OUT))},
    }).to_csv(BASE_DIR / "step2_sample_profiles.csv", index=False)

    # ── Figures ───────────────────────────────────────────────────────────
    print("\nGenerating figures …")
    fig1_profiles(profiles_in, profiles_oos,
                  OUT / "s2_fig1_profiles.png")
    fig2_available(available_in, available_oos, b_alsox, b_cvar,
                   OUT / "s2_fig2_available_dist.png")
    fig3_oos_verification(available_oos, b_alsox, b_cvar,
                          OUT / "s2_fig3_oos_verification.png")
    fig4_sensitivity(sens_df, OUT / "s2_fig4_sensitivity.png")

    # ── Summary ───────────────────────────────────────────────────────────
    print_summary(available_in, available_oos,
                  b_alsox, b_cvar, cvar_val,
                  oos_alsox, oos_cvar)

    print(f"\nDone. Figures saved to: {OUT}")


if __name__ == "__main__":
    main()
