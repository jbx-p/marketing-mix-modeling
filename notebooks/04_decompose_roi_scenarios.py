"""
Week 3: Decomposition, ROI, response curves, and budget reallocation scenarios.

Loads the fitted model from Week 2 (re-fits, since re-loading an MMM's full
sampling context from disk is finicky across pymc-marketing versions -- for
a portfolio project, re-fitting with the same settings is simpler and more
reliable than serialization workarounds).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
from pymc_extras.prior import Prior

# ---------------------------------------------------------------------
# 1. Load data and refit (see docstring note on why we refit vs reload)
# ---------------------------------------------------------------------
df = pd.read_csv("data/mmm_weekly_data.csv", parse_dates=["date"])

channel_cols = [
    "tv_spend", "paid_search_spend", "paid_social_spend",
    "display_spend", "promotions_spend",
]
channel_labels = {
    "tv_spend": "TV",
    "paid_search_spend": "Paid Search",
    "paid_social_spend": "Paid Social",
    "display_spend": "Display",
    "promotions_spend": "Promotions",
}
control_cols = [
    "price_index", "distribution",
    "is_black_friday", "is_christmas", "is_back_to_school", "disruption_flag",
]

HOLDOUT_WEEKS = 12
train = df.iloc[:-HOLDOUT_WEEKS].reset_index(drop=True)

mmm = MMM(
    date_column="date",
    channel_columns=channel_cols,
    control_columns=control_cols,
    target_column="sales",
    adstock=GeometricAdstock(l_max=8),
    saturation=LogisticSaturation(
        priors={
            "lam": Prior("Gamma", alpha=3, beta=1),
            "beta": Prior("HalfNormal", sigma=0.025),
        }
    ),
    yearly_seasonality=2,
)

X_train = train[["date"] + channel_cols + control_cols]
y_train = train["sales"]

print("Fitting model...")
mmm.fit(
    X_train, y_train,
    target_accept=0.95,
    chains=4,
    cores=1,
    draws=1000,
    tune=1000,
    progressbar=True,
)
print("Fit complete.\n")

# ---------------------------------------------------------------------
# 2. Decompose sales into baseline vs incremental (per channel)
# ---------------------------------------------------------------------
contrib_df = mmm.compute_mean_contributions_over_time(central_tendency="mean")
contrib_df = contrib_df.merge(train[["date"]], on="date")

baseline_cols = ["intercept", "yearly_seasonality"] + control_cols
baseline_cols = [c for c in baseline_cols if c in contrib_df.columns]

contrib_df["baseline_total"] = contrib_df[baseline_cols].sum(axis=1)
contrib_df["incremental_total"] = contrib_df[channel_cols].sum(axis=1)
contrib_df["predicted_total"] = contrib_df["baseline_total"] + contrib_df["incremental_total"]

total_baseline = contrib_df["baseline_total"].sum()
total_incremental = contrib_df["incremental_total"].sum()
total_sales_actual = y_train.sum()

print("=== SALES DECOMPOSITION (training period) ===")
print(f"Total actual sales:        ${total_sales_actual:,.0f}")
print(f"Baseline (no marketing):   ${total_baseline:,.0f}  ({total_baseline/total_sales_actual*100:.1f}%)")
print(f"Incremental (from media):  ${total_incremental:,.0f}  ({total_incremental/total_sales_actual*100:.1f}%)")

# ---------------------------------------------------------------------
# 3. ROI per channel: incremental sales generated per dollar spent
# ---------------------------------------------------------------------
print("\n=== ROI BY CHANNEL ===")
roi_rows = []
for ch in channel_cols:
    total_contribution = contrib_df[ch].sum()
    total_spend = train[ch].sum()
    roi = total_contribution / total_spend if total_spend > 0 else np.nan
    roi_rows.append({
        "channel": channel_labels[ch],
        "total_spend": total_spend,
        "total_contribution": total_contribution,
        "roi": roi,
    })
    print(f"{channel_labels[ch]:15s}  spend=${total_spend:>10,.0f}  "
          f"contribution=${total_contribution:>10,.0f}  ROI={roi:.2f}x")

roi_df = pd.DataFrame(roi_rows).sort_values("roi", ascending=False)
roi_df.to_csv("outputs/roi_by_channel.csv", index=False)
print("\nSaved: outputs/roi_by_channel.csv")

# ---------------------------------------------------------------------
# 4. ROI bar chart
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
colors = ["tab:green" if r >= 1 else "tab:red" for r in roi_df["roi"]]
ax.barh(roi_df["channel"], roi_df["roi"], color=colors)
ax.axvline(1.0, color="black", linestyle="--", linewidth=1, label="Break-even (ROI=1.0)")
ax.set_xlabel("ROI ($ incremental sales per $ spent)")
ax.set_title("Marketing ROI by Channel")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/figures/04_roi_by_channel.png", dpi=150)
print("Saved: outputs/figures/04_roi_by_channel.png")

# ---------------------------------------------------------------------
# 5. Response (saturation) curves per channel
# ---------------------------------------------------------------------
posterior = mmm.idata.posterior
sat_lam = posterior["saturation_lam"].mean(dim=["chain", "draw"]).to_pandas()
sat_beta = posterior["saturation_beta"].mean(dim=["chain", "draw"]).to_pandas()

scales = mmm.get_scales_as_xarray()
channel_scale = scales["channel_scale"].to_pandas()
target_scale = float(scales["target_scale"].values)

def hill_response(spend_scaled, lam, beta):
    sat = (1 - np.exp(-lam * spend_scaled)) / (1 + np.exp(-lam * spend_scaled))
    return beta * sat

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

response_curve_data = {}
for i, ch in enumerate(channel_cols):
    max_spend = train[ch].max() * 1.5
    spend_range = np.linspace(0, max_spend, 100)
    spend_scaled = spend_range / float(channel_scale[ch])
    response_scaled = hill_response(spend_scaled, sat_lam[ch], sat_beta[ch])
    response_dollars = response_scaled * target_scale

    response_curve_data[ch] = pd.DataFrame({
        "spend": spend_range,
        "response": response_dollars,
    })

    axes[i].plot(spend_range, response_dollars, color="tab:blue")
    current_spend = train[ch].mean()
    current_idx = np.searchsorted(spend_range, current_spend)
    if current_idx < len(response_dollars):
        axes[i].scatter([current_spend], [response_dollars[current_idx]],
                         color="red", zorder=5, label="Current avg. weekly spend")
    axes[i].set_title(channel_labels[ch])
    axes[i].set_xlabel("Weekly spend ($000s)")
    axes[i].set_ylabel("Response ($ sales)")
    axes[i].legend(fontsize=8)

axes[-1].axis("off")
plt.suptitle("Response Curves: Diminishing Returns by Channel", fontsize=14)
plt.tight_layout()
plt.savefig("outputs/figures/05_response_curves.png", dpi=150)
print("Saved: outputs/figures/05_response_curves.png")

# ---------------------------------------------------------------------
# 6. Budget reallocation scenarios
# ---------------------------------------------------------------------
def total_incremental_for_spend(spend_dict):
    total = 0
    for ch, total_spend in spend_dict.items():
        weekly_avg = total_spend / len(train)
        spend_scaled = weekly_avg / float(channel_scale[ch])
        response_scaled = hill_response(spend_scaled, sat_lam[ch], sat_beta[ch])
        response_dollars = response_scaled * target_scale
        total += response_dollars * len(train)
    return total

current_spend = {ch: train[ch].sum() for ch in channel_cols}
current_total_incremental = total_incremental_for_spend(current_spend)

scenarios = {
    "Current allocation": current_spend,
}

shift_pct = 0.15
shift_amount = current_spend["tv_spend"] * shift_pct
scenario_a = dict(current_spend)
scenario_a["tv_spend"] -= shift_amount
scenario_a["paid_social_spend"] += shift_amount
scenarios["Shift 15% TV -> Paid Social"] = scenario_a

shift_amount_b = current_spend["display_spend"] * shift_pct
scenario_b = dict(current_spend)
scenario_b["display_spend"] -= shift_amount_b
scenario_b["paid_search_spend"] += shift_amount_b
scenarios["Shift 15% Display -> Paid Search"] = scenario_b

shift_pct_c = 0.30
shift_amount_c = current_spend["tv_spend"] * shift_pct_c
scenario_c = dict(current_spend)
scenario_c["tv_spend"] -= shift_amount_c
scenario_c["display_spend"] += shift_amount_c
scenarios["Shift 30% TV -> Display (lowest ROI -> highest ROI)"] = scenario_c

print("\n=== BUDGET REALLOCATION SCENARIOS ===")
scenario_results = []
for name, spend_dict in scenarios.items():
    total_inc = total_incremental_for_spend(spend_dict)
    pct_change = (total_inc - current_total_incremental) / current_total_incremental * 100
    scenario_results.append({
        "scenario": name,
        "total_incremental_sales": total_inc,
        "pct_change_vs_current": pct_change,
    })
    print(f"{name:35s}  incremental sales=${total_inc:>12,.0f}  "
          f"({pct_change:+.2f}% vs current)")

scenario_df = pd.DataFrame(scenario_results)
scenario_df.to_csv("outputs/budget_scenarios.csv", index=False)
print("\nSaved: outputs/budget_scenarios.csv")

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(scenario_df["scenario"], scenario_df["total_incremental_sales"], color="tab:blue")
ax.set_ylabel("Total incremental sales ($)")
ax.set_title("Budget Reallocation Scenario Comparison")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig("outputs/figures/06_scenario_comparison.png", dpi=150)
print("Saved: outputs/figures/06_scenario_comparison.png")

print("\nDone. Review outputs/roi_by_channel.csv and outputs/budget_scenarios.csv for the memo.")
