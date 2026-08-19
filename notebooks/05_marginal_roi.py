"""
Week 3 (extension): Marginal ROI analysis.

Average ROI (contribution / total spend) can be misleading for budget
decisions -- a channel can show high average ROI simply because it
receives little spend, while already sitting near saturation, meaning
the NEXT dollar there buys almost nothing. Marginal ROI (the local slope
of the response curve at current spend) is the correct metric for
"where should the next dollar go" decisions. This script computes both,
side by side, and tests a smaller (5%) reallocation to confirm that
following marginal ROI in small steps behaves as theory predicts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
from pymc_extras.prior import Prior

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

posterior = mmm.idata.posterior
sat_lam = posterior["saturation_lam"].mean(dim=["chain", "draw"]).to_pandas()
sat_beta = posterior["saturation_beta"].mean(dim=["chain", "draw"]).to_pandas()
scales = mmm.get_scales_as_xarray()
channel_scale = scales["channel_scale"].to_pandas()
target_scale = float(scales["target_scale"].values)

def hill_response(spend_scaled, lam, beta):
    sat = (1 - np.exp(-lam * spend_scaled)) / (1 + np.exp(-lam * spend_scaled))
    return beta * sat

def response_dollars(weekly_spend, ch):
    spend_scaled = weekly_spend / float(channel_scale[ch])
    return hill_response(spend_scaled, sat_lam[ch], sat_beta[ch]) * target_scale

print("=== MARGINAL ROI vs AVERAGE ROI (at current weekly spend) ===")
rows = []
for ch in channel_cols:
    current_weekly = train[ch].mean()
    eps = current_weekly * 0.01
    r0 = response_dollars(current_weekly, ch)
    r1 = response_dollars(current_weekly + eps, ch)
    marginal_roi = (r1 - r0) / eps
    avg_roi = r0 / current_weekly
    rows.append({
        "channel": channel_labels[ch],
        "current_weekly_spend": current_weekly,
        "marginal_roi": marginal_roi,
        "average_roi": avg_roi,
    })
    print(f"{channel_labels[ch]:15s}  weekly_spend=${current_weekly:6.1f}  "
          f"marginal_ROI={marginal_roi:6.2f}x  average_ROI={avg_roi:6.2f}x")

marginal_df = pd.DataFrame(rows).sort_values("marginal_roi", ascending=False)
marginal_df.to_csv("outputs/marginal_roi_by_channel.csv", index=False)
print("\nSaved: outputs/marginal_roi_by_channel.csv")

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(marginal_df))
width = 0.35
ax.bar(x - width/2, marginal_df["average_roi"], width, label="Average ROI", color="tab:blue", alpha=0.7)
ax.bar(x + width/2, marginal_df["marginal_roi"], width, label="Marginal ROI (next $)", color="tab:orange")
ax.set_xticks(x)
ax.set_xticklabels(marginal_df["channel"])
ax.set_ylabel("ROI (x)")
ax.set_title("Average ROI vs. Marginal ROI by Channel")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/figures/07_marginal_vs_average_roi.png", dpi=150)
print("Saved: outputs/figures/07_marginal_vs_average_roi.png")

def total_incremental_for_spend(spend_dict):
    total = 0
    for ch, total_spend in spend_dict.items():
        weekly_avg = total_spend / len(train)
        total += response_dollars(weekly_avg, ch) * len(train)
    return total

current_spend = {ch: train[ch].sum() for ch in channel_cols}
current_total = total_incremental_for_spend(current_spend)

print("\n=== SMALL vs LARGE REALLOCATION: TV -> Display ===")
for pct in [0.05, 0.15, 0.30, 0.50]:
    shift = current_spend["tv_spend"] * pct
    scen = dict(current_spend)
    scen["tv_spend"] -= shift
    scen["display_spend"] += shift
    total = total_incremental_for_spend(scen)
    pct_change = (total - current_total) / current_total * 100
    print(f"Shift {pct*100:>4.0f}% of TV budget to Display:  "
          f"incremental sales=${total:>12,.0f}  ({pct_change:+.2f}% vs current)")

print("\nDone.")
