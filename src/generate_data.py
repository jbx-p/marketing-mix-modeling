"""
Marketing Mix Modeling - Synthetic Data Generator
Generates 156 weeks (3 years) of weekly sales + channel spend data with
realistic trend, seasonality, adstock, and saturation baked in as ground truth.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# ---------------------------------------------------------------------
# 1. Time index
# ---------------------------------------------------------------------
N_WEEKS = 156
dates = pd.date_range(start="2022-01-03", periods=N_WEEKS, freq="W-MON")
week_num = np.arange(N_WEEKS)

# ---------------------------------------------------------------------
# 2. Adstock + saturation helper functions (also used later in modeling)
# ---------------------------------------------------------------------
def adstock(spend, decay_rate):
    """Geometric adstock: current + decayed carryover from prior weeks."""
    adstocked = np.zeros_like(spend, dtype=float)
    adstocked[0] = spend[0]
    for t in range(1, len(spend)):
        adstocked[t] = spend[t] + decay_rate * adstocked[t - 1]
    return adstocked

def saturate(x, alpha, gamma):
    """Hill saturation curve: diminishing returns, bounded [0,1] * scale later."""
    return x**alpha / (x**alpha + gamma**alpha)

# ---------------------------------------------------------------------
# 3. Channel spend patterns (weekly, in $000s)
# ---------------------------------------------------------------------
spend = pd.DataFrame(index=dates)

# TV: flighted bursts - on for ~6 weeks, off for ~6 weeks, varying intensity
tv = np.zeros(N_WEEKS)
t = 0
while t < N_WEEKS:
    on_weeks = np.random.randint(4, 8)
    intensity = np.random.uniform(60, 140)
    tv[t:t+on_weeks] = intensity + np.random.normal(0, 8, size=min(on_weeks, N_WEEKS - t))
    off_weeks = np.random.randint(4, 9)
    t += on_weeks + off_weeks
spend["tv"] = np.clip(tv, 0, None)

# Paid Search: steady with mild upward trend + noise
spend["paid_search"] = np.clip(
    25 + 0.08 * week_num + np.random.normal(0, 4, N_WEEKS), 5, None
)

# Paid Social: growing over time (reflects real-world budget shift trend)
spend["paid_social"] = np.clip(
    10 + 0.25 * week_num + np.random.normal(0, 5, N_WEEKS), 0, None
)

# Display: moderate, roughly flat
spend["display"] = np.clip(
    18 + np.random.normal(0, 3, N_WEEKS), 0, None
)

# Promotions: discrete promo weeks (roughly monthly, bigger around Nov/Dec)
promo = np.zeros(N_WEEKS)
promo_weeks = sorted(np.random.choice(N_WEEKS, size=22, replace=False))
for w in promo_weeks:
    promo[w] = np.random.uniform(15, 50)
# Boost promo spend/frequency around Black Friday & Christmas
for w in range(N_WEEKS):
    month = dates[w].month
    if month == 11 and dates[w].day >= 20:
        promo[w] = max(promo[w], np.random.uniform(60, 90))
spend["promotions"] = promo

# ---------------------------------------------------------------------
# 4. Control variables
# ---------------------------------------------------------------------
# Price index: dips during promo weeks
price_index = 100 - 0.15 * spend["promotions"].values + np.random.normal(0, 1, N_WEEKS)

# Distribution / ACV: slow upward trend (store rollout), plateaus later
distribution = np.clip(70 + 15 * (1 - np.exp(-week_num / 60)), 0, 100)

# One-off disruption: simulate a supply shock / stockout, weeks 60-65
disruption = np.zeros(N_WEEKS)
disruption[60:66] = -1  # flag weeks with negative demand shock

# Holiday flags
is_black_friday = np.array([(d.month == 11 and 22 <= d.day <= 28) for d in dates]).astype(int)
is_christmas = np.array([(d.month == 12 and 15 <= d.day <= 31) for d in dates]).astype(int)
is_back_to_school = np.array([(d.month == 8 and d.day >= 15) or (d.month == 9 and d.day <= 10) for d in dates]).astype(int)

# ---------------------------------------------------------------------
# 5. True media effects (ground truth, for later validation)
# ---------------------------------------------------------------------
true_params = {
    "tv":           {"decay": 0.55, "alpha": 2.0, "gamma": 300, "coef": 5500},
    "paid_search":  {"decay": 0.20, "alpha": 2.5, "gamma": 80,  "coef": 3200},
    "paid_social":  {"decay": 0.35, "alpha": 2.2, "gamma": 120, "coef": 2800},
    "display":      {"decay": 0.30, "alpha": 2.0, "gamma": 60,  "coef": 1400},
    "promotions":   {"decay": 0.10, "alpha": 1.8, "gamma": 40,  "coef": 4000},
}

channel_contribution = pd.DataFrame(index=dates)
for ch, p in true_params.items():
    ad = adstock(spend[ch].values, p["decay"])
    sat = saturate(ad, p["alpha"], p["gamma"])
    channel_contribution[ch] = sat * p["coef"]

# ---------------------------------------------------------------------
# 6. Baseline sales: trend + yearly seasonality + holiday bumps + noise
# ---------------------------------------------------------------------
trend = 40000 + 25 * week_num
yearly_seasonality = 4000 * np.sin(2 * np.pi * week_num / 52 + 0.3)
holiday_bump = (is_black_friday * 9000) + (is_christmas * 7000) + (is_back_to_school * 3000)
price_effect = -180 * (price_index - 100)
distribution_effect = 150 * (distribution - 70)
disruption_effect = disruption * 12000
noise = np.random.normal(0, 1800, N_WEEKS)

baseline = trend + yearly_seasonality + holiday_bump + price_effect + distribution_effect + disruption_effect

sales = baseline + channel_contribution.sum(axis=1).values + noise
sales = np.clip(sales, 1000, None)

# ---------------------------------------------------------------------
# 7. Assemble final dataframe
# ---------------------------------------------------------------------
df = pd.DataFrame({
    "date": dates,
    "sales": sales.round(0),
    "tv_spend": spend["tv"].round(1),
    "paid_search_spend": spend["paid_search"].round(1),
    "paid_social_spend": spend["paid_social"].round(1),
    "display_spend": spend["display"].round(1),
    "promotions_spend": spend["promotions"].round(1),
    "price_index": price_index.round(2),
    "distribution": distribution.round(1),
    "is_black_friday": is_black_friday,
    "is_christmas": is_christmas,
    "is_back_to_school": is_back_to_school,
    "disruption_flag": (disruption != 0).astype(int),
})

df.to_csv("data/mmm_weekly_data.csv", index=False)

ground_truth = pd.DataFrame(true_params).T
ground_truth.to_csv("data/ground_truth_params.csv")
channel_contribution["date"] = dates
channel_contribution.to_csv("data/ground_truth_contributions.csv", index=False)

print("Data generated:", df.shape)
print(df.head())
print("\nSpend totals ($000s) over full period:")
print(df[["tv_spend","paid_search_spend","paid_social_spend","display_spend","promotions_spend"]].sum())
print("\nTotal sales:", df["sales"].sum())
print("\nSaved: data/mmm_weekly_data.csv, data/ground_truth_params.csv, data/ground_truth_contributions.csv")
