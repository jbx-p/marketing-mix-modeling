import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/mmm_weekly_data.csv", parse_dates=["date"])

fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)

# 1. Sales over time
axes[0].plot(df["date"], df["sales"], color="black", linewidth=1.2)
axes[0].axvspan(df["date"].iloc[60], df["date"].iloc[65], color="red", alpha=0.15, label="Disruption event")
axes[0].set_title("Weekly Sales")
axes[0].legend()

# 2. Channel spend, stacked
channels = ["tv_spend", "paid_search_spend", "paid_social_spend", "display_spend", "promotions_spend"]
axes[1].stackplot(df["date"], [df[c] for c in channels], labels=channels, alpha=0.8)
axes[1].set_title("Channel Spend ($000s, stacked)")
axes[1].legend(loc="upper left", fontsize=8)

# 3. Price index and distribution
ax3b = axes[2].twinx()
axes[2].plot(df["date"], df["price_index"], color="tab:blue", label="Price Index")
ax3b.plot(df["date"], df["distribution"], color="tab:green", label="Distribution")
axes[2].set_title("Price Index (left) vs Distribution (right)")
axes[2].set_ylabel("Price Index", color="tab:blue")
ax3b.set_ylabel("Distribution", color="tab:green")

# 4. Holiday markers on sales
axes[3].plot(df["date"], df["sales"], color="black", linewidth=1)
bf = df[df["is_black_friday"] == 1]
xmas = df[df["is_christmas"] == 1]
bts = df[df["is_back_to_school"] == 1]
axes[3].scatter(bf["date"], bf["sales"], color="orange", label="Black Friday", zorder=5)
axes[3].scatter(xmas["date"], xmas["sales"], color="red", label="Christmas", zorder=5)
axes[3].scatter(bts["date"], bts["sales"], color="purple", label="Back to School", zorder=5)
axes[3].set_title("Sales with Holiday Weeks Marked")
axes[3].legend()

plt.tight_layout()
plt.savefig("outputs/figures/01_raw_data_overview.png", dpi=150)
print("Saved: outputs/figures/01_raw_data_overview.png")

# Quick correlation check: each channel's raw spend vs sales (same week, no adstock yet)
print("\nRaw same-week correlation with sales (expect these to UNDERSTATE true effect, since adstock/lag not applied yet):")
print(df[channels + ["sales"]].corr()["sales"].drop("sales").sort_values(ascending=False))
