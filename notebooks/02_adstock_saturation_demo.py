"""
Adstock & Saturation Demo
Illustrates, on a single synthetic spend series, why MMM differs from
plain regression: (1) adstock = carryover of media effect over time,
(2) saturation = diminishing returns as spend increases.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(1)

# ---------------------------------------------------------------------
# A simple hypothetical spend series: a single 4-week TV burst
# ---------------------------------------------------------------------
weeks = np.arange(20)
spend = np.zeros(20)
spend[5:9] = [100, 100, 100, 100]   # a 4-week campaign burst

# ---------------------------------------------------------------------
# 1. Adstock: geometric decay of media effect over time
#    adstock[t] = spend[t] + decay_rate * adstock[t-1]
#    decay_rate close to 1 => effect lingers a long time (e.g. TV, brand awareness)
#    decay_rate close to 0 => effect is almost immediate (e.g. flash sale)
# ---------------------------------------------------------------------
def adstock(spend, decay_rate):
    out = np.zeros_like(spend, dtype=float)
    out[0] = spend[0]
    for t in range(1, len(spend)):
        out[t] = spend[t] + decay_rate * out[t - 1]
    return out

adstock_low  = adstock(spend, decay_rate=0.2)   # fast decay
adstock_high = adstock(spend, decay_rate=0.7)   # slow decay, long carryover

# ---------------------------------------------------------------------
# 2. Saturation: diminishing returns via Hill transform
#    response = x^alpha / (x^alpha + gamma^alpha)
#    alpha controls curve steepness, gamma is the "half-saturation" point
#    (the spend level at which you get 50% of max response)
# ---------------------------------------------------------------------
def saturate(x, alpha, gamma):
    return x**alpha / (x**alpha + gamma**alpha)

spend_range = np.linspace(0, 300, 200)
sat_curve = saturate(spend_range, alpha=2.0, gamma=100)

# ---------------------------------------------------------------------
# Plot both concepts side by side
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(weeks, spend, color="lightgray", label="Raw weekly spend")
axes[0].plot(weeks, adstock_low, color="tab:blue", marker="o", label="Adstocked (decay=0.2, fast fade)")
axes[0].plot(weeks, adstock_high, color="tab:red", marker="o", label="Adstocked (decay=0.7, long carryover)")
axes[0].set_title("Adstock: Effect Persists After Spend Stops")
axes[0].set_xlabel("Week")
axes[0].set_ylabel("Spend / Adstocked value")
axes[0].legend()

axes[1].plot(spend_range, sat_curve, color="tab:green", linewidth=2)
axes[1].axvline(100, color="gray", linestyle="--", alpha=0.6)
axes[1].text(105, 0.1, "gamma = 100\n(half-saturation point)", fontsize=9)
axes[1].set_title("Saturation: Diminishing Returns on Spend")
axes[1].set_xlabel("Spend ($000s)")
axes[1].set_ylabel("Response (0-1 scale)")

plt.tight_layout()
plt.savefig("outputs/figures/02_adstock_saturation_demo.png", dpi=150)
print("Saved: outputs/figures/02_adstock_saturation_demo.png")

# ---------------------------------------------------------------------
# Key numbers for the writeup
# ---------------------------------------------------------------------
print("\n--- Adstock: spend stops at week 8, but effect lingers ---")
for w in [8, 9, 10, 12, 15]:
    print(f"Week {w}: raw spend={spend[w]:.0f}, "
          f"adstock(decay=0.2)={adstock_low[w]:.1f}, "
          f"adstock(decay=0.7)={adstock_high[w]:.1f}")

print("\n--- Saturation: doubling spend does NOT double response ---")
for s in [50, 100, 200, 300]:
    r = saturate(np.array([s]), alpha=2.0, gamma=100)[0]
    print(f"Spend={s}: response={r:.3f}")
