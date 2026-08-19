"""
Week 2: Fit a Bayesian Marketing Mix Model using pymc-marketing.
Uses geometric adstock + logistic saturation.
Includes control variables and holdout validation.

NOTE ON SCALING (real bug found during dev, documented here for the writeup):
pymc-marketing's MMM class internally scales the target variable (divides by
max(y)) before fitting, for numerical stability. mmm.predict() returns values
on that SCALED space, not original sales units. You must multiply by
mmm.get_scales_as_xarray()["target_scale"] to get predictions back in dollars.
Skipping this step silently produces predictions ~60,000x too small and a
catastrophic negative R^2 that looks like a modeling failure but is actually
a units bug.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

# ---------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------
df = pd.read_csv("data/mmm_weekly_data.csv", parse_dates=["date"])

channel_cols = [
    "tv_spend", "paid_search_spend", "paid_social_spend",
    "display_spend", "promotions_spend",
]
control_cols = [
    "price_index", "distribution",
    "is_black_friday", "is_christmas", "is_back_to_school", "disruption_flag",
]

# ---------------------------------------------------------------------
# 2. Train / holdout split (last 12 weeks held out)
# ---------------------------------------------------------------------
HOLDOUT_WEEKS = 12
train = df.iloc[:-HOLDOUT_WEEKS].reset_index(drop=True)
test = df.iloc[-HOLDOUT_WEEKS:].reset_index(drop=True)

print(f"Train: {train.shape[0]} weeks ({train['date'].min().date()} to {train['date'].max().date()})")
print(f"Test:  {test.shape[0]} weeks ({test['date'].min().date()} to {test['date'].max().date()})")

# ---------------------------------------------------------------------
# 3. Build the model
# ---------------------------------------------------------------------
mmm = MMM(
    date_column="date",
    channel_columns=channel_cols,
    control_columns=control_cols,
    target_column="sales",
    adstock=GeometricAdstock(l_max=8),
    saturation=LogisticSaturation(),
    yearly_seasonality=2,
)

X_train = train[["date"] + channel_cols + control_cols]
y_train = train["sales"]

print("\nFitting model (this samples from the posterior)...")
mmm.fit(
    X_train, y_train,
    target_accept=0.95,
    chains=4,
    cores=1,
    draws=1000,
    tune=1000,
    progressbar=True,
)
print("Fit complete.")

# ---------------------------------------------------------------------
# 4. Convergence diagnostics
# ---------------------------------------------------------------------
import arviz as az
summary = az.summary(mmm.idata, var_names=["saturation_lam", "saturation_beta", "adstock_alpha"])
print("\n--- Convergence check (r_hat should be ~1.00-1.01) ---")
print(summary[["mean", "sd", "r_hat"]])

max_rhat = summary["r_hat"].max()
if max_rhat > 1.05:
    print(f"\nWARNING: max r_hat = {max_rhat:.3f} -- model may not have converged. Consider more draws/tuning.")
else:
    print(f"\nConvergence looks good (max r_hat = {max_rhat:.3f})")

# ---------------------------------------------------------------------
# 5. Get target scale factor (CRITICAL - see module docstring)
# ---------------------------------------------------------------------
scales = mmm.get_scales_as_xarray()
target_scale = float(scales["target_scale"].values)
print(f"\nTarget scale factor: {target_scale:.1f} (multiply raw predict() output by this)")

# ---------------------------------------------------------------------
# 6. In-sample fit vs actual (rescaled to original units)
# ---------------------------------------------------------------------
y_pred_train_raw = mmm.predict(X_train)
y_pred_train = y_pred_train_raw * target_scale

r2_train = 1 - np.sum((y_train.values - y_pred_train)**2) / np.sum((y_train.values - y_train.values.mean())**2)
print(f"\nIn-sample R^2: {r2_train:.3f}")
print(f"Naive baseline (predict mean) R^2: 0.000  <- model should beat this easily")

# ---------------------------------------------------------------------
# 7. Out-of-sample holdout validation
# ---------------------------------------------------------------------
X_test = test[["date"] + channel_cols + control_cols]
y_test = test["sales"]

y_pred_test_raw = mmm.predict(X_test)
y_pred_test = y_pred_test_raw * target_scale

r2_test = 1 - np.sum((y_test.values - y_pred_test)**2) / np.sum((y_test.values - y_test.values.mean())**2)
mape_test = np.mean(np.abs((y_test.values - y_pred_test) / y_test.values)) * 100

print(f"\nOut-of-sample (holdout) R^2: {r2_test:.3f}")
print(f"Out-of-sample MAPE: {mape_test:.1f}%")

# ---------------------------------------------------------------------
# 8. Plot: actual vs predicted, train + holdout
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(train["date"], y_train, color="black", label="Actual (train)")
ax.plot(train["date"], y_pred_train, color="tab:blue", alpha=0.7, label="Predicted (train)")
ax.plot(test["date"], y_test, color="black", linestyle="--", label="Actual (holdout)")
ax.plot(test["date"], y_pred_test, color="tab:red", alpha=0.7, label="Predicted (holdout)")
ax.axvline(train["date"].iloc[-1], color="gray", linestyle=":", label="Train/holdout split")
ax.set_title("MMM Fit: Actual vs Predicted Sales")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/figures/03_model_fit_holdout.png", dpi=150)
print("\nSaved: outputs/figures/03_model_fit_holdout.png")

# ---------------------------------------------------------------------
# 9. Save the fitted model + predictions for later steps
# ---------------------------------------------------------------------
mmm.idata.to_netcdf("outputs/mmm_fitted_model.nc")
print("Saved: outputs/mmm_fitted_model.nc")

results = pd.DataFrame({
    "date": pd.concat([train["date"], test["date"]]).values,
    "actual": pd.concat([y_train, y_test]).values,
    "predicted": np.concatenate([y_pred_train, y_pred_test]),
    "is_holdout": [0]*len(train) + [1]*len(test),
})
results.to_csv("outputs/model_predictions.csv", index=False)
print("Saved: outputs/model_predictions.csv")

# Save target_scale for reuse in later scripts (decomposition, ROI, optimization)
with open("outputs/target_scale.txt", "w") as f:
    f.write(str(target_scale))
print(f"Saved: outputs/target_scale.txt ({target_scale})")
