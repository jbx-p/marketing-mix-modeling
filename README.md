# Marketing Mix Modeling: Channel ROI & Budget Optimization

A Bayesian Marketing Mix Model (MMM) quantifying how TV, Paid Search, Paid Social,
Display, and Promotions each contribute to sales -- and identifying a specific budget
reallocation opportunity backed by response-curve analysis.

**[Read the business memo](business_memo.md)** for the full findings and recommendation.

## Why MMM, and why now

Privacy changes (iOS 14 App Tracking Transparency, third-party cookie deprecation) broke
digital multi-touch attribution's ability to track users across platforms. Marketing Mix
Modeling -- a statistical, non-tracking-dependent approach to measuring channel
effectiveness -- is seeing a resurgence as companies rebuild the measurement capability
attribution used to provide.

## Results

- **Holdout R^2 = 0.905-0.924**, MAPE = 2.5% on a 12-week out-of-sample test period
- Decomposed $6.6M in sales into **92.7% baseline demand / 7.3% marketing-driven**
- Identified that **Display's headline ROI (58x average) overstates its true room to
  scale** -- marginal ROI (32x) tells a more accurate story
- Found the **optimal reallocation point** (15% of TV budget -> Display, +1.16%
  incremental sales) by testing multiple shift sizes, showing the model correctly
  captures diminishing returns rather than just chasing the highest-ROI label

## Key concepts (and why they matter)

**Adstock** -- marketing effects don't happen only in the week money is spent; they decay
over subsequent weeks. A TV campaign that ends this week still influences purchases 4-6
weeks from now. Ignoring this (treating spend and sales as same-week only) understates
slow-decay channels like TV and overstates fast-decay ones. This project uses geometric
adstock (`GeometricAdstock`, up to 8 weeks of carryover) fit per channel.

**Saturation** -- the 10th dollar spent on a channel in a given week does not work as hard
as the 1st. Every channel has a point of diminishing returns. This project uses a
logistic (Hill-type) saturation curve per channel, letting the model learn each channel's
individual saturation point from the data.

Together, these two transformations are what separate MMM from a plain linear regression
of sales on spend -- and why simple attribution/correlation analysis on this project's raw
data (see `notebooks/01_eda.py`) produced misleading same-week correlations (TV: 0.10,
despite being one of the stronger true drivers of sales).

## Real bugs found during development

This project surfaced two genuine debugging challenges worth documenting:

1. **Output scaling bug.** `pymc-marketing`'s `MMM.predict()` returns values on an
   internally normalized scale (target divided by its max value for numerical
   stability), not original sales dollars. Naively comparing raw `predict()` output to
   actual sales produced a catastrophic negative R^2 that looked like total model
   failure. Fix: multiply by `mmm.get_scales_as_xarray()["target_scale"]`.

2. **Weak identification for low-variance channels.** With default priors, the model
   drastically overstated the contribution of channels with limited week-to-week spend
   variation (Display's estimated contribution was 22x its known true value in testing
   against synthetic ground truth) -- these channels are statistically difficult to
   separate from baseline trend. Fix: tightened the saturation `beta` prior using the
   business-reasoning principle that no single channel should plausibly explain an
   outsized share of total sales. This reduced the overstatement to ~4x while holdout
   accuracy stayed the same or slightly improved (R^2 0.905 -> 0.924) -- the fix cost
   nothing in predictive performance.

Both are documented in code comments in `notebooks/03_fit_mmm.py` and
`notebooks/04_decompose_roi_scenarios.py`.

## Methodology

1. **Data**: 156 weeks (3 years) of synthetic weekly data with realistic trend,
   yearly seasonality, holiday effects, a simulated disruption event (supply shock,
   weeks 60-65), and channel-specific spend patterns (TV in flighted bursts, Paid
   Social growing over time, Display roughly flat). Ground-truth channel effects were
   generated and saved separately for model validation.
2. **Model**: Bayesian MMM (`pymc-marketing`) with geometric adstock, logistic
   saturation, yearly Fourier seasonality, and control variables (price index,
   distribution/ACV, holiday flags, disruption flag).
3. **Validation**: 12-week out-of-sample holdout, convergence diagnostics (r-hat),
   and -- since this is synthetic data -- direct comparison against known ground truth
   to catch estimation bias that would be invisible with only fit-quality metrics.
4. **Decomposition**: posterior-based counterfactual contribution estimates
   (`compute_mean_contributions_over_time`) split sales into baseline vs. per-channel
   incremental contribution.
5. **Optimization**: average and marginal ROI computed per channel; multiple budget
   reallocation scenarios tested to find the point where reallocation gains turn
   negative (channel saturation).

## Tech stack

Python, PyMC, PyMC-Marketing, ArviZ, pandas, matplotlib

## Repo structure

## Reproducing this project

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src\generate_data.py
python notebooks\01_eda.py
python notebooks\02_adstock_saturation_demo.py
python notebooks\03_fit_mmm.py
python notebooks\04_decompose_roi_scenarios.py
python notebooks\05_marginal_roi.py
```

## Status

Complete: data generation, EDA, adstock/saturation modeling, validation, decomposition,
ROI analysis (average + marginal), budget reallocation scenarios, business memo.
