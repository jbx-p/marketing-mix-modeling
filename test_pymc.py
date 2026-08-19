import pymc as pm
import arviz as az
print("PyMC version:", pm.__version__)

with pm.Model() as test_model:
    x = pm.Normal("x", mu=0, sigma=1)
    trace = pm.sample(200, tune=200, chains=1, progressbar=False)

print("Sampling worked. Summary:")
print(az.summary(trace))
