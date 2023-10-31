import numpy as np
import pymc3 as pm
import arviz as az

lmbda = 20
media_norm = 2
deviatie_std_norm = 0.5
alpha = 3
num_samples = 100

timp_mediu_asteptare = []
for _ in range(num_samples):
    num_clienti = np.random.poisson(lmbda)
    timp_plasare_plata = np.random.normal(media_norm, deviatie_std_norm, num_clienti)
    timp_gatit = np.random.exponential(alpha, num_clienti)
    timp_mediu_asteptare.append(np.mean(timp_plasare_plata + timp_gatit))

print(timp_mediu_asteptare)

with pm.Model() as model:
    alpha = pm.Uniform("alpha", 0.1, 10)
    wait_time = pm.Normal("wait_time", mu=alpha, sigma=0.5, observed=timp_mediu_asteptare)
    trace = pm.sample(2000, tune=1000, chains=4)

summary = az.summary(trace, round_to=2)
print(summary)

az.plot_posterior(trace, var_names=["alpha"], credible_interval=0.95, kind='kde')

import matplotlib.pyplot as plt
plt.show()
