import pymc as pm
import matplotlib.pyplot as plt
import random

#generare 100 timpi de asteptare random folosind mu = 5 si sigma = 2
timp_mediu = [random.normalvariate(mu=5, sigma=2) for _ in range(100)]

with pm.Model() as model:
    mu = pm.Normal('mu', mu=5, sigma=5) 
    sigma = pm.HalfNormal('sigma', sigma=2)
    timp_mediu_asteptare = pm.Normal('timp_mediu_asteptare', mu=mu, sigma=sigma, observed=timp_mediu)
    #distr normala pt mu, distr half-normal pt sigma, distrb normala pt timp_mediu_asteptare => curs 5

with model:
    trace = pm.sample(1000, cores = 1, tune=1000)

#reprezentarea grafica a distributiei lui mu
pm.plot_posterior(trace.posterior['mu'])
plt.show()
#mu = 5