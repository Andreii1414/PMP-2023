import pandas as pd
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

data = pd.read_csv('Lab08\\Prices.csv')

#1
with pm.Model() as model:

    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta1 = pm.Normal('beta1', mu=0, sigma=10)
    beta2 = pm.Normal('beta2', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=1)

    mu = alpha + beta1 * data['Speed'] + beta2 * np.log(data['HardDrive'])
    
    y = pm.Normal('y', mu=mu, sigma=sigma, observed=data['Price'])

with model:
    trace = pm.sample(1000, cores = 1, tune=1000)

#2
pm.plot_posterior(trace, var_names=['beta1', 'beta2'], hdi_prob=0.95)
plt.show()
#3
#Frecventa procesorului si marimea hard disk-ului sunt predictori utili pentru pretul de vanzare al unui laptop,
#beta1 si beta2 sunt pozitive, deci cu cat acestea cresc, cu atat creste si pretul de vanzare al unui laptop

#4
new_data = {'Speed': 33, 'HardDrive': np.log(540)}
with model:
    post_pred = pm.sample_posterior_predictive(trace, samples=1000, var_names=['y'], data=new_data)

az.hdi(post_pred['y'], hdi_prob=0.9)

#5
with model:
    post_pred_all = pm.sample_posterior_predictive(trace, samples=1000, var_names=['y'])

az.hdi(post_pred_all['y'], hdi_prob=0.9)

#Bonus
data['Premium'] =  data['Premium'].map({'yes': 1, 'no': 0})

with pm.Model() as model_premium:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta1 = pm.Normal('beta1', mu=0, sigma=10)
    beta2 = pm.Normal('beta2', mu=0, sigma=10)
    beta_premium = pm.Normal('beta_premium', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=1)

    mu = (
        alpha
        + beta1 * data['Speed']
        + beta2 * np.log(data['HardDrive'])
        + beta_premium * data['Premium']
    )

    y = pm.Normal('y', mu=mu, sigma=sigma, observed=data['Price'])

    trace_premium = pm.sample(1000, tune=1000, cores=1)

pm.plot_posterior(trace_premium, var_names=['beta_premium'])
plt.show()