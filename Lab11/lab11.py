import numpy as np
import arviz as az
import pymc as pm

#1
np.random.seed(42)
clusters = 3
n_cluster = [200, 150, 150]
n_total = sum(n_cluster)
means = [5, 0, -5]
std_devs = [2, 2, 2]
mix = np.concatenate([np.random.normal(loc=mu, scale=std, size=n) for mu, std, n in zip(means, std_devs, n_cluster)])

az.plot_kde(np.array(mix))

#2
for components in [2, 3, 4]:
    with pm.Model() as model:
        weights = pm.Dirichlet('weights', a=np.ones(components))
        means = pm.Normal('means', mu=np.arange(components) * 5, sigma=2, shape=components)
        sigma = pm.HalfNormal('sigma', sigma=2, shape=components)
        obs = pm.NormalMixture('obs', w=weights, mu=means, sigma=sigma, observed=mix)
        
        trace = pm.sample(20, cores = 1, tune=20, random_seed=42)
        az.plot_posterior(trace, var_names=['weights', 'means', 'sigma'], round_to=2)

waic_scores = []
loo_scores = []

#3
for components in [2, 3, 4]:
    with pm.Model() as model:
        weights = pm.Dirichlet('weights', a=np.ones(components))
        means = pm.Normal('means', mu=np.arange(components) * 5, sigma=2, shape=components)
        sigma = pm.HalfNormal('sigma', sigma=2, shape=components)
        obs = pm.NormalMixture('obs', w=weights, mu=means, sigma=sigma, observed=mix)
        
        trace = pm.sample(20, cores = 1, tune=20, random_seed=42)
        waic = az.waic(trace)
        loo = az.loo(trace)
        
        waic_scores.append(waic.waic)
        loo_scores.append(loo.loo)

print("WAIC Scores:", waic_scores)
print("LOO Scores:", loo_scores)

best_model_index = np.argmin(waic_scores)
print(f"Modelul cu {best_model_index + 2} componente este cel mai bun conform scorurilor WAIC și LOO.")
