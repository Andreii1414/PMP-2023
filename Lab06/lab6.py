import pymc as pm
import matplotlib.pyplot as plt
import arviz as az

Y_values = [0, 5, 10]
teta_values = [0.2, 0.5]
p_lambda = 10

with pm.Model() as model:
    n = pm.Poisson('n', mu=p_lambda)
    for Y in Y_values:
        for teta in teta_values:
            obs = pm.Binomial(f'obs_{Y}_{teta}', n=n, p=teta, observed=Y)
            trace = pm.sample(1000, tune=1000, cores=1)
            az.plot_posterior(trace)

plt.show()




