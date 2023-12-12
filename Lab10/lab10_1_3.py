import pymc as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az
az.style.use('arviz-darkgrid')
dummy_data = np.loadtxt('C:\\Users\\andrei14\\Downloads\\dummy.csv')

#1
x_1 = dummy_data[:, 0]
y_1 = dummy_data[:, 1]
order = 5
x_1p = np.vstack([x_1s**i for i in range(1, order+1)])
x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
y_1s = (y_1 - y_1.mean()) / y_1.std()

#a
with pm.Model() as model_p:
    alpha = pm.Normal('alpha', mu=0, sigma=1)
    beta = pm.Normal('beta', mu=0, sigma=10, shape=order)
    epsilon = pm.HalfNormal('epsilon', 5)
    mu = alpha + pm.math.dot(beta, x_1s)
    y_pred = pm.Normal('y_pred', mu=mu, sigma=epsilon, observed=y_1s)
    idata_p = pm.sample(20,20, cores = 1, return_inferencedata=True)

pm.plot_posterior_predictive_glm(idata_p, var_names=['y_pred'], samples=100)
plt.scatter(x_1s[0], y_1s, alpha=0.5, color='red', label='Observed data')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Regression (Order=5)')
plt.legend()
plt.show()

#b
with pm.Model() as model_p_sd_100:
    alpha = pm.Normal('alpha', mu=0, sigma=1)
    beta = pm.Normal('beta', mu=0, sigma=100, shape=order)
    epsilon = pm.HalfNormal('epsilon', 5)
    mu = alpha + pm.math.dot(beta, x_1s)
    y_pred = pm.Normal('y_pred', mu=mu, sigma=epsilon, observed=y_1s)
    idata_p_sd_100 = pm.sample(20,20, cores = 1, return_inferencedata=True)

with pm.Model() as model_p_sd_array:
    alpha = pm.Normal('alpha', mu=0, sigma=1)
    beta = pm.Normal('beta', mu=0, sigma=np.array([10, 0.1, 0.1, 0.1, 0.1]), shape=order)
    epsilon = pm.HalfNormal('epsilon', 5)
    mu = alpha + pm.math.dot(beta, x_1s)
    y_pred = pm.Normal('y_pred', mu=mu, sigma=epsilon, observed=y_1s)
    idata_p_sd_array = pm.sample(20,20, cores = 1, return_inferencedata=True)


pm.plot_posterior_predictive_glm(idata_p_sd_100, var_names=['y_pred'], samples=100, color='blue', label='SD=100')
pm.plot_posterior_predictive_glm(idata_p_sd_array, var_names=['y_pred'], samples=100, color='green', label='SD Array')
plt.scatter(x_1s[0], y_1s, alpha=0.5, color='red', label='Observed data')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Regression with Different Beta Distributions')
plt.legend()
plt.show()

#3
with pm.Model() as model_l:
    α = pm.Normal('α', mu=0, sigma=1)
    β = pm.Normal('β', mu=0, sigma=10)
    ε = pm.HalfNormal('ε', 5)
    μ = α + β * x_1s[0]
    y_pred = pm.Normal('y_pred', mu=μ, sigma=ε, observed=y_1s)
    idata_l = pm.sample(20, 20, cores = 1, return_inferencedata=True)

order_cubic = 3
x_1p_cubic = np.vstack([x_1s**i for i in range(1, order_cubic+1)])
x_1s_cubic = (x_1p_cubic - x_1p_cubic.mean(axis=1, keepdims=True)) / x_1p_cubic.std(axis=1, keepdims=True)

with pm.Model() as model_cubic:
    α = pm.Normal('α', mu=0, sigma=1)
    β = pm.Normal('β', mu=0, sigma=10, shape=order_cubic)
    ε = pm.HalfNormal('ε', 5)
    μ = α + pm.math.dot(β, x_1s_cubic)
    y_pred_cubic = pm.Normal('y_pred_cubic', mu=μ, sigma=ε, observed=y_1s)
    idata_cubic = pm.sample(20, 20, cores = 1, return_inferencedata=True)

waic_linear = pm.waic(idata_l).waic
waic_quadratic = pm.waic(idata_p).waic
waic_cubic = pm.waic(idata_cubic).waic

loo_linear = pm.loo(idata_l).loo
loo_quadratic = pm.loo(idata_p).loo
loo_cubic = pm.loo(idata_cubic).loo

print("WAIC - Linear: {:.2f}, Quadratic: {:.2f}, Cubic: {:.2f}".format(waic_linear, waic_quadratic, waic_cubic))
print("LOO - Linear: {:.2f}, Quadratic: {:.2f}, Cubic: {:.2f}".format(loo_linear, loo_quadratic, loo_cubic))

pm.compare({model_l: idata_l, model_p: idata_p, model_cubic: idata_cubic})
plt.show()


