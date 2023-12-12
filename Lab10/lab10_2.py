import pymc as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az
az.style.use('arviz-darkgrid')

#2
x_1 = np.linspace(-1, 1, 100)
y_1_large = 2 * x_1**5 - 4 * x_1**4 + 3 * x_1**3 + np.random.normal(0, 1, size=500)
order = 5
x_1p = np.vstack([x_1s**i for i in range(1, order+1)])
x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
y_1s = (y_1_large - y_1_large.mean()) / y_1_large.std()

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