import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import numpy as np

if __name__ == '__main__':
#a
    df = pd.read_csv('Lab07\\auto-mpg.csv')

    df = df[df['horsepower'] != '?']

    plt.scatter(df['horsepower'], df['mpg'])
    plt.xlabel('Cai putere')
    plt.ylabel('Mile per galon (mpg)')
    plt.title('Relatia dintre cai putere si mpg')
    plt.show()

    #b
    with pm.Model() as model:

        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=10)
        sigma = pm.Uniform('sigma', lower=0, upper=10)
        mu = pm.Deterministic('mu', alpha + beta * np.array(df['horsepower'][1], dtype=np.int32))
        mpg = pm.Normal('mpg', mu=mu, sigma=sigma, observed=df['mpg'])

    #c
    with model:
        map_estimate = pm.find_MAP()
        alpha_map = map_estimate['alpha']
        beta_map = map_estimate['beta']

    with model:
        trace = pm.sample(5, tune=5)
    print(trace.keys())
    pm.plot_trace(trace)
    plt.show()
    pm.summary(trace).round(2)

    alpha_est = np.mean(trace['alpha'])
    beta_est = np.mean(trace['beta'])
    sigma_est = np.mean(trace['sigma'])

    x_values = np.linspace(df['horsepower'].min(), df['horsepower'].max(), 100)
    y_values = alpha_est + beta_est * x_values

    plt.scatter(df['horsepower'], df['mpg'])
    plt.plot(x_values, y_values, color='red', label=f'Regresie: y = {alpha_est:.2f} + {beta_est:.2f} * x')
    plt.xlabel('Cai putere')
    plt.ylabel('Mile per galon (mpg)')
    plt.title('Dreapta de regresie si date observate')
    plt.legend()
    plt.show()

    #d
    with model:
        pm.plot_posterior_predictive_glm(trace, samples=100, eval=np.linspace(df['horsepower'].min(), df['horsepower'].max(), 100), color='blue', alpha=0.1)

        plt.scatter(df['horsepower'], df['mpg'], alpha=0.5, label='Date observate')

        x_values = np.linspace(df['horsepower'].min(), df['horsepower'].max(), 100)
        y_values = alpha_est + beta_est * x_values
        plt.plot(x_values, y_values, color='red', label=f'Regresie: y = {alpha_est:.2f} + {beta_est:.2f} * x')

        plt.title('Regiunea 95%HDI pentru distributia predictiva a posteriori')
        plt.xlabel('Cai putere')
        plt.ylabel('Mile per galon (mpg)')
        plt.legend()
        plt.show()
        