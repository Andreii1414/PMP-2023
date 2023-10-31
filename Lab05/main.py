import pandas as pd
import numpy as np
import pymc3 as pm

traffic_csv = pd.read_csv('trafic.csv')

with pm.Model() as model:
    lambda_ = pm.Exponential('lambda_', lam=1)

    switch = [7 * 60, 8 * 60, 16 * 60, 19 * 60]     #7:00, 8:00, 16:00, 19:00

    poisson_dists = []
    for i in range(len(switch) + 1):
        if i == 0:
            poisson_dists.append(
                pm.Poisson('poisson_{}'.format(i), mu=lambda_, observed=traffic_csv.iloc[:switch[i], 1]))
        elif i == len(switch):
            poisson_dists.append(
                pm.Poisson('poisson_{}'.format(i), mu=lambda_, observed=traffic_csv.iloc[switch[i - 1]:, 1]))
        else:
            poisson_dists.append(pm.Poisson('poisson_{}'.format(i), mu=lambda_,
                                            observed=traffic_csv.iloc[switch[i - 1]:switch[i], 1]))

    trace = pm.sample(1000, tune=1000)

intervals = []
lambdas = []
for i in range(len(switch) + 1):
    if i == 0:
        interval = (0, switch[i])
        lambda_val = np.mean(trace['lambda_'][:switch[i]])
    elif i == len(switch):
        interval = (switch[i - 1], len(traffic_csv))
        lambda_val = np.mean(trace['lambda_'][switch[i - 1]:])
    else:
        interval = (switch[i - 1], switch[i])
        lambda_val = np.mean(trace['lambda_'][switch[i - 1]:switch[i]])
    intervals.append(interval)
    lambdas.append(lambda_val)

print('Cele mai probabile intervale:')
for interval in intervals:
    print('{}-{}'.format(traffic_csv.iloc[interval[0], 0], traffic_csv.iloc[interval[1] - 1, 0]))
print('Cele mai probabile valori pentru lambda:')
for lambda_val in lambdas:
    print(lambda_val)
