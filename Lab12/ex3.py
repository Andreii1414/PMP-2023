import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def metropolis(data, iterations, proposal_std, prior_params):
    theta_values = np.zeros(iterations)
    theta_values[0] = np.random.uniform(0, 1)

    for i in range(1, iterations):
        theta_prime = np.random.normal(theta_values[i-1], proposal_std)
        alpha = min(1, func(theta_prime, data, prior_params) / func(theta_values[i-1], data, prior_params))
        if np.random.rand() < alpha:
            theta_values[i] = theta_prime
        else:
            theta_values[i] = theta_values[i-1]

    return theta_values

def func(theta, data, prior_params):
    likelihood = stats.binom.pmf(sum(data), len(data), theta)
    prior = stats.beta.pdf(theta, *prior_params)
    return likelihood * prior

plt.figure(figsize=(10, 8))
n_trials = [0, 1, 2, 3, 4, 8, 16, 32, 50, 150]
data = [0, 1, 1, 1, 1, 4, 6, 9, 13, 48]
theta_real = 0.35
beta_params = [(1, 1), (20, 20), (1, 4)]
dist = stats.beta
x = np.linspace(0, 1, 200)

for idx, N in enumerate(n_trials):
    if idx == 0:
        plt.subplot(4, 3, 2)
        plt.xlabel('θ')
    else:
        plt.subplot(4, 3, idx+3)
        plt.xticks([])
    
    y = data[idx]
    
    for (a_prior, b_prior) in beta_params:
        p_theta_given_y = dist.pdf(x, a_prior + y, b_prior + N - y)
        plt.fill_between(x, 0, p_theta_given_y, alpha=0.7)
        plt.axvline(theta_real, ymax=0.3, color='k')
        plt.plot(0, 0, label=f'{N:4d} aruncari\n{y:4d} steme', alpha=0)

        iterations = 10000
        proposal_std = 0.1
        prior_params_metropolis = (a_prior, b_prior)
        samples = metropolis(data[:idx+1], iterations, proposal_std, prior_params_metropolis)
        plt.hist(samples, density=True, bins=30, alpha=0.5, color='green')

    plt.xlim(0, 1)
    plt.ylim(0, 12)
    plt.legend()
    plt.yticks([])
    plt.tight_layout()

plt.show()
