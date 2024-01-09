import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def posterior_grid(grid_points=50, heads=6, tails=9, prior_type='uniform'):

    grid = np.linspace(0, 1, grid_points)

    if prior_type == 'uniform':
        prior = np.repeat(1/grid_points, grid_points)  
    elif prior_type == 'step':
        prior = (grid <= 0.5).astype(int)  
    elif prior_type == 'absolute_difference':
        prior = abs(grid - 0.5) 

    likelihood = stats.binom.pmf(heads, heads + tails, grid)
    posterior = likelihood * prior
    posterior /= posterior.sum()

    return grid, posterior

data = np.repeat([0, 1], (10, 3))
points = 10
h = data.sum()
t = len(data) - h

grid_uniform, posterior_uniform = posterior_grid(points, h, t, prior_type='uniform')
plt.plot(grid_uniform, posterior_uniform, 'o-', label='Uniform Prior')

grid_step, posterior_step = posterior_grid(points, h, t, prior_type='step')
plt.plot(grid_step, posterior_step, 'o-', label='Step Function Prior')

grid_absolute, posterior_absolute = posterior_grid(points, h, t, prior_type='absolute_difference')
plt.plot(grid_absolute, posterior_absolute, 'o-', label='Absolute Difference Prior')

plt.title(f'heads = {h}, tails = {t}')
plt.yticks([])
plt.xlabel('θ')
plt.legend()
plt.show()
