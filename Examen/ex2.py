import numpy as np
from scipy import stats

def posterior_grid_geometric(grid_points=50, heads=1, tails=0):
    grid = np.linspace(0, 1, grid_points)
    
    #Inlocuim distributia binomiala cu cea geometrica
    prior = stats.geom.pmf(np.arange(1, grid_points + 1), 1 / grid)
    
    likelihood = stats.geom.pmf(heads, grid)
    
    posterior = likelihood * prior
    posterior /= posterior.sum()
    
    return grid, posterior

#10 heads 1 tail
data = np.repeat([0, 1], (10, 1))
points = 10
h = data.sum()
t = len(data) - h

grid, posterior = posterior_grid_geometric(points, h, t)

#Estimarea maxima a posteriori
map_estimate = grid[np.argmax(posterior)]

print(f"Maximum a posteriori estimate: {map_estimate}")
