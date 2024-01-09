import numpy as np
import matplotlib.pyplot as plt

def estimate_pi_and_error(N):
    x, y = np.random.uniform(-1, 1, size=(2, N))
    inside = (x**2 + y**2) <= 1
    pi = inside.sum()*4/N
    error = abs((pi - np.pi) / pi) * 100
    return error

Ns = [100, 1000, 10000]

errors = []

for N in Ns:
    current_errors = [estimate_pi_and_error(N) for _ in range(100)] 
    errors.append(current_errors)

mean_errors = np.mean(errors, axis=1)
std_errors = np.std(errors, axis=1)

plt.errorbar(Ns, mean_errors, yerr=std_errors, fmt='o-', capsize=5)
plt.xscale('log')  
plt.xlabel('Number of points (N)')
plt.ylabel('Error (%)')
plt.title('Estimation of PI with Error Bars')
plt.show()
