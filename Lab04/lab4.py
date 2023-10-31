import numpy as np
from scipy.stats import poisson, norm, expon

lambda_ = 20 
media = 2  
deviatia = 0.5  
max_time = 15 
target_prob = 0.95
alpha = 0

while True:
    num_simulations = 1
    total_times = []
    for _ in range(num_simulations):
        num_customers = poisson.rvs(lambda_, size=1)[0]
        order_times = norm.rvs(loc=media, scale=deviatia, size=num_customers)
        cook_times = expon.rvs(scale=alpha, size=num_customers)
        total_time = order_times + cook_times
        total_times.append(total_time)

    success_count = 0
    for i in total_times:
        for j in i:
              if j <= max_time:
                success_count += 1

    actual_prob = success_count / num_customers
    if actual_prob <= target_prob:
        break
    alpha += 0.01

print(f"Valoarea alpha maxima pentru probabilitatea de 95%: {alpha:.2f}")

timp_mediu_asteptare = (media + alpha) / 2
print(f"Timpul mediu de asteptare pentru un client: {timp_mediu_asteptare:.2f} minute")
