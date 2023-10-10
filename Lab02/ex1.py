import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import arviz as az

scale1 = 1/4
scale2 = 1/6 

probabilitateM1 = 0.4

#Generez 10000 valori random
values = []
for i in range(10000):
    if np.random.rand() < probabilitateM1:
        x = stats.expon.rvs(0, scale1, size = 1)
    else:
        x = stats.expon.rvs(0, scale2, size = 1)
    values.append(x)
    
#Calculez si afisez media si deviatia standard
media = np.mean(values)
deviatia = np.std(values)
print("Media lui X: ", media)
print("Deviatia standard a lui X: ", deviatia)

#Afisez graficul
plt.hist(np.array(values), bins = 100, density = True)
plt.xlabel('Ore')
plt.ylabel('Densitate')
plt.show()

