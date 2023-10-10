import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

probabilitateS1 = 0.25
probabilitateS2 = 0.25
probabilitateS3 = 0.3
probabilitateS4 = 0.2

def total_time(var):
    return(
        probabilitateS1 * stats.gamma(4, scale = 1/3).pdf(var) + 
        probabilitateS2 * stats.gamma(4, scale = 1/2).pdf(var) +
        probabilitateS3 * stats.gamma(5, scale = 1/2).pdf(var) + 
        probabilitateS4 * stats.gamma(5, scale = 1/3).pdf(var))

values = np.linspace(0, 10, 1000) #Generez uniform 1000 valori in intervalul 0-10
vvalues = []
for i in values:
    vvalues.append(total_time(i)) 

#Afisez graficul distributiei
plt.plot(values, vvalues)
plt.xlabel('Timp')
plt.ylabel('Densitate')
plt.show()

#Calculez probabilitatea ca timpul necesar servirii unui client sa fie mai mare de 3 milisecunde
prob = 1 - total_time(3)
print(f"Probabilitatea ca timpul necesar servirii unui client sa fie mai mare de 3 milisecunde: {prob}")