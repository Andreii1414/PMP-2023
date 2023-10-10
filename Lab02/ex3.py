import numpy as np
import matplotlib.pyplot as plt

#Generez un nr random si in functie de probabilitate returnez s/b ( <0.3 => s; >=0.3 => b)
def arunca_moneda(prob):
    if(np.random.rand() < prob):
        return 's'
    else: return 'b'

#Sunt aruncate 2 monede de cate 10 ori, una cu probabilitate 0.3 pentru stema, una 0.5(nemasluita)
def arunca10():
    rez = {'ss' : 0, 'sb' : 0, 'bs' : 0, 'bb' : 0}
    for i in range(10):
        m1 = arunca_moneda(0.5)
        m2 = arunca_moneda(0.3)
        rez[m1 + m2] += 1
    return rez

rez = {'ss' : 0, 'sb' : 0, 'bs' : 0, 'bb' : 0}

#Se repeta de 100 de ori experimentul functiei "arunca10()"
for i in range(100):
    arunca = arunca10()
    for e in arunca:
        rez[e] += arunca[e]

#Generez un grafic cu distributia rezultatelor
etichete = ['ss', 'sb', 'bs', 'bb']
values = []
for e in etichete:
    values.append(rez[e])
    
plt.bar(etichete, values)
plt.xlabel('Rezultat')
plt.ylabel('Aparitii')
plt.show()
