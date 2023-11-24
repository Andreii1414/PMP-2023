from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import numpy as np

results = {"J": [], "N": [], "M": []}

for _ in range(10000):
    #se arunca moneda pentru a vedea cine incepe jocul
    J_start = np.random.choice([0, 1])

    #prima runda, in functie de jucatorul care incepe
    if J_start == 0:
        N_result = np.random.choice([0, 1], p=[1 / 2, 1 / 2]) 
    else:
        N_result = np.random.choice([0, 1], p=[1 / 3, 2 / 3])

    #runda a doua
    if J_start == 0:
        #daca a inceput 0, 1 arunca, daca 0 a obtinut 0 steme, se arunca o data, daca a obtinut o stema, se arunca de doua ori cu prob 2/3 pt stema
        if N_result == 0:
            M_result = np.random.choice([0, 1], p=[1 / 3, 2 / 3])
        else: 
            r1 = np.random.choice([0, 1], p=[1 / 3, 2 / 3])
            r2 = np.random.choice([0, 1], p=[1 / 3, 2 / 3])
            M_result = r1 + r2
    else:
        #daca a inceput 1, 0 arunca, daca 1 a obtinut 0 steme, se arunca o data, daca a obtinut o stema, se arunca de doua ori cu prob 1/2
        if N_result == 0:
            M_result = np.random.choice([0, 1], p=[1 / 2, 1 / 2])
        else: 
            r1 = np.random.choice([0, 1], p=[1 / 2, 1 / 2])
            r2 = np.random.choice([0, 1], p=[1 / 2, 1 / 2])
            M_result = r1 + r2

    results["J"].append(J_start)
    results["N"].append(N_result)
    results["M"].append(M_result)

wins_player_0 = 0
wins_player_1 = 0

for i in range(10000):
    #daca 0 a inceput, atunci cand n >= m => 0 castiga
    if results["J"][i] == 0:
        if results["N"][i] >= results["M"][i]:
            wins_player_0 += 1
        else:
            wins_player_1 += 1
    #daca 1 a inceput, atunci cand n >= m => 1 castiga
    else:
        if results["N"][i] >= results["M"][i]:
            wins_player_1 += 1
        else:
            wins_player_0 += 1

print("Jucatorul 0 are probabilitate de castig: ", wins_player_0 / 10000) # aprox 0.39
print("Jucatorul 1 are probabilitate de castig: ", wins_player_1 / 10000) # aprox 0.60



model = BayesianNetwork([('J', 'N'), ('J', 'M'), ('N', 'M')])

cpd_J = TabularCPD(variable='J', variable_card=2, values=[[0.5], [0.5]]) #prob ca fiecare jucator sa inceapa jocul
cpd_N = TabularCPD(variable='N', variable_card=2, values=[[1 / 2, 2 / 3], [1 / 2, 1 / 3]],
                   evidence=['J'], evidence_card=[2])  #prob pentru fiecare jucator in prima runda
cpd_M = TabularCPD(variable='M', variable_card=2, 
                   values=[[1 / 2, 1 / 2, 2 / 3, 2 / 3], 
                           [1 / 2, 1 / 2, 1 / 3, 1 / 3]],
                   evidence=['J', 'N'], evidence_card=[2, 2]) #prob pentru fiecare jucator in a doua runda

model.add_cpds(cpd_J, cpd_N, cpd_M)
infer = VariableElimination(model)

result = infer.query(variables=['J'], evidence={'M': 1})

print(result) #aprox 0.6 pentru jucatorul 0