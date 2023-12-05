import pymc as pm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('Lab09\\Admission.csv')

gre_scores = data['GRE'].values
gpa_scores = data['GPA'].values
admission_status = data['Admission'].values

#1
with pm.Model() as logistic_model:
    
    beta0 = pm.Normal('beta0', mu=0, sigma=10)
    beta1 = pm.Normal('beta1', mu=0, sigma=10)
    beta2 = pm.Normal('beta2', mu=0, sigma=10)

    pi = pm.math.invlogit(beta0 + beta1 * gre_scores + beta2 * gpa_scores)

    admission_likelihood = pm.Bernoulli('admission_likelihood', p=pi, observed=admission_status)


with logistic_model:
    step = pm.Metropolis()
    trace = pm.sample(100, tune=100,cores = 1, step=step)

#2
mean_beta_0 = np.mean(trace.posterior['beta0'])
mean_beta_1 = np.mean(trace.posterior['beta1'])
mean_beta_2 = np.mean(trace.posterior['beta2'])

def decision_boundary(gre, gpa):
    return 1 / (1 + np.exp(-(mean_beta_0 + mean_beta_1 * gre + mean_beta_2 * gpa)))

hdi_94 = pm.hdi(trace.posterior['beta0'], hdi_prob=0.94)

gre_values = np.linspace(data['GRE'].min(), data['GRE'].max(), 100)
gpa_values = np.linspace(data['GPA'].min(), data['GPA'].max(), 100)
decision_boundary_values = np.zeros((100, 100))

for i, gre in enumerate(gre_values):
    for j, gpa in enumerate(gpa_values):
        decision_boundary_values[i, j] = decision_boundary(gre, gpa)

plt.scatter(data['GRE'], data['GPA'], c=data['Admission'], cmap='coolwarm', alpha=0.7)
plt.contour(gre_values, gpa_values, decision_boundary_values, levels=[0.5], colors='black')
plt.fill_betweenx(y=[data['GPA'].min(), data['GPA'].max()], x1=hdi_94.beta0[0], x2=hdi_94.beta0[1], color='gray', alpha=0.3)
plt.show()

#3
new_student_data = {'GRE': 550, 'GPA': 3.5}
posterior_probs = 1 / (1 + np.exp(-(trace.posterior['beta0'] + trace.posterior['beta1'] * new_student_data['GRE'] + trace.posterior['beta2'] * new_student_data['GPA'])))
hdi_90_prob = pm.hdi(posterior_probs, hdi_prob=0.9)
print("Intervalul HDI pentru probabilitatea de admitere:", hdi_90_prob)

#4
new_student_data_2 = {'GRE': 500, 'GPA': 3.2}
posterior_probs_2 = 1 / (1 + np.exp(-(trace.posterior['beta0'] + trace.posterior['beta1'] * new_student_data_2['GRE'] + trace.posterior['beta2'] * new_student_data_2['GPA'])))
hdi_90_prob_2 = pm.hdi(posterior_probs_2, hdi_prob=0.9)
print("Intervalul HDI pentru probabilitatea de admitere: ", hdi_90_prob_2)
#Intervalul de probabilitate scade o data cu scaderea GRE si GPA