import pandas as pd
import pymc as pm
import arviz as az
import numpy as np

#incarcarea setului de date intr-un Pandas Dataframe folosind pandas.read_csv
df = pd.read_csv('Examen//BostonHousing.csv', usecols=['medv', 'rm', 'crim', 'indus'])

with pm.Model() as model:
    #Variabilele independente
    rm = pm.Data('rm', df.rm)
    crim = pm.Data('crim', df.crim)
    indus = pm.Data('indus', df.indus)

    #Variabila dependenta
    medv = pm.Data('medv', df.medv)

    #Coeficientii
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta_rm = pm.Normal('beta_rm', mu=0, sigma=10)
    beta_crim = pm.Normal('beta_crim', mu=0, sigma=10)
    beta_indus = pm.Normal('beta_indus', mu=0, sigma=10)

    #Modelul liniar
    mu = alpha + beta_rm * rm + beta_crim * crim + beta_indus * indus


    medv_obs = pm.Normal('medv_obs', mu=mu, sigma=1, observed=medv)

    trace = pm.sample(250, tune=250, cores=1)

#estimarea de 95% hdi pentru toti parametrii
az.plot_forest(trace,hdi_prob=0.95,var_names=['alpha', 'beta_rm', 'beta_crim', 'beta_indus'])
az.summary(trace,hdi_prob=0.95,var_names=['alpha', 'beta_rm', 'beta_crim', 'beta_indus'])
#Rezultatul este in poza 1c.PNG (am rulat pe google colab pentru ca dura mai putin), rm este cel care influenteaza cel mai mult
#rezultatul, adica numarul mediu de camere

#extrageri din distributia predictiva
with model:
    post_pred = pm.sample_posterior_predictive(trace)

#50% HDI pentru valoarea locuintelor
az.plot_forest(post_pred,hdi_prob=0.50,var_names=['medv_obs'])
az.summary(post_pred,hdi_prob=0.50,var_names=['medv_obs'])

