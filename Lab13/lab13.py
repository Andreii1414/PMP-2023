import arviz as az
import matplotlib.pyplot as plt
import pandas as pd

centered_data = az.load_arviz_data("centered_eight")
non_centered_data = az.load_arviz_data("non_centered_eight")

print("Centered Model:")
print("Number of chains:", centered_data.posterior.chain.shape[0])
print("Total sample size:", centered_data.posterior.draw.shape[0])

print("\nNon-Centered Model:")
print("Number of chains:", non_centered_data.posterior.chain.shape[0])
print("Total sample size:", non_centered_data.posterior.draw.shape[0])
print("\n")

az.plot_posterior(centered_data)
az.plot_posterior(non_centered_data)
plt.show()

summary_combined = pd.concat([az.summary(centered_data, var_names=["mu", "tau"]),
                              az.summary(non_centered_data, var_names=["mu", "tau"])])
print(summary_combined["r_hat"])
print("\n")

az.plot_autocorr(centered_data, var_names=["mu", "tau"])
az.plot_autocorr(non_centered_data, var_names=["mu", "tau"])
plt.show()

print(centered_data.sample_stats.diverging.sum())
print(non_centered_data.sample_stats.diverging.sum())

az.plot_pair(centered_data, var_names=["mu", "tau"], divergences=True)
az.plot_pair(non_centered_data, var_names=["mu", "tau"], divergences=True)
plt.show()
