# This is my .py file for Part A of section 7

# Task 1

# Part a

import json
import numpy as np
from scipy.stats import binom
from scipy.special import beta, gamma
import matplotlib.pyplot as plt

datasets = ['dataset_1.json', 'dataset_2.json', 'dataset_3.json']
N = 500

results = {}

def perform_inference(dataset_path):
    with open(dataset_path, 'r') as file:
        data = json.load(file)
    
    M = sum(data)
    likelihoods = [binom.pmf(M, N, p) for p in p_values]
    prior = 1 / N 
    unnormalized_posterior = [likelihood * prior for likelihood in likelihoods]
    normalization_constant = sum(unnormalized_posterior)
    posterior = [value / normalization_constant for value in unnormalized_posterior]
    expectation = sum(p * prob for p, prob in zip(p_values, posterior))
    variance = sum((p - expectation) ** 2 * prob for p, prob in zip(p_values, posterior))
    
    return M, expectation, variance, posterior

fig, axs = plt.subplots(3, 1, figsize=(10, 15))
fig.suptitle('Posterior Distributions for Each Dataset')
p_values = np.linspace(0, 1, 100)

for i, dataset in enumerate(datasets):
    M, expectation, variance, posterior = perform_inference(dataset)
    results[dataset] = {"Heads": M, "Expectation": expectation, "Variance": variance}
    
    axs[i].plot(p_values, posterior, label=f"Dataset {i+1} (M={M})")
    axs[i].set_title(f"Dataset {i+1} - Expectation: {expectation:.4f}, Variance: {variance:.4f}")
    axs[i].set_xlabel('Probability of Heads (p)')
    axs[i].set_ylabel('Posterior Probability')
    axs[i].legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
plt.savefig(f'T1_part_a.png')

for key, value in results.items():
    print(f"{key}: Heads={value['Heads']}, Expectation={value['Expectation']:.4f}, Variance={value['Variance']:.4f}")


# Part b

from scipy.special import factorial
from math import log, sqrt, pi

N_values = np.arange(1, 11)  # From 1 to 10
gamma_factorials = gamma(N_values + 1)
stirling_approx = N_values * np.log(N_values) - N_values + 0.5 * np.log(2 * pi * N_values)
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.scatter(N_values, gamma_factorials, color='blue', label='Gamma Function (Exact Factorial)')
plt.plot(N_values, np.exp(stirling_approx), 'r-', label='Stirling\'s Approximation')
plt.xlabel('N')
plt.ylabel('Value')
plt.title('Comparison of Factorial and Stirling\'s Approximation')
plt.legend()

plt.subplot(2, 1, 2)
errors = np.abs(gamma_factorials - np.exp(stirling_approx))
plt.plot(N_values, errors, 'g-', label='Error between Gamma and Stirling\'s')
plt.xlabel('N')
plt.ylabel('Error')
plt.title('Error between Exact Factorials and Stirling\'s Approximation')
plt.legend()

plt.tight_layout()
plt.savefig("T1_part_b.png")
plt.show()

# Part c (actually labeled part e)

from sklearn.utils import resample

datasets2 = [json.load(open(path, 'r')) for path in datasets]
sample_sizes = [5, 15, 40, 60, 90, 150, 210, 300, 400]
num_bootstrap = 100

fig, axs = plt.subplots(3, 3, figsize=(15, 10))
fig.suptitle('Bootstrapping Histograms for Various Sample Sizes')

for i, dataset in enumerate(datasets2):
    for j, size in enumerate(sample_sizes):
        bootstrap_samples = [resample(dataset, n_samples=size) for _ in range(num_bootstrap)]
        heads_proportions = [np.mean(sample) for sample in bootstrap_samples]
        mean_heads = np.mean(heads_proportions)
        variance_heads = np.var(heads_proportions)
        ax = axs[i, j % 3]
        ax.hist(heads_proportions, bins=10, color='skyblue', edgecolor='black')
        ax.set_title(f'Dataset {i+1}, Sample {size}\nMean: {mean_heads:.3f}, Var: {variance_heads:.3f}')
        ax.set_xlabel('Proportion of Heads')
        ax.set_ylabel('Frequency')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("T1_part_c.png")
plt.show()






