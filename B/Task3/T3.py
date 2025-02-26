# This is the .py file for Task 3 of Section 7B

# Part a

import numpy as np
import matplotlib.pyplot as plt

def rejection_sampling(a, b, N, tf):
    t_vals = np.linspace(0, tf, 1000)
    p_vals = np.exp(-b * t_vals) * np.cos(a * t_vals)**2
    c = max(p_vals)
    samples = []
    count = 0

    while len(samples) < N:
        t_proposed = np.random.uniform(0, tf)
        p_t_proposed = np.exp(-b * t_proposed) * np.cos(a * t_proposed)**2
        q_t_proposed = 1 / tf
        acceptance_prob = p_t_proposed / (c * q_t_proposed)

        if np.random.rand() < acceptance_prob:
            samples.append(t_proposed)
        count += 1
    rejection_ratio = N / count

    plt.hist(samples, bins=30, density=True, alpha=0.6, color='blue', label='Sampled Distribution')
    plt.plot(t_vals, p_vals / (c * tf), label='Scaled True Distribution', color='red')
    plt.legend()
    plt.title(f'Rejection Sampling Histogram with N={N}')
    plt.xlabel('t')
    plt.ylabel('Density')
    plt.show()
    plt.savefig('T3_part_a.png')

    return samples, rejection_ratio

a = 4
b = 4
N = 10000 
tf = 2 * np.pi / a 

samples, rejection_ratio = rejection_sampling(a, b, N, tf)
print("Rejection ratio:", rejection_ratio)

# Part b

def rejection_sampling_exp(a, b, lambda_proposal, N):
    t_vals = np.linspace(0, 5, 1000) 
    p_vals = np.exp(-b * t_vals) * np.cos(a * t_vals)**2
    q_vals = lambda_proposal * np.exp(-lambda_proposal * t_vals)
    ratios = p_vals / q_vals
    c = max(ratios)
    samples = []
    count = 0

    while len(samples) < N:
        t_proposed = np.random.exponential(1 / lambda_proposal)
        p_t_proposed = np.exp(-b * t_proposed) * np.cos(a * t_proposed)**2
        q_t_proposed = lambda_proposal * np.exp(-lambda_proposal * t_proposed)
        acceptance_prob = p_t_proposed / (c * q_t_proposed)

        if np.random.rand() < acceptance_prob:
            samples.append(t_proposed)
        count += 1

    rejection_ratio = N / count

    plt.hist(samples, bins=30, density=True, alpha=0.6, color='blue', label='Sampled Distribution')
    plt.plot(t_vals, p_vals / (c * np.sum(q_vals * np.diff(t_vals[0:2]))), label='Scaled True Distribution', color='red')
    plt.legend()
    plt.title(f'Rejection Sampling with Exponential Proposal N={N}')
    plt.xlabel('t')
    plt.ylabel('Density')
    plt.show()
    plt.savefig('T3_part_b.png')

    return samples, rejection_ratio

a = 4
b = 4
lambda_proposal = 2
N = 10000 

samples, rejection_ratio = rejection_sampling_exp(a, b, lambda_proposal, N)
print("Rejection ratio:", rejection_ratio)

