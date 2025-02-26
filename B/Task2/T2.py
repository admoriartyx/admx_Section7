# This is the .py file for Task2 of Section 7B

# Part a is attached as a .pdf in this directory

# Part b

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, quadrature

def exact_area(beta, c):
    e = np.sqrt(1 - (beta**2 / c**2))
    return 2 * np.pi * beta**2 * (1 + (c / beta * e) * np.arcsin(e))

def integrand(theta, beta, c):
    return 2 * np.pi * beta * np.sin(theta) * np.sqrt(beta**2 * np.cos(theta)**2 + c**2 * np.sin(theta)**2)

def midpoint_rule(beta, c, n=100):
    theta = np.linspace(0, np.pi, n+1)
    midpoints = (theta[:-1] + theta[1:]) / 2
    dx = np.pi / n
    area = sum(integrand(mid, beta, c) * dx for mid in midpoints)
    return area

def gaussian_quadrature(beta, c):
    area, _ = quadrature(integrand, 0, np.pi, args=(beta, c))
    return area

betas = np.linspace(0.001, 2, 50)
cs = np.linspace(0.001, 2, 50)
errors_midpoint = np.zeros((50, 50))
errors_gaussian = np.zeros((50, 50))

for i, beta in enumerate(betas):
    for j, c in enumerate(cs):
        exact = exact_area(beta, c)
        approx_midpoint = midpoint_rule(beta, c)
        approx_gaussian = gaussian_quadrature(beta, c)
        errors_midpoint[i, j] = np.abs(exact - approx_midpoint) / exact
        errors_gaussian[i, j] = np.abs(exact - approx_gaussian) / exact

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(errors_midpoint, extent=(betas.min(), betas.max(), cs.min(), cs.max()), origin='lower')
plt.colorbar()
plt.title('Midpoint Rule Error')
plt.xlabel('Beta')
plt.ylabel('c')

plt.subplot(1, 2, 2)
plt.imshow(errors_gaussian, extent=(betas.min(), betas.max(), cs.min(), cs.max()), origin='lower')
plt.colorbar()
plt.title('Gaussian Quadrature Error')
plt.xlabel('Beta')
plt.ylabel('c')
plt.show()
plt.savefig('T2_part_b.png')

# According to the heatmaps, we can fully expect beta and c quantities to have an effect on the error
# in mapping the ellipsoid.

# Part c

def is_inside_ellipsoid(x, y, z, beta=1, c=1):
    return (x**2 + y**2) / beta**2 + z**2 / c**2 <= 1

def exact_area(beta, c):
    return 4 / 3 * np.pi * beta**2 * c

def monte_carlo_area(beta, c, N):
    x = np.random.uniform(-beta, beta, N)
    y = np.random.uniform(-beta, beta, N)
    z = np.random.uniform(-c, c, N)
    
    inside_count = np.sum(is_inside_ellipsoid(x, y, z, beta, c))
    volume = (2 * beta)**2 * (2 * c)
    
    return (inside_count / N) * volume

Ns = [10, 100, 1000, 10000, 100000]
errors = []
exact = exact_area(1, 1)

for N in Ns:
    estimated_area = monte_carlo_area(1, 1, N)
    error = np.abs(estimated_area - exact) / exact
    errors.append(error)

plt.figure(figsize=(10, 6))
plt.plot(Ns, errors, marker='o')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of Samples (N)')
plt.ylabel('Relative Error')
plt.title('Error in Monte Carlo Estimation of Ellipsoid Area')
plt.grid(True)
plt.show()
plt.savefig('T2_part_c.png')

# Part d

def is_inside_ellipsoid(x, y, z, beta=1, c=1):
    return (x**2 + y**2) / beta**2 + z**2 / c**2 <= 1

def importance_sampling_exponential(N):
    lam = 3
    x = np.random.exponential(1/lam, N)
    y = np.random.exponential(1/lam, N)
    z = np.random.exponential(1/lam, N)
    weights = (1/np.exp(-lam * x))**3 / (2*1)**3 
    inside = is_inside_ellipsoid(x, y, z, 1, 1)
    return np.mean(weights * inside) * 8 

def importance_sampling_sinusoidal(N):
    x = np.linspace(0, 1, 10000)
    pdf = np.sin(5 * np.pi * x)**2
    cdf = np.cumsum(pdf) / np.sum(pdf)
    samples = np.interp(np.random.rand(N), cdf, x)

    y = samples * 2 - 1
    z = samples * 2 - 1
    weights = (1 / np.sin(5 * np.pi * samples)**2)

    inside = is_inside_ellipsoid(samples, y, z, 1, 1)
    return np.mean(weights * inside) * 8 

def exact_area(beta, c):
    return 4 / 3 * np.pi * beta**2 * c

exact = exact_area(1, 1)
Ns = [10, 100, 1000, 10000, 100000]
errors_exp = []
errors_sin = []

for N in Ns:
    estimated_area_exp = importance_sampling_exponential(N)
    estimated_area_sin = importance_sampling_sinusoidal(N)
    errors_exp.append(np.abs(estimated_area_exp - exact) / exact)
    errors_sin.append(np.abs(estimated_area_sin - exact) / exact)

plt.figure(figsize=(10, 6))
plt.plot(Ns, errors_exp, label='Exponential', marker='o')
plt.plot(Ns, errors_sin, label='Sinusoidal', marker='o')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of Samples (N)')
plt.ylabel('Relative Error')
plt.title('Error in Monte Carlo Estimation Using Importance Sampling')
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('T2_part_d.png')

# The sinusoidal method of importance sampling seems to minimize error in contrast to exponential method.

# Part e

def box_muller(N):
    u1 = np.random.rand(N // 2)
    u2 = np.random.rand(N // 2)
    R = np.sqrt(-2 * np.log(u1))
    theta = 2 * np.pi * u2

    x = R * np.cos(theta)
    y = R * np.sin(theta)

    return np.concatenate((x, y))

Ns = [10, 100, 1000, 10000, 100000]

plt.figure(figsize=(15, 8))
for i, N in enumerate(Ns, 1):
    samples = box_muller(N)
    plt.subplot(2, 3, i)
    plt.hist(samples, bins=30, density=True, alpha=0.75, color='blue')
    plt.title(f'N={N}')
    plt.xlabel('Value')
    plt.ylabel('Density')

plt.tight_layout()
plt.show()
plt.savefig('T2_part_e.png')

# Part f

def box_muller(N, mu=0, sigma=1):
    u1 = np.random.rand(N // 2)
    u2 = np.random.rand(N // 2)
    R = np.sqrt(-2 * np.log(u1)) * sigma
    theta = 2 * np.pi * u2
    x = R * np.cos(theta) + mu
    y = R * np.sin(theta) + mu
    return np.concatenate((x, y))

def monte_carlo_integration(N, mu, sigma):
    samples = box_muller(N, mu, sigma)
    integrand_values = samples 
    integral_estimate = np.mean(integrand_values)
    return integral_estimate

mus = [-2, 0, 2]
sigmas = [0.5, 1, 2]
N = 10000
results = {}

for mu in mus:
    for sigma in sigmas:
        estimate = monte_carlo_integration(N, mu, sigma)
        if not np.isnan(estimate) and not np.isinf(estimate):
            results[(mu, sigma)] = estimate

fig, ax = plt.subplots()
for (mu, sigma), estimate in results.items():
    if estimate > 0:  # Ensure positive and valid size values
        ax.scatter(mu, sigma, s=100 * estimate, label=f'mu={mu}, sigma={sigma}, estimate={estimate:.2f}')
    else:
        print(f"Invalid estimate for mu={mu}, sigma={sigma}: {estimate}")

ax.set_xlabel('Mu')
ax.set_ylabel('Sigma')
ax.set_title('Monte Carlo Estimates with Different Mu and Sigma')
ax.legend()
plt.show()
plt.savefig('T2_part_f.png')





