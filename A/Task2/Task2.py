# This is the .py file for Task 1 of Section 7A

# Part a

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the exponential decay function
def exponential_decay(x, lam):
    return (1 / lam) * np.exp(-x / lam)

def load_data(filepath):
    try:
        with open(filepath, 'r') as file:
            return json.load(file)
    except Exception as e:
        print(f"Failed to read {filepath}: {e}")
        return None

# Load the datasets safely
vacuum_data = load_data('Vacuum_decay_dataset.json')
cavity_data = load_data('Cavity_decay_dataset.json')

if vacuum_data is None or cavity_data is None:
    print("Error loading one or more datasets.")
else:
    # Convert data to numpy arrays for easier manipulation
    vacuum_data = np.array(vacuum_data)
    cavity_data = np.array(cavity_data)

    # Prepare a linear space for probability fitting assuming a normalized dataset
    vacuum_probability = np.linspace(0, 1, len(vacuum_data))
    cavity_probability = np.linspace(0, 1, len(cavity_data))

    # Curve fitting to find the decay constants
    params_vacuum, _ = curve_fit(exponential_decay, np.sort(vacuum_data), vacuum_probability)
    params_cavity, _ = curve_fit(exponential_decay, np.sort(cavity_data), cavity_probability)

    # Plotting the results
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    # Histogram of decay events
    ax[0].hist(vacuum_data, bins=50, alpha=0.7, label=f'Vacuum λ={params_vacuum[0]:.2f}')
    ax[0].hist(cavity_data, bins=50, alpha=0.7, label=f'Cavity λ={params_cavity[0]:.2f}')
    ax[0].set_xlabel('Decay Distance')
    ax[0].set_ylabel('Frequency')
    ax[0].set_title('Histogram of Decay Events')
    ax[0].legend()

    # Plotting the fitted curves
    x_values = np.linspace(0, max(np.max(vacuum_data), np.max(cavity_data)), 100)
    ax[1].plot(x_values, exponential_decay(x_values, *params_vacuum), label=f'Fitted Vacuum Decay λ={params_vacuum[0]:.2f}')
    ax[1].plot(x_values, exponential_decay(x_values, *params_cavity), label=f'Fitted Cavity Decay λ={params_cavity[0]:.2f}')
    ax[1].set_xlabel('Decay Distance')
    ax[1].set_ylabel('Probability Density')
    ax[1].set_title('Exponential Decay Fit')
    ax[1].legend()

    plt.tight_layout()
    plt.savefig('T2_part_a_decay_analysis.png')
    plt.show()

    print(f"Vacuum decay constant: λ = {params_vacuum[0]:.2f}")
    print(f"Cavity decay constant: λ = {params_cavity[0]:.2f}")

    # Lot's of issues with opening the .json files for Task 2, will come back later potentially

    

