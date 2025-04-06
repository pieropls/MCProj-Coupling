### THIS FILE SHOULD BE DELETED BUT LET's DISCUSS IT FIRST ###

# It's the visualization of ensemble_rejection_coupling (useless as we still don't have the gamma)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def maximal_coupling_categorical(p, q):
    """
    Sample from the maximal coupling of two categorical distributions.

    Parameters:
    - p: probabilities of the first categorical distribution.
    - q: probabilities of the second categorical distribution.
    
    Returns:
    - i: index sampled from the first distribution.
    - j: index sampled from the second distribution.
    """
    assert np.all(p >= 0) and np.all(q >= 0)   # Ensure non-negativity
    assert np.isclose(np.sum(p), 1) and np.isclose(np.sum(q), 1) # Ensure valid probabilities
    
    N = len(p)
    min_pq = np.minimum(p, q)
    total_min = np.sum(min_pq)
    
    if np.random.rand() < total_min:
        i = np.random.choice(N, p=min_pq / total_min)   # Sample from the minimu
        return i, i
    else:
        I = np.random.choice(N, p=p)    # Sample from the first distribution
        J = np.random.choice(N, p=q)    # Sample from the second distribution
        return I, J
    
# Parameters 
mu_p, sigma_p = 0.0, 1.0
mu_q, sigma_q = 0.0, 2.0
hat_sigma = 2.5

# Densities
p = lambda x: norm(loc=mu_p, scale=sigma_p).pdf(x) # p ~ N(0, 1)
q = lambda y: norm(loc=mu_q, scale=sigma_q).pdf(y) # q ~ N(0, 4)
hat_p = lambda x: norm(loc=mu_p, scale=hat_sigma).pdf(x) # hat_p ~ N(0, 2.5)

# Constants
M_p = hat_sigma / sigma_p 
M_q = hat_sigma / sigma_q

# Samplers
sample_hat_gamma = lambda: (
    norm(loc=mu_p, scale=hat_sigma).rvs(),
    norm(loc=mu_q, scale=hat_sigma).rvs()
)
sample_pq = lambda: (
    norm(loc=mu_p, scale=sigma_p).rvs(),
    norm(loc=mu_q, scale=sigma_q).rvs()
)

# Run the experiment
n = 1000 # Number of samples
N_ensemble = 10 # Number of samples in the ensemble
epsilon = 1e-8 # tolerance for floating-point equality

results = [ensemble_rejection_coupling(sample_hat_gamma, p, q, hat_p, M_p, M_q, N_ensemble, sample_pq)
           for _ in range(n)] 

# Count approximate matches
matches = sum(abs(x - y) < epsilon for x, y in results)
percentage = (matches / n) * 100

print(f"Algorithms 2 : approximate matches (|X - Y| < {epsilon}): {matches}/{n} ({percentage:.2f}%)")

# Define X_vals and Y_vals from the results
X_vals, Y_vals = zip(*results)

# Re-run the plot with the now-defined X_vals and Y_vals
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
x_grid = np.linspace(-6, 6, 300)

# Scaled proposal curves
hat_p_scaled = lambda x: M_p * norm.pdf(x, mu_p, hat_sigma)
hat_q_scaled = lambda x: M_q * norm.pdf(x, mu_q, hat_sigma)

# Plot X
ax1.hist(X_vals, bins=30, alpha=0.6, color='blue', edgecolor='black', density=True, label="Empirical X")
ax1.plot(x_grid, norm.pdf(x_grid, mu_p, sigma_p), color='black', linewidth=2, label="Target p")
ax1.plot(x_grid, hat_p_scaled(x_grid), color='orange', linestyle='--', linewidth=2, label=r"$M_p \cdot \hat{p}(x)$")
ax1.set_title(r"Marginal distribution of $X \sim p = \mathcal{N}(0, 1)$", fontweight='bold')
ax1.set_xlabel("X values")
ax1.set_ylabel("Density")
ax1.legend()
ax1.grid(True)

# Plot Y
ax2.hist(Y_vals, bins=45, alpha=0.6, color='red', edgecolor='black', density=True, label="Empirical Y")
ax2.plot(x_grid, norm.pdf(x_grid, mu_q, sigma_q), color='black', linewidth=2, label="Target q")
ax2.plot(x_grid, hat_q_scaled(x_grid), color='orange', linestyle='--', linewidth=2, label=r"$M_q \cdot \hat{q}(x)$")
ax2.set_title(r"Marginal distribution of $Y \sim q = \mathcal{N}(0, 4)$", fontweight='bold')
ax2.set_xlabel("Y values")
ax2.set_ylabel("Density")
ax2.legend()
ax2.grid(True)

# Super title
plt.suptitle(
    rf"Rejection coupling marginals and dominating proposals" "\n"
    rf"with $\hat{{\sigma}} = {hat_sigma}$, $M_p = {M_p:.2f}$, $M_q = {M_q:.2f}$",
    fontsize=14,
    fontweight='bold'
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()