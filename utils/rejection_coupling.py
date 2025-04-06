import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters
mu_p, sigma_p = 0.0, 1.0        # p ~ N(0, 1)
mu_q, sigma_q = 0.0, 2.0        # q ~ N(0, 4)
hat_sigma = 2.5

# Densities
p = lambda x: norm(loc=mu_p, scale=sigma_p).pdf(x)
q = lambda y: norm(loc=mu_q, scale=sigma_q).pdf(y)
hat_p = lambda x: norm(loc=mu_p, scale=hat_sigma).pdf(x)
hat_q = lambda y: norm(loc=mu_q, scale=hat_sigma).pdf(y)

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

# Generate samples
n = 1000
epsilon = 1e-1
np.random.seed(42)

# The rejection_coupling function is defined in the notebook and passed here

def run_rejection_coupling(rejection_coupling):
    results = [
        rejection_coupling(p, q, hat_p, hat_q, sample_hat_gamma, sample_pq, M_p, M_q)
        for _ in range(n)
    ]
    X_values, Y_values = zip(*results)
    matches = sum(abs(x - y) < epsilon for x, y in results)
    percentage = (matches / n) * 100

    print(f"Approximate matches (|X - Y| < {epsilon}): {matches}/{n} ({percentage:.2f}%)")
    return X_values, Y_values, matches, percentage

def plot_rejection_coupling(X_values, Y_values):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x_grid = np.linspace(-6, 6, 300)

    # Scaled proposal curves
    hat_p_scaled = lambda x: M_p * norm.pdf(x, mu_p, hat_sigma)
    hat_q_scaled = lambda x: M_q * norm.pdf(x, mu_q, hat_sigma)

    ax1.hist(X_values, bins=30, alpha=0.6, color='blue', edgecolor='black', density=True, label="Empirical X")
    ax1.plot(x_grid, norm.pdf(x_grid, mu_p, sigma_p), color='black', linewidth=2, label="Target p")
    ax1.plot(x_grid, hat_p_scaled(x_grid), color='orange', linestyle='--', linewidth=2, label=r"$M_p \cdot \hat{p}(x)$")
    ax1.set_title(r"Marginal distribution of $X \sim p = \mathcal{N}(0, 1)$", fontweight='bold')
    ax1.set_xlabel("X values")
    ax1.set_ylabel("Density")
    ax1.legend()
    ax1.grid(True)

    ax2.hist(Y_values, bins=45, alpha=0.6, color='red', edgecolor='black', density=True, label="Empirical Y")
    ax2.plot(x_grid, norm.pdf(x_grid, mu_q, sigma_q), color='black', linewidth=2, label="Target q")
    ax2.plot(x_grid, hat_q_scaled(x_grid), color='orange', linestyle='--', linewidth=2, label=r"$M_q \cdot \hat{q}(x)$")
    ax2.set_title(r"Marginal distribution of $Y \sim q = \mathcal{N}(0, 4)$", fontweight='bold')
    ax2.set_xlabel("Y values")
    ax2.set_ylabel("Density")
    ax2.legend()
    ax2.grid(True)

    plt.suptitle(
        rf"Rejection coupling marginals and dominating proposals" "\n"
        rf"with $\hat{{\sigma}} = {hat_sigma}$, $M_p = {M_p:.2f}$, $M_q = {M_q:.2f}$",
        fontsize=14,
        fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()