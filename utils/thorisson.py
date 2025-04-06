import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.lines import Line2D

def run_thorisson_coupling(n, thorisson_coupling, sample_p, pdf_p, sample_q, pdf_q, C):
    """Run Thorisson coupling for n samples using the provided algorithm."""
    results = [thorisson_coupling(sample_p, pdf_p, sample_q, pdf_q, C) for _ in range(n)]
    return zip(*results)

def analyze_matches(X, Y):
    """Return match indicators, number, and percentage of matches."""
    matches = [x == y for x, y in zip(X, Y)]
    match_count = sum(matches)
    match_percentage = (match_count / len(X)) * 100
    return matches, match_count, match_percentage

def plot_results(X, Y, matches, mu_p, sigma_p, mu_q, sigma_q, C):
    """Generate histogram and scatter plots of the coupled samples."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x_grid = np.linspace(-8, 8, 300)

    # Marginal histograms and theoretical densities
    ax1.hist(X, bins=30, alpha=0.6, color='blue', edgecolor='black', label=r"$X \sim p = \mathcal{N}(0, 1)$", density=True)
    ax1.hist(Y, bins=30, alpha=0.4, color='red', edgecolor='black', label=r"$Y \sim q = \mathcal{N}(0, 4)$", density=True)
    ax1.plot(x_grid, norm(loc=mu_p, scale=sigma_p).pdf(x_grid), color='blue', linestyle='-', linewidth=2)
    ax1.plot(x_grid, norm(loc=mu_q, scale=sigma_q).pdf(x_grid), color='red', linestyle='-', linewidth=2)
    ax1.set_title("Thorisson coupling: marginal distributions", fontweight='bold')
    ax1.set_xlabel("Values")
    ax1.set_ylabel("Density")
    ax1.legend()
    ax1.grid(True)

    # Scatter plot of X vs Y with color-coded matches
    colors = ['green' if m else 'red' for m in matches]
    ax2.scatter(X, Y, c=colors, alpha=0.6, s=10)
    ax2.set_title("Scatter plot of coupled samples", fontweight='bold')
    ax2.set_xlabel("$X$")
    ax2.set_ylabel("$Y$")
    ax2.grid(True)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Match (X = Y)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Mismatch (X ≠ Y)')
    ]
    ax2.legend(handles=legend_elements)

    # Main title
    plt.suptitle(
        rf"Thorisson coupling between $p = \mathcal{{N}}({int(mu_p)}, {int(sigma_p**2)})$ and $q = \mathcal{{N}}({mu_q}, {sigma_q**2})$ with $C = {C}$",
        fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()