import numpy as np
import matplotlib.pyplot as plt

def plot_coupled_rwm(dimensions, sigma=0.5, n_iter=300, coupled_rwm=None):
    """
    This function runs the coupled random walk Metropolis (RWM) for different dimensions and plots the results.

    Parameters:
    - dimensions: List of dimensions (e.g., [1, 5, 10]) for the RWM.
    - sigma: Standard deviation for the initial Sigma matrix.
    - n_iter: Number of iterations for the RWM.
    - coupled_rwm: Function to execute the coupled random walk Metropolis algorithm.
    """
    np.random.seed(2025)

    fig, axes = plt.subplots(1, len(dimensions), figsize=(18, 5))

    for idx, d in enumerate(dimensions):
        Sigma = sigma**2 * np.eye(d)
        x0 = np.ones(d) * 5
        y0 = -np.ones(d) * 5
        x_chain, y_chain, meeting_time = coupled_rwm(x0, y0, Sigma, n_iter=n_iter)

        ax = axes[idx]

        if d == 1:
            ax.plot(np.arange(len(x_chain)), x_chain[:, 0], label='X chain', color='blue')
            ax.plot(np.arange(len(y_chain)), y_chain[:, 0], label='Y chain', color='orange')
            ax.scatter(0, x_chain[0, 0], c='blue', label='Start X')
            ax.scatter(0, y_chain[0, 0], c='red', label='Start Y')
            ax.set_xlabel('Iterations')
            ax.set_ylabel('Values')
        else:
            ax.plot(x_chain[:, 0], x_chain[:, 1], label='X chain', color='blue')
            ax.plot(y_chain[:, 0], y_chain[:, 1], label='Y chain', color='orange')
            ax.scatter(x_chain[0, 0], x_chain[0, 1], c='blue', label='Start X')
            ax.scatter(y_chain[0, 0], y_chain[0, 1], c='red', label='Start Y')
            ax.set_xlabel('Dim 1')
            ax.set_ylabel('Dim 2')

        ax.set_title(f"d = {d}, Meeting time = {meeting_time}", weight='bold')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()