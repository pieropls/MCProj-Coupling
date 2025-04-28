import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import multivariate_normal, uniform
from IPython.display import display, Markdown

# ----------------------- Core functions -----------------------

def is_spd(matrix):
    return np.allclose(matrix, matrix.T) and np.all(np.linalg.eigvalsh(matrix) > 0)

def compute_sigma_opt(sigma_p, sigma_q):
    if not is_spd(sigma_p) or not is_spd(sigma_q):
        raise ValueError("Input matrices must be SPD.")

    C = np.linalg.cholesky(sigma_q)
    sigma_p_inv = np.linalg.inv(sigma_p)
    Q = C.T @ sigma_p_inv @ C
    D_vals, V = np.linalg.eigh(Q)
    U_vals = 1 / np.minimum(1, D_vals)
    U = np.diag(U_vals)
    sigma_opt = C @ V @ U @ V.T @ C.T
    return sigma_opt

def domination_constants(sigma_hat, sigma_p, sigma_q):
    factor = 2 * np.pi
    det_hat = np.linalg.det(factor * sigma_hat)
    det_p = np.linalg.det(factor * sigma_p)
    det_q = np.linalg.det(factor * sigma_q)
    M_p = np.sqrt(det_hat) / np.sqrt(det_p)
    M_q = np.sqrt(det_hat) / np.sqrt(det_q)
    return M_p, M_q

def reflection_maximal_coupling(a, b, Sigma):
    C = np.linalg.cholesky(Sigma)
    Sigma_sqrt_inv = np.linalg.inv(C)
    z = Sigma_sqrt_inv @ (a - b)
    norm_z = np.linalg.norm(z)
    e = z / norm_z

    X_dot = np.random.normal(0, 1, size=a.shape)
    U = np.random.uniform()
    std_normal = multivariate_normal(mean=np.zeros_like(a), cov=np.eye(len(a)))

    if std_normal.pdf(X_dot) * U < std_normal.pdf(X_dot + z):
        Y_dot = X_dot + z
    else:
        Y_dot = X_dot - 2 * np.dot(X_dot, e) * e

    X = a + C @ X_dot
    Y = b + C @ Y_dot
    return X, Y

def rejection_coupling(gamma_hat, p, q):
    A_x, A_y = 0, 0
    while A_x == 0 and A_y == 0:
        X1, Y1 = gamma_hat()
        U = uniform.rvs()
        if U < p['pdf'](X1) / (p['M'] * p['proposal_pdf'](X1)):
            A_x = 1
        if U < q['pdf'](Y1) / (q['M'] * q['proposal_pdf'](Y1)):
            A_y = 1
    X2 = p['sample']()
    Y2 = q['sample']()
    X = A_x * X1 + (1 - A_x) * X2
    Y = A_y * Y1 + (1 - A_y) * Y2
    return X, Y

def thorisson_coupling(p, q, C):
    X = p['sample']()
    U = np.random.uniform(0, 1)
    if U < min(q['pdf'](X) / p['pdf'](X), C):
        Y = X
    else:
        A = 0
        while A != 1:
            U = np.random.uniform(0, 1)
            Z = q['sample']()
            if U > min(1, C * p['pdf'](Z) / q['pdf'](Z)):
                A = 1
                Y = Z
    return X, Y

# ----------------------- Testing functions -----------------------

def matrix_to_latex(M):
    return "\\begin{bmatrix}" + " \\\\ ".join([
        " & ".join(f"{val:.2f}" for val in row) for row in M
    ]) + "\\end{bmatrix}"

def vector_to_latex(v):
    return "\\begin{bmatrix} " + " \\\\ ".join(f"{x:.2f}" for x in v) + " \\end{bmatrix}"

def run_coupling_test(mu_p, mu_q, sigma_p, sigma_q, method="rejection", N=10_000, display_output=True):
    p = {
        'pdf': lambda x: multivariate_normal.pdf(x, mean=mu_p, cov=sigma_p),
        'sample': lambda: multivariate_normal.rvs(mean=mu_p, cov=sigma_p)
    }
    q = {
        'pdf': lambda x: multivariate_normal.pdf(x, mean=mu_q, cov=sigma_q),
        'sample': lambda: multivariate_normal.rvs(mean=mu_q, cov=sigma_q)
    }

    if method == "rejection":
        sigma_hat = compute_sigma_opt(sigma_p, sigma_q)
        M_p, M_q = domination_constants(sigma_hat, sigma_p, sigma_q)
        def gamma_hat():
            return reflection_maximal_coupling(mu_p, mu_q, sigma_hat)
        def run_one():
            return rejection_coupling(gamma_hat, {
                **p,
                'proposal_pdf': lambda x: multivariate_normal.pdf(x, mean=mu_p, cov=sigma_hat),
                'M': M_p
            }, {
                **q,
                'proposal_pdf': lambda x: multivariate_normal.pdf(x, mean=mu_q, cov=sigma_hat),
                'M': M_q
            })

    elif method == "thorisson":
        factor = 2 * np.pi
        det_q = np.linalg.det(factor * sigma_q)
        det_p = np.linalg.det(factor * sigma_p)
        C = np.sqrt(det_q / det_p)
        def run_one():
            return thorisson_coupling(p, q, C)
    else:
        raise ValueError("Unknown method: choose 'rejection' or 'thorisson'")

    count_equal = 0
    for _ in range(N):
        X, Y = run_one()
        if np.allclose(X, Y, atol=1e-8):
            count_equal += 1

    rate = count_equal / N

    # ----------------------- LaTeX Output -----------------------

    mu_p_str = f"\\mu_p = {vector_to_latex(mu_p)}"
    mu_q_str = f"\\mu_q = {vector_to_latex(mu_q)}"
    sigma_p_str = matrix_to_latex(sigma_p)
    sigma_q_str = matrix_to_latex(sigma_q)

    markdown = rf"""
We test the **{method.capitalize()} coupling** on the following distributions:

$${mu_p_str}, \quad {mu_q_str}$$

$${{\Sigma}}_p = {sigma_p_str}, \quad {{\Sigma}}_q = {sigma_q_str}$$

**Coupling success rate**:
$$
\mathbb{{P}}(X = Y) \approx {rate:.4f}
$$
"""
    if display_output:
        display(Markdown(markdown))
    else:
        return rate

# ----------------------- Compare Coupling Methods -----------------------

def compare_precision(dim, N=1000, test=run_coupling_test, ax=None):
    """
    Compare success rates of Rejection Coupling and Thorisson Coupling methods across different values of ρ.
    
    Parameters:
    - dim: Dimension of the random vectors (e.g., 1, 2, 5, etc.).
    - N: Number of iterations for the coupling test (default is 1000).
    - test: Function to execute the coupling test (Rejection or Thorisson).
    - ax: Axes object to plot the graph (if None, it creates a new figure).
    """
    rho_values = np.arange(-1, 3.2, 0.1)
    success_rates_rej = []
    success_rates_tho = []

    mu_X = np.ones(dim)
    sigma_X = np.eye(dim) if dim > 1 else np.array([[1.0]])

    for rho in rho_values:
        mu_Y = rho * mu_X
        sigma_Y = (rho**2) * sigma_X

        rate_rej = test(mu_X, mu_Y, sigma_X, sigma_Y, method="rejection", N=N, display_output=False)
        rate_tho = test(mu_X, mu_Y, sigma_X, sigma_Y, method="thorisson", N=N, display_output=False)

        success_rates_rej.append(rate_rej)
        success_rates_tho.append(rate_tho)

    if ax is None:
        plt.figure(figsize=(9, 5))
    ax.plot(rho_values, success_rates_rej, marker='o', color='red', label='Rejection Coupling')
    ax.plot(rho_values, success_rates_tho, marker='*', color='green', label='Thorisson Coupling')
    ax.set_title(f"Coupling Success Rate vs. ρ (d = {dim})", weight='bold')
    ax.set_xlabel(r"$\rho$ values")
    ax.set_ylabel(r"$P(X = Y)$")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True)
    if ax is None:
        plt.tight_layout()
        plt.show()

# ----------------------- Compare Coupling Methods Time with Attempts -----------------------

def compare_runtime(n_range=2000, dims=None):
    if dims is None:
        dims = list(range(2, 11))  # {2, ..., 10}
    
    execution_times_rc = []
    execution_times_tc = []

    for d in dims:
        mu_p = np.zeros(d)
        mu_q = np.ones(d)
        sigma_p = np.eye(d)
        sigma_q = 1.5 * np.eye(d)

        runtimes_rc_d = []
        runtimes_tc_d = []

        def gamma_hat():
            return reflection_maximal_coupling(mu_p, mu_q, compute_sigma_opt(sigma_p, sigma_q))

        p = {
            'pdf': lambda x: multivariate_normal.pdf(x, mean=mu_p, cov=sigma_p),
            'sample': lambda: multivariate_normal.rvs(mean=mu_p, cov=sigma_p)
        }
        q = {
            'pdf': lambda x: multivariate_normal.pdf(x, mean=mu_q, cov=sigma_q),
            'sample': lambda: multivariate_normal.rvs(mean=mu_q, cov=sigma_q)
        }

        for _ in range(n_range):
            # Rejection Coupling
            X = np.zeros(d)
            Y = np.ones(d)
            t0 = time.perf_counter()
            while not np.allclose(X, Y, atol=1e-10):
                X, Y = rejection_coupling(gamma_hat, {
                    'pdf': lambda x: multivariate_normal.pdf(x, mean=mu_p, cov=sigma_p),
                    'sample': lambda: multivariate_normal.rvs(mean=mu_p, cov=sigma_p),
                    'M': 1.5,
                    'proposal_pdf': lambda x: multivariate_normal.pdf(x, mean=mu_p, cov=sigma_p)
                }, {
                    'pdf': lambda x: multivariate_normal.pdf(x, mean=mu_q, cov=sigma_q),
                    'sample': lambda: multivariate_normal.rvs(mean=mu_q, cov=sigma_q),
                    'M': 1.5,
                    'proposal_pdf': lambda x: multivariate_normal.pdf(x, mean=mu_q, cov=sigma_q)
                })
            t1 = time.perf_counter()
            runtimes_rc_d.append(t1 - t0)

            # Thorisson Coupling
            X = p['sample']()
            Y = q['sample']()
            t0 = time.perf_counter()
            while not np.allclose(X, Y, atol=1e-8):
                X, Y = thorisson_coupling(p, q, C=1)
            t1 = time.perf_counter()
            runtimes_tc_d.append(t1 - t0)

        execution_times_rc.append(runtimes_rc_d)
        execution_times_tc.append(runtimes_tc_d)

    # Compute means and IQRs
    means_rejection = [np.mean(t) for t in execution_times_rc]
    means_thorisson = [np.mean(t) for t in execution_times_tc]

    iqr_rejection = [(
        np.maximum(np.mean(t) - np.percentile(t, 25), 0),
        np.maximum(np.percentile(t, 75) - np.mean(t), 0)
    ) for t in execution_times_rc]

    iqr_thorisson = [(
        np.maximum(np.mean(t) - np.percentile(t, 25), 0),
        np.maximum(np.percentile(t, 75) - np.mean(t), 0)
    ) for t in execution_times_tc]

    low_rc, high_rc = zip(*iqr_rejection)
    low_tc, high_tc = zip(*iqr_thorisson)

    # Plot layout
    plt.figure(figsize=(18, 6))

    # First plot (both methods)
    plt.subplot(121)
    plt.errorbar(dims, means_rejection, yerr=[low_rc, high_rc], fmt='o-', capsize=5, label="Rejection Coupling", color='tab:blue')
    plt.errorbar(dims, means_thorisson, yerr=[low_tc, high_tc], fmt='s--', capsize=5, label="Thorisson Coupling", color='tab:red')
    plt.xlabel("Dimension")
    plt.ylabel("Average Runtime (s)")
    plt.title("Rejection and Thorisson Coupling", fontweight='bold')
    plt.grid(True)
    plt.legend()

    # Second plot (zoom on Rejection Coupling only)
    plt.subplot(222)
    plt.errorbar(dims, means_rejection, yerr=[low_rc, high_rc], fmt='o-', capsize=5, label="Rejection Coupling", color='tab:blue')
    plt.xlabel("Dimension")
    plt.ylabel("Average Runtime (s)")
    plt.title("Rejection Coupling", fontweight='bold')
    plt.grid(True)
    plt.legend()

    # Third plot (zoom on Thorisson Coupling only)
    plt.subplot(224)
    plt.errorbar(dims, means_thorisson, yerr=[low_tc, high_tc], fmt='s--', capsize=5, label="Thorisson Coupling", color='tab:red')
    plt.xlabel("Dimension")
    plt.ylabel("Average Runtime (s)")
    plt.title("Thorisson Coupling", fontweight='bold')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()