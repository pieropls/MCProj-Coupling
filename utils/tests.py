import numpy as np
from scipy.stats import multivariate_normal, uniform
from IPython.display import display, Markdown

# ===================== CORE FUNCTIONS =====================

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

from scipy.stats import multivariate_normal, uniform
from IPython.display import display, Markdown
import numpy as np

# ===================== TESTING FUNCTION =====================

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

    # ------------------------ Output ------------------------

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
    else :
        return rate
