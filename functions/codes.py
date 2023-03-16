import numpy as np

# compute solution of loss function sum_i (r_i - z_i beta)^2
def solution_cda(z, r):
    if sum(z * z) < 1e-10:
        return 0
    else:
        return sum(r * z) / sum(z * z)
    
# compute solutions of least squares problem via coordinate descent algorithm
def cda(x, y, n_iter = 1000, thres = 1e-08, verbose = False):
    n, p = x.shape
    beta = np.zeros(p)
    fitted_values = np.dot(x, beta)
    residuals = y - fitted_values
    rss_before = sum(residuals * residuals)
    for iter in range(n_iter):
        if verbose:
            print(iter, "th iteration runs")
        for j in range(p):
            z = x[:, j]
            r = residuals + beta[j] * z
            beta[j] = solution_cda(z, r)
            residuals = r - beta[j] * z
        rss = sum(residuals * residuals)
        if verbose:
            print("rss =", round(rss, 5), "\n")
        if rss_before - rss < thres:
            break
        else:
            rss_before = rss
    return beta

# generate P-spline basis given knots and degree
def Pspline_basis(x, knots, degree):
    n = len(x)
    J = len(knots) + degree + 1
    basis = np.zeros(n * J).reshape(n, J)
    # 다항식들
    for k in range(degree + 1):
        basis[:, k] = x ** k
    # 조각들
    for k in range(degree + 1, J):
        ind = (x - knots[k - degree - 1]) > 0
        basis[ind, k] = (x - knots[k - degree - 1])[ind] ** degree
    return basis