import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import cvxpy as cp
from numpy import matlib as mb


def test_phase_reg(set_up, coef, color, name="Regression"):
    plt.plot(
        np.arange(-4, 5, 0.1),
        np.arange(-4, 5, 0.1) * coef[0] + coef[1],
        color=color,
        linewidth=4,
        label=name,
    )
    # Error in the testing phase
    y_est = set_up["Xtest"][:, 0] * coef[0] + coef[1]
    error = 1 / set_up["Niter_test"] * np.linalg.norm(set_up["ytest"][:, 0] - y_est)
    print(error)
    return error


def test_phase_class(set_up, color, coef, ax, name="Classification"):
    ax.plot(
        np.arange(-4, 4, 0.1),
        (-coef[2] - np.arange(-4, 4, 0.1) * coef[0]) / coef[1],
        color=color,
        linewidth=4,
        label=name,
    )
    error = (
        np.sum(
            set_up["ytest"][:, 0]
            != np.sign(set_up["Xtest"][:, 0 : set_up["d"] + 1] @ coef)
        )
        / set_up["Niter_test"]
    )
    return error


def solver_cvx(set_up, objective_fn):
    n_rows = set_up["d"] + 1

    X_train = set_up["Xtrain"][:, : set_up["d"] + 1]
    Y_train = set_up["ytrain"][:, 0]

    w = cp.Variable(n_rows)
    lambd = cp.Parameter(nonneg=True)
    lambd.value = set_up["Lambda"]
    problem = cp.Problem(
        cp.Minimize(objective_fn(set_up["Niter_train"], X_train, Y_train, w, lambd))
    )
    problem.solve()
    return w.value


def eval_loss(input, set_up, loss):
    out = np.zeros((1, set_up["Niter_train"]))
    for k in range(set_up["Niter_train"]):
        out[0, k] = loss(
            set_up["Niter_train"],
            set_up["Xtrain"][:, 0 : set_up["d"] + 1],
            set_up["ytrain"][:, 0],
            input[:, k],
            set_up["Lambda"],
        ).value
    return out


def grad_FOM(set_up, grad):
    out = np.zeros((set_up["d"] + 1, set_up["Niter_train"]))
    out[:, 0] = set_up["Initial"]
    for k in range(0, set_up["Niter_train"] - 1):
        out[:, k + 1] = out[:, k] - set_up["mu_grad"] * grad(
            set_up["Niter_train"],
            set_up["Xtrain"][:, 0 : set_up["d"] + 1],
            set_up["ytrain"][:, 0],
            out[:, k],
            set_up["Lambda"],
        )
    return out


def grad_prox(set_up, grad, a):
    out = np.zeros((set_up["d"] + 1, set_up["Niter_train"]))
    out[:, 0] = set_up["Initial"]
    for k in range(0, set_up["Niter_train"] - 1):
        out[:, k + 1] = out[:, k] - set_up["mu_grad"] * grad(
            set_up["Niter_train"],
            set_up["Xtrain"],
            set_up["ytrain"],
            out[:, k],
            set_up["ro"],
            a,
        )
    last_out = out[:, set_up["Niter_train"] - 1]
    return last_out


def grad_FOM_decay(set_up, grad):
    out = np.zeros((set_up["d"] + 1, set_up["Niter_train"]))
    out[:, 0] = set_up["Initial"]
    for k in range(0, set_up["Niter_train"] - 1):
        out[:, k + 1] = out[:, k] - set_up["mu_grad"] * (1.5 / np.sqrt(k + 1)) * grad(
            set_up["Niter_train"],
            set_up["Xtrain"][:, 0 : set_up["d"] + 1],
            set_up["ytrain"][:, 0],
            out[:, k],
            set_up["Lambda"],
        )
    return out


def grad_Adam(set_up, grad):
    beta1 = 0.9
    beta2 = 0.999
    v = np.zeros((set_up["d"] + 1))
    g = np.zeros((set_up["d"] + 1))
    out = np.zeros((set_up["d"] + 1, set_up["Niter_train"]))
    out[:, 0] = set_up["Initial"]
    for k in range(0, set_up["Niter_train"] - 1):
        gradient = grad(
            set_up["Niter_train"],
            set_up["Xtrain"][:, 0 : set_up["d"] + 1],
            set_up["ytrain"][:, 0],
            out[:, k],
            set_up["Lambda"],
        )
        v = beta1 * v + (1 - beta1) * gradient
        vn = v / (1 - np.power(beta1, k + 1))
        g = beta2 * g + (1 - beta2) * np.power(gradient, 2)
        gn = g / (1 - np.power(beta2, k + 1))
        out[:, k + 1] = out[:, k] - set_up["mu_grad"] * vn / (np.sqrt(gn) + 1e-8)
    return out


def grad_SOM(set_up, grad, hess):
    out = np.zeros((set_up["d"] + 1, set_up["Niter_train"]))
    out[:, 0] = set_up["Initial"]
    for k in range(0, set_up["Niter_train"] - 1):
        out[:, k + 1] = out[:, k] - set_up["mu_hess"] * np.linalg.inv(
            hess(
                set_up["Niter_train"],
                set_up["Xtrain"][:, 0 : set_up["d"] + 1],
                set_up["ytrain"][:, 0],
                out[:, k],
                set_up["Lambda"],
            )
        ) @ grad(
            set_up["Niter_train"],
            set_up["Xtrain"][:, 0 : set_up["d"] + 1],
            set_up["ytrain"][:, 0],
            out[:, k],
            set_up["Lambda"],
        )
    return out


def BFGS(set_up, grad):
    out = np.zeros((set_up["d"] + 1, set_up["Niter_train"]))
    out[:, 0] = set_up["Initial"]
    Gk_1 = np.eye(set_up["d"] + 1)
    epsilon = 1e-4
    for k in range(set_up["Niter_train"] - 1):
        g_k = grad(
            set_up["Niter_train"],
            set_up["Xtrain"][:, 0 : set_up["d"] + 1],
            set_up["ytrain"][:, 0],
            out[:, k],
            set_up["Lambda"],
        )
        out[:, k + 1] = out[:, k] - set_up["mu_hess"] * Gk_1 @ g_k
        g_k_plus1 = grad(
            set_up["Niter_train"],
            set_up["Xtrain"][:, 0 : set_up["d"] + 1],
            set_up["ytrain"][:, 0],
            out[:, k + 1],
            set_up["Lambda"],
        )
        sk = (out[:, k + 1] - out[:, k]).reshape(-1, 1)
        if np.linalg.norm(sk) > epsilon:
            yk = (g_k_plus1 - g_k).reshape(-1, 1)
            ro = 1 / (yk.T @ sk)
            Gk_1 = (np.eye(set_up["d"] + 1) - ro * sk @ yk.T) @ Gk_1 @ (
                np.eye(set_up["d"] + 1) - ro * yk @ sk.T
            ) + ro * sk @ sk.T
    return out


def grad_ConjGrad(set_up, grad, hess):
    out = np.zeros((set_up["d"] + 1, set_up["Niter_train"]))
    p = np.zeros((set_up["d"] + 1, set_up["Niter_train"]))
    out[:, 0] = set_up["Initial"]
    H = hess(
        set_up["Niter_train"],
        set_up["Xtrain"][:, 0 : set_up["d"] + 1],
        set_up["ytrain"][:, 0],
        out[:, 0],
        set_up["Lambda"],
    )
    p[:, 0] = grad(
        set_up["Niter_train"],
        set_up["Xtrain"][:, 0 : set_up["d"] + 1],
        set_up["ytrain"][:, 0],
        out[:, 0],
        set_up["Lambda"],
    )
    g = grad(
        set_up["Niter_train"],
        set_up["Xtrain"][:, 0 : set_up["d"] + 1],
        set_up["ytrain"][:, 0],
        out[:, 0],
        set_up["Lambda"],
    )
    beta = 0
    for k in range(0, set_up["Niter_train"] - 1):
        alpha = -g.T @ p[:, k] / ((p[:, k]).T @ H @ p[:, k])
        out[:, k + 1] = out[:, k] + alpha * p[:, k]
        g = grad(
            set_up["Niter_train"],
            set_up["Xtrain"][:, 0 : set_up["d"] + 1],
            set_up["ytrain"][:, 0],
            out[:, k + 1],
            set_up["Lambda"],
        )
        beta = -g.T @ H @ p[:, k] / ((p[:, k]).T @ H @ p[:, k])
        p[:, k + 1] = g + beta * p[:, k]
    return out


def grad_SteepestDes(set_up, grad, hess):
    out = np.zeros((set_up["d"] + 1, set_up["Niter_train"]))
    p = np.zeros((set_up["d"] + 1, set_up["Niter_train"]))
    out[:, 0] = set_up["Initial"]
    H = hess(
        set_up["Niter_train"],
        set_up["Xtrain"][:, 0 : set_up["d"] + 1],
        set_up["ytrain"][:, 0],
        out[:, 0],
        set_up["Lambda"],
    )
    for k in range(0, set_up["Niter_train"] - 1):
        p[:, k] = grad(
            set_up["Niter_train"],
            set_up["Xtrain"][:, 0 : set_up["d"] + 1],
            set_up["ytrain"][:, 0],
            out[:, k],
            set_up["Lambda"],
        )
        alpha = -(p[:, k]).T @ p[:, k] / ((p[:, k]).T @ H @ p[:, k])
        out[:, k + 1] = out[:, k] + alpha * p[:, k]
    return out


def grad_inst(set_up, grad, order):
    out = np.zeros((set_up["d"] + 1, set_up["Niter_train"]))
    out[:, 0] = set_up["Initial"]
    for k in range(set_up["Niter_train"] - 1):
        x = set_up["Xtrain"][
            k, order * (set_up["d"] + 1) : (order + 1) * (set_up["d"] + 1)
        ].reshape(1, -1)
        y = set_up["ytrain"][k, order].reshape(1, -1)
        out[:, k + 1] = np.squeeze(
            out[:, k].reshape(-1, 1)
            - set_up["mu_grad"]
            * grad(1, x, y, out[:, k].reshape(-1, 1), set_up["Lambda"])
        )
    return out


def grad_inst_decay(set_up, grad, order):
    out = np.zeros((set_up["d"] + 1, set_up["Niter_train"]))
    out[:, 0] = set_up["Initial"]
    for k in range(set_up["Niter_train"] - 1):
        x = set_up["Xtrain"][
            k, order * (set_up["d"] + 1) : (order + 1) * (set_up["d"] + 1)
        ].reshape(1, -1)
        y = set_up["ytrain"][k, order].reshape(1, -1)
        out[:, k + 1] = np.squeeze(
            out[:, k].reshape(-1, 1)
            - set_up["mu_grad"]
            * (1.5 / np.sqrt(k + 1))
            * grad(1, x, y, out[:, k].reshape(-1, 1), set_up["Lambda"])
        )
    return out


def calculation_Hessian_logistic(Niter_train, Xtrain, ytrain, w, Lambda):
    H = np.zeros((len(w), len(w)))
    aux = np.exp(np.diag(ytrain.flatten()) @ (Xtrain @ w)) / (
        (1 + np.exp(np.diag(ytrain.flatten()) @ (Xtrain @ w))) ** 2
    )
    for k1 in range(len(w)):
        for k2 in range(len(w)):
            H[k1, k2] = np.sum(Xtrain[:, k1] * Xtrain[:, k2] * aux)
    H = (1 / Niter_train) * H + Lambda * np.eye(len(w))
    return H


def bounds(Niter, L, Lmu, mu, start, start_s):
    # u_b_c=upper bound convex
    # l_b_c=lower bound convex
    # u_b_s_c=upper bound strongly convex
    # l_b_s_c=lower bound strongly convex
    # u_b_c_acc=upper bound convex accelerated
    # u_b_s_c_acc=upper bound strongly convex accelerated

    kappa = Lmu / mu
    k = np.arange(1, Niter + 1)
    u_b_c = 2 * L / (k + 4) * start
    l_b_c = ((3 * L) / (32 * (k + 1) ** 2)) * start
    u_b_s_c = ((Lmu / 2) * ((kappa - 1) / (kappa + 1)) ** (2 * k)) * start_s
    l_b_s_c = (
        (mu / 2) * ((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)) ** (2 * k)
    ) * start_s
    u_b_c_acc = (4 * L / ((k + 2) ** 2)) * start
    u_b_s_c_acc = (Lmu * (1 - np.sqrt(1 / kappa)) ** k) * start_s

    return u_b_c, l_b_c, u_b_c_acc, u_b_s_c, l_b_s_c, u_b_s_c_acc


def ridge_no_reg(Niter, wnr, X, y, eta, wopt):
    for k in range(Niter):
        wnr[:, k + 1] = wnr[:, k] - eta * X.T @ (X @ wnr[:, k] - y.flatten())

    f = 0.5 * np.sum((X @ wnr - np.kron(y, np.ones((1, Niter + 1)))) ** 2, axis=0)
    fopt = 0.5 * np.linalg.norm(X @ wopt - y) ** 2
    return f, fopt


def ridge_reg(Niter, w, X, y, eta, lambd, wopt):
    for k in range(Niter):
        w[:, k + 1] = (w[:, k] * (1 - lambd * eta)) - eta * X.T @ (
            X @ w[:, k] - y.flatten()
        )
    f = 0.5 * np.sum(
        (X @ w - np.kron(y, np.ones((1, Niter + 1)))) ** 2, axis=0
    ) + 0.5 * lambd * np.sum(w**2, axis=0)
    fopt = (
        0.5 * np.linalg.norm(X @ wopt - y) ** 2
        + 0.5 * lambd * np.linalg.norm(wopt) ** 2
    )
    return f, fopt


def ridge_reg_acc(Niter, waux, X, y, lambd, wopt, L, mu):
    w2 = waux.copy()
    gamma = (np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu))
    for k in range(Niter):
        w2[:, k + 1] = waux[:, k] * (1 - lambd / L) - (1 / L) * X.T @ (
            X @ waux[:, k] - y.flatten()
        )
        waux[:, k + 1] = w2[:, k + 1] + gamma * (w2[:, k + 1] - w2[:, k])

    f = 0.5 * np.sum(
        (X @ waux - np.kron(y, np.ones((1, Niter + 1)))) ** 2, axis=0
    ) + 0.5 * lambd * np.sum(waux**2, axis=0)
    fopt = (
        0.5 * np.linalg.norm(X @ wopt - y) ** 2
        + 0.5 * lambd * np.linalg.norm(wopt) ** 2
    )
    return f, fopt


def ridge_reg_gcd(Niter, w, X, y, lambd, wopt):  # gradient coordinate descent
    indx = np.arange(0, len(w))
    for k in range(Niter):
        wk = w[:, k]
        for j in range(len(w)):
            Xm = X[:, indx != j]  # Xm is X without the j-th column
            wm = wk[indx != j]  # wm is w without the j-th component
            wk[j] = (X[:, j].T @ (y.flatten() - Xm @ wm)) / (
                X[:, j].T @ X[:, j] + lambd
            )
        w[:, k + 1] = wk
    f = 0.5 * np.sum(
        (X @ w - np.kron(y, np.ones((1, Niter + 1)))) ** 2, axis=0
    ) + 0.5 * lambd * np.sum(w**2, axis=0)
    fopt = (
        0.5 * np.linalg.norm(X @ wopt - y) ** 2
        + 0.5 * lambd * np.linalg.norm(wopt) ** 2
    )
    return f, fopt


def calculation_subgrad_svm(Niter_train, Xtrain, ytrain, w, Lambd):
    g = np.zeros((len(w), 1))
    diag_y_X = np.diag(ytrain) @ Xtrain
    ind = diag_y_X @ w < 1
    g = (-1 / Niter_train) * np.sum(diag_y_X[ind, :], axis=0).T + Lambd * w
    return g


def calculation_subgrad_svm_inst(n, X, y, w, lamb):
    if y * X @ w < 1:
        grad_hinge = -y * X.T
    else:
        grad_hinge = 0
    return grad_hinge + lamb * w


def ista_lasso(set_up, grad):
    out = np.zeros((set_up["d"] + 1, set_up["Niter_train"]))
    out[:, 0] = set_up["Initial"]
    for k in range(set_up["Niter_train"] - 1):
        v = out[:, k] - set_up["mu_grad"] * grad(
            set_up["Niter_train"],
            set_up["Xtrain"][:, : set_up["d"] + 1],
            set_up["ytrain"][:, 0],
            out[:, k],
            set_up["Lambda"],
        )
        out[:, k + 1] = prox_normL1(v, set_up["Lambda"] / set_up["q"])
    return out


def fista_lasso(set_up, grad):
    out = np.zeros((set_up["d"] + 1, set_up["Niter_train"]))
    out[:, 0] = set_up["Initial"]
    aux = out
    for k in range(1, set_up["Niter_train"] - 1):
        aux[:, k] = out[:, k] + ((k - 2) / (k + 1)) * (out[:, k] - out[:, k - 1])
        v = aux[:, k] - set_up["mu_grad"] * grad(
            set_up["Niter_train"],
            set_up["Xtrain"][:, : set_up["d"] + 1],
            set_up["ytrain"][:, 0],
            aux[:, k],
            set_up["Lambda"],
        )
        out[:, k + 1] = prox_normL1(v, set_up["Lambda"] / set_up["q"])
    return out


def prox_normL1(v, lambd):
    return np.maximum(np.zeros(len(v)), v - lambd) - np.maximum(
        np.zeros(len(v)), -v - lambd
    )


def prox_quadratic(A, b, v, ro):
    # Proximal operator of the quadratic function
    # for x-z in the regularisation term
    return np.linalg.inv(A + ro * np.eye(len(A))) @ (ro * v - b)


def bcd_ridge(set_up):
    m = set_up["d"] + 1
    p = set_up["Number_blocks"]
    A = set_up["Xtrain"][:, :m]
    b = set_up["ytrain"][:, 0]

    # Parallel method (BCD)
    x_bcd = np.zeros((m, set_up["Number_iter_BCD"] + 1))
    for k in range(set_up["Number_iter_BCD"]):
        z = x_bcd[:, k]
        for i in range(int(m / p)):
            ajsum = A @ z - A[:, i * p : (i + 1) * p] @ z[i * p : (i + 1) * p]
            x_bcd[i * p : (i + 1) * p, k + 1] = (
                np.linalg.inv(
                    A[:, i * p : (i + 1) * p].T @ A[:, i * p : (i + 1) * p]
                    + (set_up["Lambda"] / 2) * set_up["Niter_train"] * np.eye(p)
                )
                @ A[:, i * p : (i + 1) * p].T
                @ (b - ajsum)
            )
    return x_bcd


def bcd_lasso(set_up):
    m = set_up["d"] + 1
    # p = 1  # set_up["Number_blocks"]
    A = set_up["Xtrain"][:, :m]
    b = set_up["ytrain"][:, 0]

    tot_iters_flexa = set_up["Number_iter_BCD"]
    lambd = set_up["Lambda"]
    n = set_up["Niter_train"]

    # Parallel method (CD):
    x_bcd = np.zeros((m, set_up["Number_iter_BCD"] + 1))
    for k in range(tot_iters_flexa):
        z = x_bcd[:, k]
        for i in range(m):
            ajsum = A @ z - A[:, i] * z[i]
            # ajsum = A @ z - A[:, i * p : (i + 1) * p] @ z[i * p : (i + 1) * p]
            d = A[:, i].T @ A[:, i]
            # d = np.linalg.inv(A[:, i * p : (i + 1) * p].T @ A[:, i * p : (i + 1) * p])
            s = A[:, i].T @ (b - ajsum) / d
            # s = d @ A[:, i * p : (i + 1) * p].T @ (b - ajsum)
            x_bcd[i, k + 1] = prox_normL1(s.reshape(1), n * lambd / (2 * d))[0]
            # x_bcd[i * p : (i + 1) * p, k + 1] = prox_normL1(
            #    s, np.diag(d) * n * lambd / 2
            # )
    return x_bcd


def admm_lasso(set_up):
    # Definition of variables
    x_admm = np.zeros((set_up["d"] + 1, set_up["Niter_train"]))
    z_admm = x_admm.copy()
    u_admm = x_admm.copy()
    A = set_up["Xtrain"][:, : set_up["d"] + 1]
    b = set_up["ytrain"][:, 0]
    n = set_up["Niter_train"]
    lambd = set_up["Lambda"]
    rho = set_up["ro"]

    for k in range(1, n - 1):
        x_admm[:, k + 1] = prox_quadratic(
            A.T @ A * 2 / n, -2 * A.T @ b / n, z_admm[:, k] - u_admm[:, k], rho
        )
        z_admm[:, k + 1] = prox_normL1(x_admm[:, k + 1] + u_admm[:, k], lambd / rho)
        u_admm[:, k + 1] = u_admm[:, k] + x_admm[:, k + 1] - z_admm[:, k + 1]
    return x_admm


def admm_logistic(set_up, grad):
    # Definition of variables
    x_admm = np.zeros((set_up["d"] + 1, set_up["Niter_train"]))
    z_admm = x_admm.copy()
    u_admm = x_admm.copy()
    n = set_up["Niter_train"]
    lambd = set_up["Lambda"]
    rho = set_up["ro"]

    for k in range(1, n - 1):
        x_admm[:, k + 1] = grad_prox(set_up, grad, z_admm[:, k] - u_admm[:, k])
        z_admm[:, k + 1] = prox_quadratic(
            np.eye(set_up["d"] + 1) * lambd,
            np.zeros(set_up["d"] + 1),
            x_admm[:, k + 1] + u_admm[:, k],
            rho,
        )
        u_admm[:, k + 1] = u_admm[:, k] + x_admm[:, k + 1] - z_admm[:, k + 1]
    return x_admm


def admm_lasso_dist(set_up):
    nb = set_up["Number_nodes"]
    n = set_up["Niter_train"]
    Naux = int(n / nb)
    x_admm_dist = np.zeros((set_up["d"] + 1, set_up["Niter_train"]))
    vv = np.zeros((set_up["d"] + 1, nb))
    xx = np.zeros((set_up["d"] + 1, nb))
    z_ave = np.zeros(set_up["d"] + 1)
    for k in range(1, Naux):
        for kk in range(nb):
            Xaux = set_up["Xtrain"][kk * Naux : (kk + 1) * Naux, : set_up["d"] + 1]
            yaux = set_up["ytrain"][kk * Naux : (kk + 1) * Naux, 0]
            xx[:, kk] = prox_quadratic(
                Xaux.T @ Xaux * 2 / n,
                -2 * Xaux.T @ yaux / n,
                z_ave - vv[:, kk],
                set_up["ro"],
            )
        x_admm_dist[:, k + 1] = np.sum(xx, axis=1) / nb
        v_hat = np.sum(vv, axis=1) / nb
        z_ave = prox_normL1(
            x_admm_dist[:, k + 1] + v_hat, set_up["Lambda"] / (set_up["ro"] * nb)
        )
        vv = vv + xx - np.kron(z_ave.reshape(-1, 1), np.ones((1, nb)))
    return x_admm_dist


def admm_ridge_dist(set_up):
    nb = set_up["Number_nodes"]
    n = set_up["Niter_train"]
    Naux = int(n / nb)
    x_admm_dist = np.zeros((set_up["d"] + 1, set_up["Niter_train"]))
    vv = np.zeros((set_up["d"] + 1, nb))
    xx = np.zeros((set_up["d"] + 1, nb))
    z_ave = np.zeros(set_up["d"] + 1)
    for k in range(1, Naux):
        for kk in range(nb):
            Xaux = set_up["Xtrain"][kk * Naux : (kk + 1) * Naux, : set_up["d"] + 1]
            yaux = set_up["ytrain"][kk * Naux : (kk + 1) * Naux, 0]
            xx[:, kk] = prox_quadratic(
                Xaux.T @ Xaux * 2 / n,
                -2 * Xaux.T @ yaux / n,
                z_ave - vv[:, kk],
                set_up["ro"],
            )
        x_admm_dist[:, k + 1] = np.sum(xx, axis=1) / nb
        v_hat = np.sum(vv, axis=1) / nb
        z_ave = prox_quadratic(
            np.eye(set_up["d"] + 1) * set_up["Lambda"],
            np.zeros(set_up["d"] + 1),
            x_admm_dist[:, k + 1] + v_hat,
            set_up["ro"] * nb,
        )
        vv = vv + xx - np.kron(z_ave.reshape(-1, 1), np.ones((1, nb)))
    return x_admm_dist


def forward(W1, W2, b1, b2, y0, tipo):
    v1 = W1 @ y0 + b1
    y1 = logistic(v1)
    v2 = W2 @ y1 + b2
    if tipo == "linear":
        y2 = v2
    else:
        y2 = logistic(v2)
    return y1, y2, v1, v2


def backward(W2, y1, y2, e, tipo):
    if tipo == "linear":
        e1 = e
    else:
        e1 = e * y2 * (1 - y2)
    e2 = e1 * np.diag(y1 * (1 - y1)) @ W2.T
    return e1, e2


def logistic(x):
    return 1 / (1 + np.exp(-x))


def plot_surface(
    set_up, loss, c_opt, include_grad=False, grad="None", color="None", linestyle="None"
):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 2, 1)
    ix = np.arange(-20, 20, 0.5)
    iy = ix.copy()
    Mx, My = np.meshgrid(ix, iy)
    S = np.zeros_like(Mx)
    for kx in range(len(ix)):
        for ky in range(len(iy)):
            ww = np.r_[Mx[kx, ky], My[kx, ky]]
            S[kx, ky] = loss(
                set_up["Niter_train"],
                set_up["Xtrain"][:, :2],
                set_up["ytrain"][:, 0],
                ww,
                set_up["Lambda"],
            ).value
    if include_grad:
        # Gradient or Newton descent
        out_gd = grad

    ax.contour(Mx, My, S, 10)
    ax.grid()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Coefficient 1")
    ax.set_ylabel("Coefficient 2")
    ax.set_title("Error surface. Contour plot.")
    ax.plot(c_opt[0], c_opt[1], marker="H", color="r", linewidth=5)
    if include_grad:
        if len(out_gd.shape) > 2:
            if color == "None" and linestyle == "None":
                for k in range(out_gd.shape[0]):
                    ax.plot(out_gd[k, 0, :], out_gd[k, 1, :], "green", linewidth=3)
            elif color == "None":
                for k in range(out_gd.shape[0]):
                    ax.plot(
                        out_gd[k, 0, :],
                        out_gd[k, 1, :],
                        linewidth=3,
                        linestyle=linestyle[k],
                    )
            elif linestyle == "None":
                for k in range(out_gd.shape[0]):
                    ax.plot(
                        out_gd[k, 0, :], out_gd[k, 1, :], color=color[k], linewidth=3
                    )
            else:
                for k in range(out_gd.shape[0]):
                    ax.plot(
                        out_gd[k, 0, :],
                        out_gd[k, 1, :],
                        color=color[k],
                        linewidth=3,
                        linestyle=linestyle[k],
                    )
        else:
            ax.plot(out_gd[0, :], out_gd[1, :], "green", linewidth=3)

    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.plot_surface(Mx, My, S, cmap=cm.viridis)
    ax.set_xlabel("Coefficient 1")
    ax.set_ylabel("Coefficient 2")
    ax.set_title("Error surface.")
    return S
