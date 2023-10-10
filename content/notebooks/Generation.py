from scipy.linalg import toeplitz
import numpy as np


def Get_data_reg(data, n):
    # data: structure with data information
    # n: number of data to be generated
    Corr_Mat = toeplitz(
        np.power(data["Var_x"] * data["Coef_corr_x"], np.arange(0, data["d"]))
    )
    X = np.random.multivariate_normal(data["Mean_x"], Corr_Mat, size=n)
    y = (
        X @ data["Reg_opt"][0 : data["d"]].reshape(-1, 1)
        + data["Reg_opt"][data["d"]]
        + np.random.randn(n, 1) * np.sqrt(data["Var_pert"])
    )
    return X, y.squeeze()


def Get_data_class(data, n):
    # data: structure with data information
    # n: number of data to be generated
    n_h = int(n / 2)
    Corr_Mat1 = toeplitz(
        np.power(data["Var_x1"] * data["Coef_corr_x1"], np.arange(0, data["d"]))
    )
    X1 = np.random.multivariate_normal(data["Mean_x1"], Corr_Mat1, size=n_h)
    Corr_Mat2 = toeplitz(
        np.power(data["Var_x2"] * data["Coef_corr_x2"], np.arange(0, data["d"]))
    )
    X2 = np.random.multivariate_normal(data["Mean_x2"], Corr_Mat2, size=n_h)
    p = np.random.permutation(n)
    i1 = p[0:n_h]
    i2 = p[n_h:n]
    X = np.zeros((n, data["d"]))
    y = np.zeros((n, 1))
    X[i1, :] = X1
    X[i2, :] = X2
    y[i1] = np.ones((n_h, 1))
    y[i2] = -np.ones((n_h, 1))
    return X, y.squeeze()


def scenarios_regression(number):
    if number == 1:
        # Main parameters
        # ===========================
        mu_grad = 0.01  # Step size (fixed) for gradient
        mu_grad_decay = mu_grad * 40  # Step size (decaying) for gradient
        mu_hess = 0.8  # Step size (fixed) for hessian
        wlambda = 1  # Weight of regularizer
        q = 1  # Weight of the proximal term
        ro = 1  # Weight of the augmented Lagrangian
        Niter_test = 3000  # Number of iterations per test
        Niter_train = 3000  # Number of iterations for training
        Number_tests = 5  # Number of tests
        Number_nodes = 5  # Number of nodes
        Number_blocks = 3  # Number of blocks of variables
        Number_iter_BCD = 30  # Number of iterations for the Block Coordinate Descent
        tau = 1  # Weight of the quadratic term in the FLEXA
        graph_degree = 3  # Degree of the graph

        # Data generation
        # ===========================
        d = 1
        # Dimensionality of data vector
        initial = [-10, -20]  # Starting point of iterative / adaptive algorithms
        mean_x = np.array([1])  # Mean of input data
        coef_corr_x = 0.2  # Correlation coefficient of input data
        var_x = 1  # Variance of input data
        reg_opt = np.array([2, 1])  # Optimum regressor
        var_pert = 1  # Variance of the perturbation

        # Build data struct
        # ===========================
        data_reg = {
            "d": d,
            "Mean_x": mean_x,
            "Coef_corr_x": coef_corr_x,
            "Var_x": var_x,
            "Reg_opt": reg_opt,
            "Var_pert": var_pert,
        }

        # Data variables
        # ===========================
        Xtrain_e = np.zeros((Niter_train, Number_nodes * (d + 1)))
        ytrain = np.zeros((Niter_train, Number_nodes))
        for k in range(Number_nodes):
            Xtrain, ytrain[:, k] = Get_data_reg(data_reg, Niter_train)
            Xtrain_e[:, k * (d + 1) : (k + 1) * (d + 1)] = np.c_[
                Xtrain, np.ones((Niter_train, 1))
            ]

        Xtest_e = np.zeros((Niter_test, Number_nodes * (d + 1)))
        ytest = np.zeros((Niter_test, Number_nodes))
        for k in range(Number_nodes):
            Xtest, ytest[:, k] = Get_data_reg(data_reg, Niter_test)
            Xtest_e[:, k * (d + 1) : (k + 1) * (d + 1)] = np.c_[
                Xtest, np.ones((Niter_test, 1))
            ]

        # Definition of the problem setup
        # ==============================
        set_up = {
            "Xtrain": Xtrain_e,
            "ytrain": ytrain,
            "Xtest": Xtest_e,
            "ytest": ytest,
            "d": d,
            "q": q,
            "ro": ro,
            "mu_grad": mu_grad,
            "mu_grad_decay": mu_grad_decay,
            "mu_hess": mu_hess,
            "Lambda": wlambda,
            "Niter_test": Niter_test,
            "Niter_train": Niter_train,
            "Number_tests": Number_tests,
            "Number_nodes": Number_nodes,
            "graph_degree": graph_degree,
            "Initial": initial,
        }

    elif number == 2:
        # Main parameters
        # ===========================
        mu_grad = 0.006  # Step size (fixed) for gradient
        mu_grad_decay = mu_grad * 70  # Step size (decaying) for gradient
        mu_hess = 0.8  # Step size (fixed) for hessian
        wlambda = 0.1  # Weight of regularizer
        q = 1 / mu_grad  # Weight of the proximal term
        Niter_test = 3000  # Number of iterations per test
        Niter_train = 3000  # Number of iterations for training
        Number_tests = 5  # Number of tests
        ro = 1  # Weight of the augmented Lagrangian
        Number_nodes = 5  # Number of nodes
        Number_blocks = 3  # Number of blocks
        tau = 1  # Weight of the quadratic term in the FLEXA
        Number_iter_BCD = 30  # Number of iterations for the Block Coordinate Descent
        graph_degree = 3  # Degree of the graph

        # Data generation
        # ===========================
        d = 10  # Dimensionality of data vector
        initial = -0.1 * np.ones(
            d + 1
        )  # Starting point of iterative / adaptive algorithms
        mean_x = np.ones(d)  # Mean of input data
        coef_corr_x = 0.2  # Correlation coefficient of input data
        var_x = 1  # Variance of input data
        reg_opt = (
            np.array([2, 1, 3, 5, -2, -4, 7, 1, 2, -4, -1]) / 10
        )  # Optimum regressor
        var_pert = 1  # Variance of the perturbation

        # Build data struct
        data_reg = {
            "d": d,
            "Mean_x": mean_x,
            "Coef_corr_x": coef_corr_x,
            "Var_x": var_x,
            "Reg_opt": reg_opt,
            "Var_pert": var_pert,
        }

        # Data variables
        # ===========================
        Xtrain_e = np.zeros((Niter_train, Number_nodes * (d + 1)))
        ytrain = np.zeros((Niter_train, Number_nodes))
        for k in range(Number_nodes):
            Xtrain, ytrain[:, k] = Get_data_reg(data_reg, Niter_train)
            Xtrain_e[:, k * (d + 1) : (k + 1) * (d + 1)] = np.c_[
                Xtrain, np.ones((Niter_train, 1))
            ]

        Xtest_e = np.zeros((Niter_test, Number_nodes * (d + 1)))
        ytest = np.zeros((Niter_test, Number_nodes))
        for k in range(Number_nodes):
            Xtest, ytest[:, k] = Get_data_reg(data_reg, Niter_test)
            Xtest_e[:, k * (d + 1) : (k + 1) * (d + 1)] = np.c_[
                Xtest, np.ones((Niter_test, 1))
            ]

        # Definition of the problem setup
        # ==============================
        set_up = {
            "Xtrain": Xtrain_e,
            "ytrain": ytrain,
            "Xtest": Xtest_e,
            "ytest": ytest,
            "d": d,
            "q": q,
            "ro": ro,
            "mu_grad": mu_grad,
            "mu_grad_decay": mu_grad_decay,
            "mu_hess": mu_hess,
            "Lambda": wlambda,
            "Niter_test": Niter_test,
            "Niter_train": Niter_train,
            "Number_tests": Number_tests,
            "Number_nodes": Number_nodes,
            "graph_degree": graph_degree,
            "Initial": initial,
        }

    elif number == 3:
        # Main parameters
        # ===========================
        mu_grad = 0.0008  # Step size (fixed) for gradient
        mu_grad_decay = mu_grad * 70  # Step size (decaying) for gradient
        mu_hess = 0.8  # Step size (fixed) for hessian
        wlambda = 1  # Weight of regularizer
        q = 1 / mu_grad
        # Weight of the proximal term
        Niter_test = 3000  # Number of iterations per test
        Niter_train = 3000  # Number of iterations for training
        Number_tests = 5  # Number of tests
        ro = 1 / Number_tests  # Weight of the augmented Lagrangian
        Number_nodes = 5  # Number of nodes
        Number_blocks = 3  # Number of blocks
        tau = 1  # Weight of the quadratic term in the FLEXA
        Number_iter_BCD = 30  # Number of iterations for the Block Coordinate Descent
        graph_degree = 3  # Degree of the graph

        # Data generation
        # ===========================
        d = 10  # Dimensionality of data vector
        initial = -0.1 * np.ones(
            d + 1
        )  # Starting point of iterative / adaptive algorithms
        mean_x = np.ones(d)  # Mean of input data
        coef_corr_x = 0.2  # Correlation coefficient of input data
        var_x = 1  # Variance of input data
        reg_opt = (
            np.array([2, 1, 3, 5, -2, -4, 7, 1, 2, -4, -1]) / 10
        )  # Optimum regressor
        var_pert = 1  # Variance of the perturbation

        # Build data struct
        data_reg = {
            "d": d,
            "Mean_x": mean_x,
            "Coef_corr_x": coef_corr_x,
            "Var_x": var_x,
            "Reg_opt": reg_opt,
            "Var_pert": var_pert,
        }

        # Data variables
        # ===========================
        Xtrain_e = np.zeros((Niter_train, Number_nodes * (d + 1)))
        ytrain = np.zeros((Niter_train, Number_nodes))
        for k in range(Number_nodes):
            Xtrain, ytrain[:, k] = Get_data_reg(data_reg, Niter_train)
            Xtrain_e[:, k * (d + 1) : (k + 1) * (d + 1)] = np.c_[
                Xtrain, np.ones((Niter_train, 1))
            ]

        Xtest_e = np.zeros((Niter_test, Number_nodes * (d + 1)))
        ytest = np.zeros((Niter_test, Number_nodes))
        for k in range(Number_nodes):
            Xtest, ytest[:, k] = Get_data_reg(data_reg, Niter_test)
            Xtest_e[:, k * (d + 1) : (k + 1) * (d + 1)] = np.c_[
                Xtest, np.ones((Niter_test, 1))
            ]

        # Definition of the problem setup
        # ==============================
        set_up = {
            "Xtrain": Xtrain_e,
            "ytrain": ytrain,
            "Xtest": Xtest_e,
            "ytest": ytest,
            "d": d,
            "q": q,
            "ro": ro,
            "mu_grad": mu_grad,
            "mu_grad_decay": mu_grad_decay,
            "mu_hess": mu_hess,
            "Lambda": wlambda,
            "Niter_test": Niter_test,
            "Niter_train": Niter_train,
            "Number_tests": Number_tests,
            "Number_nodes": Number_nodes,
            "graph_degree": graph_degree,
            "Initial": initial,
        }

    elif number == 4:
        # Main parameters
        # ===========================
        mu_grad = 0.01  # Step size (fixed) for gradient
        mu_grad_decay = mu_grad * 40  # Step size (ecaying) for gradient
        mu_hess = 0.8  # Step size (fixed) for hessian
        wlambda = 1  # Weight of regularizer
        q = 1 / mu_grad  # Weight of the proximal term
        ro = 1  # Weight of the augmented Lagrangian
        Niter_test = 3000  # Number of iterations per test
        Niter_train = 500  # Number of iterations for training
        Number_tests = 1  # Number of tests
        Number_nodes = 5  # Number of nodes
        Number_blocks = 3  # Number of blocks
        Number_iter_BCD = 10  # Number of iterations for the Block Coordinate Descent
        tau = 1  # Weight of the quadratic term in the BCD
        graph_degree = 3  # Degree of the graph

        # Data generation
        # ===========================
        d = 11
        # Dimensionality of data vector
        initial = np.random.randn(d + 1)
        # Starting point of iterative / adaptive algorithms
        mean_x = np.zeros(d)
        # Mean of input data
        coef_corr_x = 0.2
        # Correlation coefficient of input data
        var_x = 1
        # Variance of input data
        reg_opt = np.random.randn(d + 1)  # Optimum regressor
        var_pert = 1
        # Variance of the perturbation
        # Build data struct
        data_reg = {
            "d": d,
            "Mean_x": mean_x,
            "Coef_corr_x": coef_corr_x,
            "Var_x": var_x,
            "Reg_opt": reg_opt,
            "Var_pert": var_pert,
        }

        # Data variables
        # ===========================
        Xtrain_e = np.zeros((Niter_train, Number_nodes * (d + 1)))
        ytrain = np.zeros((Niter_train, Number_nodes))
        for k in range(Number_nodes):
            Xtrain, ytrain[:, k] = Get_data_reg(data_reg, Niter_train)
            Xtrain_e[:, k * (d + 1) : (k + 1) * (d + 1)] = np.c_[
                Xtrain, np.ones((Niter_train, 1))
            ]

        Xtest_e = np.zeros((Niter_test, Number_nodes * (d + 1)))
        ytest = np.zeros((Niter_test, Number_nodes))
        for k in range(Number_nodes):
            Xtest, ytest[:, k] = Get_data_reg(data_reg, Niter_test)
            Xtest_e[:, k * (d + 1) : (k + 1) * (d + 1)] = np.c_[
                Xtest, np.ones((Niter_test, 1))
            ]

        # Definition of the problem setup
        # ==============================
        set_up = {
            "Xtrain": Xtrain_e,
            "ytrain": ytrain,
            "Xtest": Xtest_e,
            "ytest": ytest,
            "d": d,
            "q": q,
            "ro": ro,
            "mu_grad": mu_grad,
            "mu_grad_decay": mu_grad_decay,
            "mu_hess": mu_hess,
            "Lambda": wlambda,
            "Niter_test": Niter_test,
            "Niter_train": Niter_train,
            "tau": tau,
            "Number_tests": Number_tests,
            "Number_iter_BCD": Number_iter_BCD,
            "Number_blocks": Number_blocks,
            "Number_nodes": Number_nodes,
            "graph_degree": graph_degree,
            "Initial": initial,
        }

    return data_reg, set_up


def scenarios_classification(number):
    if number == 1:
        # Main parameters
        # ===========================
        mu_grad = 0.08  # Step size (fixed) for gradient
        mu_hess = 0.8  # Step size (decaying) for hessian
        wlambda = 1  # Weight of regularizer
        q = 1  # Weight of the proximal term
        ro = 1.1  # Weight of the augmented Lagrangian
        Niter_test = 500  # Number of iterations per test
        Niter_train = 500  # Number of iterations for training
        Number_tests = 5  # Number of tests
        Number_nodes = 5  # Number of tests
        Number_blocks = 3  # Number of blocks
        Number_iter_BCD = 30  # Number of iterations for the Block Coordinate Descent
        graph_degree = 3  # Degree of the graph
        tau = 1  # Weight of the quadratic term in the FLEXA

        # Data generation
        # ===========================
        d = 2  # Dimensionality of data vector
        initial = np.array(
            [10, 10, 0]
        )  # Starting point of iterative / adaptive algorithms
        mean_x1 = np.array([2, 2])  # Mean of input data class 1
        coef_corr_x1 = 0.2  # Correlation coefficient of input data class 1
        var_x1 = 1.5  # Variance of input data class 1
        mean_x2 = np.array([-2, -2])  # Mean of input data class 1
        coef_corr_x2 = 0.2  # Correlation coefficient of input data class 1
        var_x2 = 1.5  # Variance of input data class 1

        # Build data struct
        data_class = {
            "d": d,
            "Mean_x1": mean_x1,
            "Coef_corr_x1": coef_corr_x1,
            "Var_x1": var_x1,
            "Mean_x2": mean_x2,
            "Coef_corr_x2": coef_corr_x2,
            "Var_x2": var_x2,
        }

        # Data variables
        # ===========================
        Xtrain_e = np.zeros((Niter_train, Number_nodes * (d + 1)))
        ytrain = np.zeros((Niter_train, Number_nodes))
        for k in range(Number_nodes):
            Xtrain, ytrain[:, k] = Get_data_class(data_class, Niter_train)
            Xtrain_e[:, k * (d + 1) : (k + 1) * (d + 1)] = np.c_[
                Xtrain, np.ones((Niter_train, 1))
            ]

        Xtest_e = np.zeros((Niter_test, Number_nodes * (d + 1)))
        ytest = np.zeros((Niter_test, Number_nodes))
        for k in range(Number_nodes):
            Xtest, ytest[:, k] = Get_data_class(data_class, Niter_test)
            Xtest_e[:, k * (d + 1) : (k + 1) * (d + 1)] = np.c_[
                Xtest, np.ones((Niter_test, 1))
            ]

        # Definition of the problem setup
        # ==============================
        set_up = {
            "Xtrain": Xtrain_e,
            "ytrain": ytrain,
            "Xtest": Xtest_e,
            "ytest": ytest,
            "d": d,
            "q": q,
            "ro": ro,
            "mu_grad": mu_grad,
            "mu_hess": mu_hess,
            "Lambda": wlambda,
            "Niter_test": Niter_test,
            "Niter_train": Niter_train,
            "Number_tests": Number_tests,
            "Number_nodes": Number_nodes,
            "Number_blocks": Number_blocks,
            "Number_iter_BCD": Number_iter_BCD,
            "graph_degree": graph_degree,
            "tau": tau,
            "Initial": initial,
        }

    elif number == 2:
        # Main parameters
        # ===========================
        mu_grad = 0.08  # Step size (fixed) for gradient
        mu_hess = 0.4  # Step size (fixed) for hessian
        wlambda = 1  # Weight of regularizer
        q = 1  # Weight of the proximal term
        ro = 1  # Weight of the augmented Lagrangian
        Niter_test = 500  # Number of iterations per test
        Niter_train = 500  # Number of iterations for training
        Number_tests = 5  # Number of tests
        Number_nodes = 5  # Number of nodes
        Number_blocks = 3  # Number of blocks
        Number_iter_BCD = 30  # Number of iterations for the Block Coordinate Descent
        graph_degree = 3  # Degree of the graph
        tau = 1  # Weight of the quadratic term in the FLEXA

        # Data generation
        # ===========================
        d = 5
        # Dimensionality of data vector
        initial = 10 * np.ones(
            d + 1
        )  # Starting point of iterative / adaptive algorithms
        mean_x1 = 2 * np.ones(d)  # Mean of input data class 1
        coef_corr_x1 = 0.2  # Correlation coefficient of input data class 1
        var_x1 = 0.5  # Variance of input data class 1
        mean_x2 = -2 * np.ones(d)  # Mean of input data class 1
        coef_corr_x2 = 0.2  # Correlation coefficient of input data class 1
        var_x2 = 0.5  # Variance of input data class 1

        # Build data struct
        data_class = {
            "d": d,
            "Mean_x1": mean_x1,
            "Coef_corr_x1": coef_corr_x1,
            "Var_x1": var_x1,
            "Mean_x2": mean_x2,
            "Coef_corr_x2": coef_corr_x2,
            "Var_x2": var_x2,
        }

        # Data variables
        # ===========================
        Xtrain_e = np.zeros((Niter_train, Number_nodes * (d + 1)))
        ytrain = np.zeros((Niter_train, Number_nodes))
        for k in range(Number_nodes):
            Xtrain, ytrain[:, k] = Get_data_class(data_class, Niter_train)
            Xtrain_e[:, k * (d + 1) : (k + 1) * (d + 1)] = np.c_[
                Xtrain, np.ones((Niter_train, 1))
            ]

        Xtest_e = np.zeros((Niter_test, Number_nodes * (d + 1)))
        ytest = np.zeros((Niter_test, Number_nodes))
        for k in range(Number_nodes):
            Xtest, ytest[:, k] = Get_data_class(data_class, Niter_test)
            Xtest_e[:, k * (d + 1) : (k + 1) * (d + 1)] = np.c_[
                Xtest, np.ones((Niter_test, 1))
            ]

        # Definition of the problem setup
        # ==============================
        set_up = {
            "Xtrain": Xtrain_e,
            "ytrain": ytrain,
            "Xtest": Xtest_e,
            "ytest": ytest,
            "d": d,
            "q": q,
            "ro": ro,
            "mu_grad": mu_grad,
            "mu_hess": mu_hess,
            "Lambda": wlambda,
            "Niter_test": Niter_test,
            "Niter_train": Niter_train,
            "Number_tests": Number_tests,
            "Number_nodes": Number_nodes,
            "Number_blocks": Number_blocks,
            "Number_iter_BCD": Number_iter_BCD,
            "graph_degree": graph_degree,
            "tau": tau,
            "Initial": initial,
        }

    elif number == 3:
        # Main parameters
        # ===========================
        mu_grad = 0.08  # Step size (fixed) for gradient
        mu_hess = 0.8  # Step size (decaying) for hessian
        wlambda = 1  # Weight of regularizer
        q = 1  # Weight of the proximal term
        Niter_test = 200  # Number of iterations per test
        Niter_train = 200  # Number of iterations for training
        Number_tests = 5  # Number of tests
        Number_nodes = 5  # Number of tests
        ro = 1  # Weight of the augmented Lagrangian
        Number_blocks = 3  # Number of blocks
        Number_iter_BCD = 30  # Number of iterations for the Block Coordinate Descent
        graph_degree = 3  # Degree of the graph
        tau = 1  # Weight of the quadratic term in the FLEXA

        # Data generation
        # ===========================
        d = 2  # Dimensionality of data vector
        initial = np.array(
            [10, 10, 0]
        )  # Starting point of iterative / adaptive algorithms
        mean_x1 = np.array([2, 2])  # Mean of input data class 1
        coef_corr_x1 = 0.2  # Correlation coefficient of input data class 1
        var_x1 = 1.5  # Variance of input data class 1
        mean_x2 = np.array([-2, -2])  # Mean of input data class 1
        coef_corr_x2 = 0.2  # Correlation coefficient of input data class 1
        var_x2 = 1.5  # Variance of input data class 1

        # Build data struct
        data_class = {
            "d": d,
            "Mean_x1": mean_x1,
            "Coef_corr_x1": coef_corr_x1,
            "Var_x1": var_x1,
            "Mean_x2": mean_x2,
            "Coef_corr_x2": coef_corr_x2,
            "Var_x2": var_x2,
        }

        # Data variables
        # ===========================
        Xtrain_e = np.zeros((Niter_train, Number_nodes * (d + 1)))
        ytrain = np.zeros((Niter_train, Number_nodes))
        for k in range(Number_nodes):
            Xtrain, ytrain[:, k] = Get_data_class(data_class, Niter_train)
            Xtrain_e[:, k * (d + 1) : (k + 1) * (d + 1)] = np.c_[
                Xtrain, np.ones((Niter_train, 1))
            ]

        Xtest_e = np.zeros((Niter_test, Number_nodes * (d + 1)))
        ytest = np.zeros((Niter_test, Number_nodes))
        for k in range(Number_nodes):
            Xtest, ytest[:, k] = Get_data_class(data_class, Niter_test)
            Xtest_e[:, k * (d + 1) : (k + 1) * (d + 1)] = np.c_[
                Xtest, np.ones((Niter_test, 1))
            ]

        # Definition of the problem setup
        # ==============================
        set_up = {
            "Xtrain": Xtrain_e,
            "ytrain": ytrain,
            "Xtest": Xtest_e,
            "ytest": ytest,
            "d": d,
            "q": q,
            "ro": ro,
            "mu_grad": mu_grad,
            "mu_hess": mu_hess,
            "Lambda": wlambda,
            "Niter_test": Niter_test,
            "Niter_train": Niter_train,
            "Number_tests": Number_tests,
            "Number_nodes": Number_nodes,
            "Number_blocks": Number_blocks,
            "Number_iter_BCD": Number_iter_BCD,
            "graph_degree": graph_degree,
            "tau": tau,
            "Initial": initial,
        }

    elif number == 4:
        # Main parameters
        # ===========================
        mu_grad = 0.08  # Step size (fixed) for gradient
        mu_hess = 0.8  # Step size (fixed) for hessian
        wlambda = 1  # Weight of regularizer
        q = 1  # Weight of the proximal term
        ro = 1.1  # Weight of the augmented Lagrangian
        Niter_test = 500  # Number of iterations per test
        Niter_train = 500  # Number of iterations for training
        Number_tests = 5  # Number of tests
        initial = 10 * np.ones(6)  # Starting point of iterative / adaptive algorithms

        # Data generation
        # ===========================
        d = 5  # Dimensionality of data vector
        mean_x1 = 2 * np.ones(d)  # Mean of input data class 1
        coef_corr_x1 = 0.2  # Correlation coefficient of input data class 1
        var_x1 = 0.5  # Variance of input data class 1
        mean_x2 = -2 * np.ones(d)  # Mean of input data class 1
        coef_corr_x2 = 0.2  # Correlation coefficient of input data class 1
        var_x2 = 0.5  # Variance of input data class 1

        # Build data struct
        data_class = {
            "d": d,
            "Mean_x1": mean_x1,
            "Coef_corr_x1": coef_corr_x1,
            "Var_x1": var_x1,
            "Mean_x2": mean_x2,
            "Coef_corr_x2": coef_corr_x2,
            "Var_x2": var_x2,
        }

        # Data variables
        # ===========================

        Xtrain, ytrain = Get_data_class(data_class, Niter_train)
        Xtrain_e = np.c_[Xtrain, np.ones((Niter_train, 1))]

        Xtest, ytest = Get_data_class(data_class, Niter_test)
        Xtest_e = np.c_[Xtest, np.ones((Niter_test, 1))]

        # Definition of the problem setup
        # ==============================
        set_up = {
            "Xtrain": Xtrain_e,
            "ytrain": ytrain[..., np.newaxis],
            "Xtest": Xtest_e,
            "ytest": ytest[..., np.newaxis],
            "d": d,
            "q": q,
            "ro": ro,
            "mu_grad": mu_grad,
            "mu_hess": mu_hess,
            "Lambda": wlambda,
            "Niter_test": Niter_test,
            "Niter_train": Niter_train,
            "Number_tests": Number_tests,
            "Initial": initial,
        }

    return data_class, set_up
