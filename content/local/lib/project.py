import random
from local.lib.data import load_data


def project_sel(name):
    random.seed(name)
    type = round(10 * random.random()) % 2
    if type == 0:
        problem = "classification"
        index = random.randint(1, 12)
    else:
        problem = "regression"
        index = random.randint(1, 19)

    # load data
    X, y = load_data(problem, index)

    # Select model
    Niter = 1000
    # Number of iterations for each algorithm
    ro = 10  # Quadratic term
    type = round(10 * random.random()) % 2
    if problem == "classification":
        type2 = round(10 * random.random()) % 2
        if type == 0:
            lambd = 10
            if type2 == 0:
                model = "Logistic regression with L1 regularization"
            else:
                model = "Logistic regression with L2 regularization"
        else:
            model = "SVM"
            lambd = 0.01
    else:
        if type == 0:
            model = "Ridge regression"
            Niter = 5000
            # Number of iterations for each algorithm
            lambd = 0.1
        else:
            model = "Lasso regression"
            lambd = 0.01
    print(
        "Use the data set of the {problem} problem given to you to train a {model} model.\n".format(
            problem=problem, model=model
        )
    )
    print(
        "The global number of iterations for each algorithm is {Niter}.\n".format(
            Niter=Niter
        )
    )
    print(
        "Use an initial regularization parameter of {lambd} for the regularization terms and {ro} for the rho quadratic term in ADMM.\n".format(
            lambd=lambd, ro=ro
        )
    )

    return X, y
