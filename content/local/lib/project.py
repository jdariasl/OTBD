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
        "You must chose three algorithms to solve the problem (one for each of the following items):\n"
    )
    print(
        "* "
        + "Gradient/subgradient descent with constant/decay learning rate; accelerated gradient/subgradient descent; proximal grandient descent\n"
    )
    print(
        "* "
        + "Conjugate gradient descend; BFGS; Block coordinate descent assuming, at least, 2 nodes\n"
    )
    print("* " + "Distributed ADMM assuming, at least, 2 nodes\n")
    print(
        "Note that not all the algorithms are suitable for all the problems, so the selection must be done carefully.\n"
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
    print(
        "Note that the values provided for the number of iterations, regularization parameter and rho are orientative, and the student must find a good hyperparameter set that solves adequatelly its problem. You can also use Bayesian optimization to find the best hyperparameters.\n"
    )
    print(
        "The student must provide this python notebook adding the following information:\n"
    )
    print("* " + "A description of the problem and the dataset used.\n")
    print("* " + "A description of each algorithm chosen.\n")
    print("* " + "The code commented.\n")
    print(
        "* "
        + "The results obtained and comment these results, including the advantages and disadvantages of each algorithm used. Use as many graphs and block diagrams as needed.\n"
    )
    print(
        "A fully executable python notebook, as well as its pdf version, must be uploaded to Moodle before the final exam.\n"
    )
    print("The evaluation will be done according to the following criteria:\n")
    print("* " + "Project acomplishment.\n")
    print("* " + "Correctness of the results.\n")
    print("* " + "Quality of the results discussions.\n")
    print("* " + "Correctness of the code.\n")
    print("* " + "Quality of the comments.\n")
    return X, y
