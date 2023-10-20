from scipy.stats import norm
import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

def PI(X, X_sample, gpr, xi=0.01):
    '''
    Probability of improvement
    
    Args:
        X: input points where PI will be estimated
        X_sample: samples observed heretofore (n x d).
        gpr: Trained surrogate GP model
        xi: Exploration-exploitation parameter
    
    Returns:
        PI for every point in X
    '''
    if X.ndim==1:
        X = X[:,None]
    mu, sigma = gpr.predict(X,  return_std=True)
    mu.ravel()
    sigma.ravel()
    mu_sample = gpr.predict(X_sample)
    mu_sample.ravel()
    mu_sample_opt = np.min(mu_sample)

    with np.errstate(divide='warn'):
        imp = mu_sample_opt - mu - xi
        Z = imp / sigma
        pi = norm.cdf(Z) 
    return pi

def EI(X, X_sample, gpr, xi=0.01):
    '''
    Expected improvement
    
    Args:
        X: input points where EI will be estimated
        X_sample: samples observed heretofore (n x d).
        gpr: Trained surrogate GP model
        xi: Exploration-exploitation parameter
    
    Returns:
        EI for every point in X
    '''
    if X.ndim==1:
        X = X[:,None]
    mu, sigma = gpr.predict(X, return_std=True)
    mu.ravel()
    sigma.ravel()
    mu_sample = gpr.predict(X_sample)
    mu_sample.ravel()
    mu_sample_opt = np.min(mu_sample)

    with np.errstate(divide='warn'):
        imp = mu_sample_opt - mu -xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei

def LCB(X, X_sample, gpr, xi=1):
    '''
    Lower Confidence Bound
    
   Args:
        X: input points where EI will be estimated
        X_sample: samples observed heretofore (n x d).
        gpr: Trained surrogate GP model
        xi: Exploration-exploitation parameter
    
    Returns:
        LCB for every point in X
    '''
    if X.ndim==1:
        X = X[:,None]
    mu, sigma = gpr.predict(X, return_std=True)
    mu=mu.reshape(-1,1)
    sigma = sigma.reshape(-1, 1)

    with np.errstate(divide='warn'):
        lcb = -(mu - xi*sigma)
        

    return lcb

def propose_location(acquisition, X_sample, gpr, bounds, n_restarts=20, epsilon=0.01):
    '''
    Find the nex query point by optimizing the acquisition function
    
    Args:
        acquisition: acquisition function.
        X_sample: samples observed heretofore (n x d).
        gpr: Trained surrogate GP model

    Returns:
        Localización del máximo de la función de adquisición.
    '''
    dim = X_sample.shape[1]
    min_val = 1
    min_x = None
    
    def min_obj(X):
        # Minimise the negative of the adquisition function (maximise)
        return -acquisition(X.reshape(-1, dim), X_sample, gpr, xi=epsilon)
    
    # Find the best optimum by starting from n_restart different random points.
    #for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
    for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
        res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')
        if res.fun < min_val:
            min_val = res.fun
            min_x = res.x           
            
    return min_x.reshape(-1, 1)

class BayesianOpt(object):
    def __init__(self, acquisition_function, X,Y,bounds,epsilon=1):
        dic_af={
            'PI': PI,
            'EI': EI,
            'LCB': LCB}
        self.acquisition_function = dic_af[acquisition_function]
        self.init_epsilon = epsilon
        self.epsilon = epsilon
        self.obs_x = X
        self.obs_y = Y
        self.bounds = bounds
        self.model = self.train_surrogate(X,Y)
        self.iter = 0
    def get_next_query_point(self):
        next_x = propose_location(self.acquisition_function,self.obs_x,self.model, self.bounds,epsilon=self.epsilon)
        return next_x
        
    def set_data(self,x,y):
        self.obs_x = np.r_[self.obs_x,x]
        self.obs_y = np.concatenate([self.obs_y,y])
        self.model = self.train_surrogate(self.obs_x,self.obs_y)
        self.iter += 1
        self.epsilon = self.init_epsilon*np.exp(-0.1*self.iter)
    
    def train_surrogate(self,X,Y):
        kernel = ConstantKernel() * RBF(length_scale=0.1, length_scale_bounds=(1e-2, 10.0)) 
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)
        gp.fit(self.obs_x, self.obs_y)
        return gp
    
def plot_bo(ax1, ax2, x,f,xi,yi,gp,next_query=None,ad = None):

    mu_pred, sigma_pred = gp.predict(x, return_std=True)
    
    ax1.plot(x, f(x), 'k', label='True')
    ax1.plot(xi, yi, 'r.', markersize=10, label='Observations')
    ax1.plot(x, mu_pred, 'b-', label=u'Mean Prediction')
    ax1.fill(np.concatenate([x, x[::-1]]),
            np.concatenate([mu_pred -  sigma_pred,
                        (mu_pred + sigma_pred)[::-1]]),
            alpha=.5, fc='b', ec='None', label='$\pm$ 1 std. dev.')
    ax2.set_xlabel('$x$')
    ax1.set_ylabel('$f(x)$')
    ax1.set_ylim(-1.5, 1.5)
    ax1.legend(loc='best')
    #---------------------
    if ad == 'PI':
        ad_pi = PI(x,xi[:, None],gp)
        ax2.plot(x,(ad_pi - np.min(ad_pi))/(np.max(ad_pi) - np.min(ad_pi)),color='dodgerblue', label = 'PI')
    elif ad == 'EI':
        ad_ei = EI(x,xi[:, None],gp)
        ax2.plot(x,(ad_ei - np.min(ad_ei))/(np.max(ad_ei)- np.min(ad_ei)),color='gray' , label= 'EI')
    elif ad == 'LCB':
        ad_ucb = LCB(x,xi[:, None],gp, xi=0.5)
        ax2.plot(x,(ad_ucb - np.min(ad_ucb))/(np.max(ad_ucb)- np.min(ad_ucb)),color='lightseagreen', label = 'LCB')
    else:
        ad_pi = PI(x,xi[:, None],gp)
        ad_ei = EI(x,xi[:, None],gp)
        ad_ucb = LCB(x,xi[:, None],gp, xi=0.5)
        ax2.plot(x,(ad_pi - np.min(ad_pi))/(np.max(ad_pi) - np.min(ad_pi)),color='dodgerblue', label = 'PI')
        ax2.plot(x,(ad_ei - np.min(ad_ei))/(np.max(ad_ei)- np.min(ad_ei)),color='gray' , label= 'EI')
        ax2.plot(x,(ad_ucb - np.min(ad_ucb))/(np.max(ad_ucb)- np.min(ad_ucb)),color='lightseagreen', label = 'LCB')

    ax2.set_title('Acquisition functions')
    ax2.set_ylabel('$\\alpha(x)$')
    if next_query:
        ax1.axvline(x = next_query, color = 'goldenrod', linestyle = 'dashed')
        ax2.axvline(x = next_query, color = 'goldenrod', linestyle = 'dashed', label = f'$x^*={np.round(next_query[0,0],2)}$')
    ax2.legend(loc='best')
     