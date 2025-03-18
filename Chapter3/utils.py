import numpy as np
from math import log
from scipy.optimize import minimize

def get_events(predictions: np.array, labels=[0, 1]) -> dict:
    """Extract start and end points of each labeled events

        Parameters
        ----------
        predictions : np.array of binary predictions
        labels : list of labels used for predictions (default=[0,1])
    

        Returns
        ----------
        events_all_labels : dict of lists of list
            Contains for each label a list of the start and end points of every event labeled as such
    """
    changes_index = np.where(predictions[:-1] != predictions[1:])[0]
    events_all_labels = {label: list() for label in labels}
    if len(changes_index):
        for i, ind in enumerate(changes_index):
            if i != len(changes_index) - 1:
                events_all_labels[predictions[ind + 1]].append((ind + 1, changes_index[i + 1] + 1))
            else:
                events_all_labels[predictions[ind + 1]].append((ind + 1, len(predictions)))
        events_all_labels[predictions[0]].append((0, changes_index[0]))
    else:
        events_all_labels[predictions[0]].append((0, len(predictions)))
    return events_all_labels

def safe_divide(a, b, filler=10e-5):
        if b:
            return a / b
        else:
            return a / filler
        
def grimshaw(peaks: np.array, Ncandidates=5, epsilon=1e-8):
    """The Grimshaw's Trick Method for estimating the GPD parameters
 
        Parameters
        ----------
        peaks : np.array with the scores' peaks (values above an initial threshold)
        Ncandidates : number of candidate solutions for the GPD paramters (default = 5)
        epsilon : numerical parameter for the solving (default = 1e-8)

    Returns:
        gamma_best : float GPD shape parameter
        sigma_best : float GPD scale parameter
    """

    ## Util functions definitions
    def function(x, threshold):
        ## We want to solve u(p) * v(p) = 1
        s = 1 + threshold * x
        u = 1 + np.log(s).mean()
        v = np.mean(1 / s)
        return u * v - 1

    def deriv_function(x, threshold):
        ## Derivation of (u(p)*(p) -1)
        s = 1 + threshold * x
        u = 1 + np.log(s).mean()
        v = np.mean(1 / s)
        deriv_u = (1 / threshold) * (1 - v)
        deriv_v = (1 / threshold) * (-v + np.mean(1 / s ** 2))
        return u * deriv_v + v * deriv_u

    def obj_function(x, function, deriv_function):
        m = 0
        n = np.zeros(len(x))
        for i_peak, peaks in enumerate(x):
            y = function(peaks)
            m += y ** 2
            n[i_peak] = 2 * y * deriv_function(peaks)
        return m, n

    def solve(function, deriv_function, bounds, Ncandidates):
        step = (bounds[1] - bounds[0]) / (Ncandidates + 1)
        x0 = np.arange(bounds[0] + step, bounds[1], step)
        optimization = minimize(
            lambda x: obj_function(x, function, deriv_function), 
            x0, 
            method='L-BFGS-B', 
            jac=True, 
            bounds=[bounds]*len(x0)
        )
        candidates = np.round(optimization.x, decimals=5)
        return np.unique(candidates)

    def log_likelihood_func(peaks, gamma, sigma):
        if gamma != 0:
            p = gamma/sigma
            log_likelihood = -peaks.size * log(sigma) - (1 + (1 / gamma)) * (np.log(1 + p * peaks)).sum()
        else: 
            log_likelihood = peaks.size * (1 + log(peaks.mean()))
        return log_likelihood
    
    min_peak = peaks.min()
    max_peak = peaks.max()
    mean_peak = peaks.mean()

    if abs(-1 / max_peak) < 2 * epsilon:
        epsilon = abs(-1 / max_peak) / Ncandidates

    lower_bound_gamma = -1 / max_peak + epsilon
    lower_bound_sigma = 2 * (mean_peak - min_peak) / (mean_peak * min_peak)
    upper_bound_sigma = 2 * (mean_peak - min_peak) / (min_peak ** 2)

    candidates_gamma = solve(
        function=lambda threshold: function(peaks, threshold), 
        deriv_function=lambda threshold: deriv_function(peaks, threshold), 
        bounds=(lower_bound_gamma, -epsilon), 
        Ncandidates=Ncandidates
    )
    candidates_sigma = solve(
        function=lambda threshold: function(peaks, threshold), 
        deriv_function=lambda threshold: deriv_function(peaks, threshold), 
        bounds=(lower_bound_sigma, upper_bound_sigma), 
        Ncandidates=Ncandidates
    )
    candidates_params = np.concatenate([candidates_gamma, candidates_sigma])

    gamma_best = 0
    sigma_best = mean_peak
    log_likelihood_best = log_likelihood_func(peaks, gamma_best, sigma_best)
        
    for candidate in candidates_params:
        gamma = np.log(1 + candidate * peaks).mean()
        sigma = safe_divide(gamma, candidate)
        log_likelihood = log_likelihood_func(peaks, gamma, sigma)
        if log_likelihood > log_likelihood_best:
            gamma_best = gamma
            sigma_best = sigma
            log_likelihood_best = log_likelihood

    return gamma_best, sigma_best