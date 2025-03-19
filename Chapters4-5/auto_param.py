import utils
import rbo
import nab
import numpy as np
from scipy import special
from torczon import solve

class AutoProfile():
    """Class for automatic hyperparametrization of compatible metrics. Only MTI and NAB are available currently.
        See Chapter 5.

    Parameters
    ----------
    x0 : list or np.array
        initial values for the hyperparameters search
        For MTI it is [lambda_AP, w_R, w_SpeM, w_CardAl]
        For NAB it is [lambda_k, Afp, Afn]
    xmin : list or np.array
        lower bound for he hyperparameters search
    xmax : list or np.array
        upper bound for he hyperparameters search
    Nguess : int
        number of initial guesses, used in torczon solver
    Niter : int
        the number of iterations by single guess, used in torczon solver
    initial_box_width : float
        the amplitude of the initial steps around each initial guess to bild the polygone of the torczon algorithm.
    """
     
    def __init__(self, x0, xmin, xmax, Nguess, Niter, initial_box_width=0.1, rbo_p: list = [0.2, 1]) -> None:
        self.rbo_p = rbo_p
        self.x0 = x0
        self.xmin = xmin
        self.xmax = xmax
        self.Nguess = Nguess
        self.Niter= Niter
        self.initial_box_width = initial_box_width

    def compute_approx_rbo(self, ranking, names, scores, p):
        """ Compute approximate RBO at finite length. RBO is normalized against RBO of best ranking."""
        pred = utils.rank_with_scores(names, scores)
        approx_rbo = rbo.rbo(ranking, pred, p) / rbo.rbo(ranking, ranking, p)
        return approx_rbo

    def compute_RBO_cost(self, ranking, names, scores):
        """ Compute cost function using RBO"""
        return - sum([self.compute_approx_rbo(ranking, names, scores, iter_p) for iter_p in self.rbo_p])

    def objective_function_mti(self, var, params):
        """ Objective function for MTI use. See chapter 5.2.1.2"""

        N_preds = len(utils.flatten(self.gt_ranking))
        anticip_score_detail = np.zeros(N_preds)
        for i_pred in range(N_preds):
            N_events = len(params['anticip_early_len'])
            tmp_score = 0
            for j_e  in range(N_events):
                tmp_len_anticip = params['anticip_areas'][j_e][1] - params['anticip_areas'][j_e][0]
                score_func = nab.scaledSigmoid((np.arange(0, params['anticip_early_len'][j_e]) - tmp_len_anticip) / params['anticip_early_len'][j_e],
                                            coef=var[0])
                tmp_score += sum(score_func * self.predictions[i_pred][params['anticip_areas'][j_e][0]:
                                                                        params['anticip_areas'][j_e][0]+params['anticip_early_len'][j_e]] / sum(score_func))
            anticip_score_detail[i_pred] = tmp_score / N_events
        params['component'][:, 2] = anticip_score_detail
        scores = np.average(params['component'], weights=[var[1], var[2], 1, var[3]], axis=1)
        return self.compute_RBO_cost(self.gt_ranking, self.names, scores)
    
    def objective_function_nab(self, var, params):
        """ Objective function for NAB use. See chapter 5.2.3"""
        N_preds = len(utils.flatten(self.gt_ranking))
        scores = np.zeros(N_preds)
        for i_pred in range(N_preds):
            scores[i_pred] = nab.nab(params['label'], self.predictions[i_pred], coef=var[0], tp_weight=1, fp_weight=var[1], fn_weight=var[2])
        return self.compute_RBO_cost(self.gt_ranking, self.names, scores)

    def fit(self, gt_ranking, names, predictions, params, metric='MTI'):
        """Fit method for auto-hyperparametrization of MTI or NAB.

        Parameters
        ----------
        gt_ranking : list
            ranking of prediction names. Ties are possible.
        names : list
            list of prediction names.
        predictions: np.array 
            np.array  of all binary predictions
        params : dict
            metric-specific parameters for auto-hyperparametrization
            for MTI it is {'component': np.array of MTI components values
                'anticip_areas': list of [start_point, end_point] of each anticipation area
                'anticip_early_len : list of length of each anticipation + earliness areas
                }
            for NAB it is {'label': list of ground truth binary labels}
        Returns
        ----------
        best_hyperparameters : list of optimal hyperparameters
        final_cost : optimal hyperparameters associated cost function value
        """
        self.gt_ranking = gt_ranking
        self.names = names
        self.predictions = predictions
        if metric == 'MTI':
            self.solution = solve(self.objective_function_mti, par=params, x0=self.x0, xmin=self.xmin, xmax=self.xmax, 
                              Nguess=self.Nguess, Niter=self.Niter, initial_box_width=self.initial_box_width)
            best_hyperparameters = self.solution.x
            final_cost = self.solution.f
            return best_hyperparameters, final_cost
        elif metric == 'NAB':
            self.solution = solve(self.objective_function_nab, par=params, x0=self.x0, xmin=self.xmin, xmax=self.xmax, 
                              Nguess=self.Nguess, Niter=self.Niter, initial_box_width=self.initial_box_width)
            best_hyperparameters = self.solution.x
            final_cost = self.solution.f
            return best_hyperparameters, final_cost