import utils
import numpy as np
from scipy import stats
from base import BaseThresholder


class Quantile_Thresholder(BaseThresholder):
    """ Wrapping for the Quantile thresholding method. See Chapter 3.2.1.1

       Parameters
       ----------
       contamination : value in ]0,1[ that indicates the prevalence of anomalies in data (default = 0.001)
    """
    def __init__(self, contamination=0.001):
        super().__init__()
        self.contamination = contamination

    def eval(self, scores):
        self.thresh_ = np.quantile(scores, 1 - self.contamination)
        return self

class Probabilistic_Quantile_Thresholder(BaseThresholder):
    """ Wrapping for the Probabilistic Quantile thresholding method. See Chapter 3.2.1.2

       Parameters
       ----------
       contamination : value in ]0,1[ that indicates the prevalence of anomalies in data (default = 0.001)
       delta : value in ]0, 1[ that indicates the estimator power (default = 10e-6)
       seed : value determining the random seed for the bootstrap (default = 0)
    """
    def __init__(self, contamination=0.001, delta=10e-6, seed=0):
        super().__init__()
        self.contamination = contamination
        self.delta = delta
        self.seed = seed

    def eval(self, scores):
        np.random.seed(self.seed)
        N = int((7.47 / self.contamination) * np.log(1 / self.delta))
        r = int(np.floor(self.contamination * N / 2))
        ech = np.random.choice(scores, size=N, replace=True)
        self.thresh_ = np.sort(ech)[-r]
        return self

class Otsu_Thresholder(BaseThresholder):
    """ Wrapping for the Otsu thresholding method. See Chapter 3.2.1.3

       Parameters
       ----------
       Ncandidates : number of candidates for the threshold estimation (default = 50)
    """
    def __init__(self, Ncandidates=50):
        super().__init__()
        self.Ncandidates = Ncandidates

    def eval(self, scores):
        candidates = np.linspace(np.quantile(scores, 0.01), np.quantile(scores, 0.99), num=self.Ncandidates)
        variance_intern = np.zeros(len(candidates))
        for i_c, candidate in enumerate(candidates):
            normal_pred = scores[scores < candidate]
            outlier_pred = scores[scores >= candidate]
            ratio_normal = len(normal_pred) / len(scores)
            ratio_outlier = len(outlier_pred) / len(scores)
            variance_intern[i_c] = ratio_normal*np.var(normal_pred) + ratio_outlier*np.var(outlier_pred)
        self.thresh_ = candidates[np.argmin(variance_intern)]
        return self

class TwoSigma_Thresholder(BaseThresholder):
    """ Wrapping for the TwoSigma thresholding method. See Chapter 3.2.2.1

       Parameters
       ----------
    """
    def __init__(self):
        super().__init__()

    def eval(self, scores):
        self.thresh_ = np.mean(scores) + 2*np.std(scores)
        return self

class KSigma_Thresholder(BaseThresholder):
    """ Wrapping for the KSigma (Telemanom) thresholding method. See Chapter 3.2.2.2

       Parameters
       ----------
        z_list = np.array of the standard deviation factors to evaluate (standard = np.arange(1, 11))
    """
    def __init__(self, z_list=np.arange(1, 11)):
        super().__init__()
        self.z_list = z_list

    def eval(self, scores):
        avg = np.mean(scores)
        std = np.std(scores)
        thresholds = np.array([avg + z*std for z in self.z_list])

        def telemanom_func(score, threshold):
            score_inf_binary =  score < threshold
            score_sup_binary = np.ravel(np.array(score >= threshold, dtype=int))
            score_inf = score[score_inf_binary]
            avg_inf = np.mean(score_inf)
            std_inf = np.std(score_inf)
            score_sup = score[score_sup_binary]
            len_sup = len(score_sup)
            len_contiguous_sup = len(utils.get_events(score_sup_binary)[1])

            return ((avg-avg_inf) / avg + (std-std_inf) / std) / (len_sup + len_contiguous_sup**2)
        
        val_thresholds = np.array([telemanom_func(scores, threshold) for threshold in thresholds])
        self.thresh_ = thresholds[np.argmax(val_thresholds)]
        return self

class FittedDistribution_TwoSigma_Thresholder(BaseThresholder):
    """ Wrapping for the Fitted Distribution (TwoSigma) thresholding method. See Chapter 3.2.2.3

       Parameters
       ----------
        distributions = list of scipy.stats rv_continuous classes corresponding to the candidates distributions
            (default = [stats.halfnorm, stats.beta, stats.gamma, stats.logistic, stats.pareto])
    """
    def __init__(self, distributions=[stats.halfnorm, stats.beta, stats.gamma, stats.logistic, stats.pareto]):
        super().__init__()
        self.distributions = distributions

    def eval(self, scores):
        m = 0
        std = np.inf
        p_value = 0
        for distribution in self.distributions:
            parameters_distrib = distribution.fit(scores)
            cdf_distrib = distribution(*parameters_distrib).cdf
            p_value_distrib = stats.kstest(scores, cdf_distrib).pvalue
            if p_value_distrib >= p_value:
                p_value = p_value_distrib
                m, std = distribution(*parameters_distrib).stats()

        self.thresh_ = m + 2*std
        return self

class FittedDistribution_Quantile_Thresholder(BaseThresholder):
    """ Wrapping for the Fitted Distribution (Quantile) thresholding method. See Chapter 3.2.2.3

       Parameters
       ----------
        distributions = list of scipy.stats rv_continuous classes corresponding to the candidates distributions
            (default = [stats.halfnorm, stats.beta, stats.gamma, stats.logistic, stats.pareto])
        contamination : value in ]0,1[ that indicates the prevalence of anomalies in data (default = 0.001)
    """
    def __init__(self, distributions=[stats.halfnorm, stats.beta, stats.gamma, stats.logistic, stats.pareto], contamination=0.001):
        super().__init__()
        self.distributions = distributions
        self.contamination = contamination

    def eval(self, scores):
        threshold = 0
        p_value = 0
        for distribution in self.distributions:
            parameters_distrib = distribution.fit(scores)
            cdf_distrib = distribution(*parameters_distrib).cdf
            p_value_distrib = stats.kstest(scores, cdf_distrib).pvalue
            if p_value_distrib >= p_value:
                p_value = p_value_distrib
                threshold = distribution(*parameters_distrib).ppf(1-self.contamination)

        self.thresh_ = threshold
        return self

class POT_Thresholder(BaseThresholder):
    """ Wrapping for the POT thresholding method. See Chapter 3.2.3

       Parameters
       ----------
        contamination : value in ]0,1[ that indicates the prevalence of anomalies in data (default = 0.001)
        init_level : quantile used for extreme values threshold (default = 0.98)
        Ncandidates : number of candidates for the Grimshaw algorithm (default = 5)
        epsilon : numerical parameter for the Grimshaw algorithm (default = 1e-8)

    """
    def __init__(self, contamination=0.001, init_level=0.98, Ncandidates=5, epsilon=1e-8):
        super().__init__()
        self.contamination = contamination
        self.init_level = init_level
        self.Ncandidates = Ncandidates
        self.epsilon = epsilon

    def eval(self, scores):
        init_threshold = np.sort(scores)[int(self.init_level * len(scores))]

        peaks = scores[scores > init_threshold] - init_threshold

        gamma, sigma = utils.grimshaw(peaks, Ncandidates=self.Ncandidates, epsilon=self.epsilon)

        if gamma != 0:
            threshold = init_threshold + (sigma / gamma) * (pow(self.contamination * len(scores) / len(peaks), -gamma) - 1)
        else:
            threshold = init_threshold - sigma * np.log(self.contamination * len(scores) / len(peaks))

        self.thresh_ = threshold
        return self