import numpy as np
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError


class BaseThresholder(BaseEstimator):
    """Abstract class for all outlier detection thresholding algorithms.

       Parameters
       ----------

       Attributes
       ----------

       thresh_ : threshold value that separates inliers from outliers

       is_fitted : binary value that indicates if the thresholder is fitted
    """

    def __init__(self):
        super().__init__()
        self.thresh_ = None
        self._is_fitted = False

    def eval(self, scores):
        """Outlier/inlier evaluation process for decision scores.

        Parameters
        ----------
        scores : np.array or list of shape (n_samples)
                   or np.array of shape (n_samples, n_detectors)
                   which are the decision scores from a
                   outlier detection.

        Returns
        -------
        outlier_labels : numpy array of shape (n_samples,)
            For each observation, tells whether or not
            it should be considered as an outlier according to the
            fitted model. 0 stands for inliers and 1 for outliers.
        """
        return self


    def fit(self, X, y=None):
        """Outlier/inlier fit process for decision scores.
        Parameters
        ----------
        X : np.array or list of shape (n_samples)
                   or np.array of shape (n_samples, n_detectors)
                   which are the decision scores from a
                   outlier detection.
        """
        self.eval(X)
        self._is_fitted = True

        return self
    

    def predict(self, X):
        """Outlier/inlier predict process for decision scores.

        Parameters
        ----------
        X : np.array or list of shape (n_samples)
                   or np.array of shape (n_samples, n_detectors)
                   which are the decision scores from a
                   outlier detection.

        Returns
        -------
        predictions : numpy array of shape (n_samples,)
            For each observation, tells whether or not
            it should be considered as an outlier according to the
            fitted model. 0 stands for inliers and 1 for outliers.
        """

        if self._is_fitted:
            predictions = np.zeros(len(X))
            predictions[X >= self.thresh_] = 1
            return predictions
        else:
            raise NotFittedError

    def fit_predict(self, X, y=None):
        """Outlier/inlier fit and predict process for decision scores.

            Parameters
            ----------
            X : np.array or list of shape (n_samples)
                    or np.array of shape (n_samples, n_detectors)
                    which are the decision scores from a
                    outlier detection.

            Returns
            -------
            predictions : numpy array of shape (n_samples,)
                For each observation, tells whether or not
                it should be considered as an outlier according to the
                fitted model. 0 stands for inliers and 1 for outliers.
            """
        
        self.fit(X, y)
        predictions = self.predict(X)
        return predictions