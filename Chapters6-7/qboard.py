import copy
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

metadata_column_names = ['timestamp', 'timestamps', 'context_id', 'label', 'index', 'item_id', 'anomaly_class']


class QBoard(BaseEstimator):
    """ Class for QBoard anomaly detection algorithm. See Chapter 6

    Parameters 
    -----------
    n_quantiles : int (default=10)
        Number of uniformly spaced quantiles to use for quantization step
    transition_length : int (default=10)
        Length of a transition window for the creation of transition pairs
    correlation_threshold : float in ]0, 1] (default=0.95)
        Correlation threshold for configuration vectors space reduction
    bound_quantiles : list (default=[0.001,0.999])
        Lower and Upper bounds (as quantiles) for the variation domain normaly parameters
    bound_threshold : float (default=1.0)
        Tolerance for variation domain deviation residuals. See epsilon in (6.27)
    nu : float (default=1.0)
        Power used for domain deviation distance estimation. DEPRECATED
    n_clusters : int > 0 (default=20)
        Number of clusters for KMeans 
    window_size = int > 0 (default=100)
        Window size
    rolling_step #TODO
    drifting_sensors : list of strings (default=[])
        Drifting sensor to ignore
    bound_method : string (default='quantile')
        Quantile estimation method. Currently available methods are 'quantile' and 'probabilistic' 
    bound_params = list (default=[0.01, 0.05])
        Parameters for probabilistic quantile estimation

    
    Attributes
    -----------
    scaler : fitted scaler 
    params : all normality parameters, detailed for each transition.
    
    """
    def __init__(self, n_quantiles=10, transition_length=10, correlation_threshold = 0.95, bound_quantiles = [0.001,0.999], bound_threshold = 1.0, 
                 nu = 1, n_clusters = 20, window_size = 100, rolling_step = None, drifting_sensors = [], 
                 bound_method = 'quantile', bound_params = [0.01,0.05]) -> None:
        super().__init__()
        self.n_quantiles = n_quantiles
        self.transition_length = transition_length
        self.correlation_threshold = correlation_threshold
        self.bound_quantiles = bound_quantiles
        self.bound_threshold = bound_threshold
        self.nu = nu
        self.n_clusters = n_clusters
        self.window_size = window_size
        self.rolling_step = rolling_step
        self.drifting_sensors = drifting_sensors
        self.scaler = copy.deepcopy(StandardScaler())
        self.params = dict()
        self.columns_data = list()
        self.bound_method = bound_method
        self.bound_params = bound_params

    def preprocess_data(self, X, fit=False):
        """Separate sensors' data from metadata"""
        columns_all = list(X.columns)
        if fit:
            columns_data = [col for col in columns_all if col not in metadata_column_names]
            self.columns_data = columns_data
        columns_metadata = list(set(columns_all) - set(self.columns_data))
        if len(columns_metadata):
            self.X_metadata = X[columns_metadata]
        return X[self.columns_data]

    def construct_bins(self, Xcol, sensor_name):
        """Quantization"""
        bins = pd.qcut(Xcol, q=self.n_quantiles, retbins=True, labels=False,duplicates='drop')[-1]
        bins[0], bins[-1] = -np.inf, np.inf
        self.params[sensor_name]['bins'] = bins
    
    def construct_transitions(self, Xcol, sensor_name):
        """Transitions construction"""
        sensor_bins = pd.cut(Xcol, self.params[sensor_name]['bins'], labels=False)
        sensor_transitions = [f"({sensor_bins[i]},{sensor_bins[i+self.transition_length]})" for i in range(Xcol.size-self.transition_length)]
        return sensor_transitions
    
    def construct_context(self, Xcol, X, trans_idx):
        """Configuration vectors construction"""
        if self.transition_length:
            context = np.zeros((trans_idx.size, X.shape[1]-1+self.transition_length))
            context[:, :X.shape[1]] = X[trans_idx-self.transition_length, :]
            context[:, X.shape[1]:] = [Xcol[trans_i-self.transition_length+1:trans_i] for trans_i in trans_idx]
        else:
           context = X[trans_idx, :] 
        return context

    def get_context_bounds(self, context, sensor_name, transition):
        """Variation Domain bounds"""
        if self.bound_method == 'quantile':
            self.params[sensor_name][transition]['lower_bounds'] = np.quantile(context, axis=0, q=self.bound_quantiles[0])
            self.params[sensor_name][transition]['upper_bounds'] = np.quantile(context, axis=0, q=self.bound_quantiles[1])
        elif self.bound_method == 'probabilistic':
            n_samples = min((7.47/self.bound_params[0])*np.log(1/self.bound_params[1]), context.shape[0])
            g = np.random.Generator(np.random.PCG64())
            sampled_contexts = g.choice(context, n_samples, replace=False)
            self.params[sensor_name][transition]['lower_bounds'] = sampled_contexts.min(axis=0)
            self.params[sensor_name][transition]['upper_bounds'] = sampled_contexts.max(axis=0)

        
    def remove_correlated(self, context):
        C = np.corrcoef(context)
        r_list = [i for i in range(len(context))]
        k_list = []
        while len(r_list)>0:
            ir = r_list[0]
            r_list.remove(ir)
            k_list += [ir]
            ind = [j for j in r_list if j>ir and abs(C[ir, j])>self.correlation_threshold]
            for i in ind:
                r_list.remove(i)

        context_lowcorr = context[k_list]
        return context_lowcorr

    def trim_contexts(self, context):
        """Configuration vectors space reduction"""
        # Correlation part
        if context.shape[0] > self.n_clusters:
            clustering = copy.deepcopy(KMeans(n_clusters=self.n_clusters, n_init='auto', random_state=42))
            clustering.fit(context)
            context = clustering.cluster_centers_

        context_lowcorr = self.remove_correlated(context)
        return context_lowcorr

    def compute_bound_residual(self, context, sensor_name, transition):
        if sensor_name in self.drifting_sensors:
            return 0
        else:
            e_min = self.params[sensor_name][transition]['lower_bounds'] - context
            e_max = context - self.params[sensor_name][transition]['upper_bounds']
            e_min = np.power(e_min[e_min > self.bound_threshold].sum()/len(context), self.nu)
            e_max = np.power(e_max[e_max > self.bound_threshold].sum()/len(context), self.nu)
            e_bound = (e_min + e_max)/2
            return e_bound
    
    def compute_correlation_residual(self, context, sensor_name, transition):
        M1 = self.params[sensor_name][transition]['context']
        M2 = context
        n1, n2 = len(M1), len(M2)
        M = np.vstack([M1, M2])
        res = abs(np.corrcoef(M)[0:n1,n1:]).max(axis=0).mean()
        tmp = abs(np.corrcoef(M)[0:n1,n1:]).max(axis=0)
        return res

    def fit(self, X_raw, fit_for_diagnosis = False):
        X_data = self.preprocess_data(X_raw, fit=True)
        X = self.scaler.fit_transform(X_data)
        for i_sensor, sensor_name in enumerate(self.columns_data):
            self.params[sensor_name] = dict()
            Xcol = X[:, i_sensor]
            self.construct_bins(Xcol, sensor_name)
            sensor_transitions = self.construct_transitions(Xcol, sensor_name)
            self.params[sensor_name]['transitions'] = list(set(sensor_transitions))
            for transition in self.params[sensor_name]['transitions']:
                self.params[sensor_name][transition] = dict()
                mask = np.concatenate([[False]*self.transition_length, np.array(sensor_transitions) == transition])
                trans_idx = np.arange(Xcol.size)[mask]
                context = self.construct_context(Xcol, X, trans_idx)
                self.get_context_bounds(context, sensor_name, transition)
                self.params[sensor_name][transition]['context'] = self.trim_contexts(context)

        if fit_for_diagnosis:
            res_fit = self.predict(X_raw, False)
            self.residual_scaler = dict()
            for i_sensor, sensor_name in enumerate(self.columns_data):
                self.residual_scaler[sensor_name] = copy.deepcopy(StandardScaler())
                self.residual_scaler[sensor_name].fit(res_fit[:, i_sensor*3:i_sensor*3+2])
    
    def predict(self, X_raw):
        X_data = self.preprocess_data(X_raw)
        X = self.scaler.transform(X_data)
        if X.shape[0] % self.window_size:
            n_windows = X.shape[0] // self.window_size + 1
        else:
            n_windows = X.shape[0] // self.window_size
        residuals = dict()
        for i_sensor, sensor_name in enumerate(self.columns_data):
            residuals_sensor = np.zeros((n_windows, 3))
            for j_win in range(n_windows):
                window_bounds_residual = 0
                window_correlation_residual = np.inf
                if (j_win+1)*self.window_size > X.shape[0]:
                    X_loc = X[-self.window_size:, :]
                    Xcol_loc = X_loc[:, i_sensor]
                else:
                    X_loc = X[j_win*self.window_size:(j_win+1)*self.window_size, :]
                    Xcol_loc = X_loc[:, i_sensor]
                sensor__all_transitions = self.construct_transitions(Xcol_loc, sensor_name)
                new_trans = list(set(sensor__all_transitions) - set(self.params[sensor_name]['transitions']))
                new_trans_full = [trans for trans in sensor__all_transitions if trans in new_trans]
                sensor_transitions = list(set(sensor__all_transitions) - set(new_trans))
                window_transition_residual = len(new_trans_full)
                for transition in sensor_transitions:
                    mask = np.concatenate([[False]*self.transition_length, np.array(sensor__all_transitions) == transition])
                    trans_idx = np.arange(Xcol_loc.size)[mask]
                    context = self.construct_context(Xcol_loc, X_loc, trans_idx)
                    window_bounds_residual += self.compute_bound_residual(context, sensor_name, transition)
                    window_correlation_residual = min((self.compute_correlation_residual(context, sensor_name, transition), window_correlation_residual))
                if len(sensor_transitions):
                    window_bounds_residual = window_bounds_residual / len(sensor_transitions)
                    window_correlation_residual = 1 - window_correlation_residual
                else:
                    window_bounds_residual = 0
                    window_correlation_residual = 0
                if j_win < n_windows:
                    residuals_sensor[j_win, 0] = window_bounds_residual
                    residuals_sensor[j_win, 1] = window_correlation_residual
                    residuals_sensor[j_win, 2] = window_transition_residual
            residuals[sensor_name] = residuals_sensor
        res = np.zeros((X.shape[0], X.shape[1]*3))
        for i_sensor, sensor_name in enumerate(self.columns_data):
            res[:(n_windows-1)*self.window_size, i_sensor*3] = np.repeat(residuals[sensor_name][:(n_windows-1), 0], self.window_size)
            res[:(n_windows-1)*self.window_size, i_sensor*3+1] = np.repeat(residuals[sensor_name][:n_windows-1, 1], self.window_size)
            res[:(n_windows-1)*self.window_size, i_sensor*3+2] = np.repeat(residuals[sensor_name][:n_windows-1, 2], self.window_size)

            res[(n_windows-1)*self.window_size:, i_sensor*3] = np.repeat(residuals[sensor_name][-1, 0], X.shape[0] - (n_windows-1)*self.window_size)
            res[(n_windows-1)*self.window_size:, i_sensor*3+1] = np.repeat(residuals[sensor_name][-1, 1], X.shape[0] - (n_windows-1)*self.window_size)
            res[(n_windows-1)*self.window_size:, i_sensor*3+2] = np.repeat(residuals[sensor_name][-1, 2], X.shape[0] - (n_windows-1)*self.window_size)
        return res
    
    def partial_fit(self, X_raw, new_clustering=False):
        """Incremental Learning for Normality Space Augmentation. See Chapter 6.4.1"""
        X_data = self.preprocess_data(X_raw)
        X = self.scaler.transform(X_data)
        for i_sensor, sensor_name in enumerate(self.columns_data):
            Xcol = X[:, i_sensor]
            sensor_all_transitions = self.construct_transitions(Xcol, sensor_name)
            sensor_transitions = list(set(sensor_all_transitions))
            for transition in sensor_transitions:
                if transition in self.params[sensor_name]['transitions']:
                    mask = np.concatenate([[False]*self.transition_length, np.array(sensor_all_transitions) == transition])
                    trans_idx = np.arange(Xcol.size)[mask]
                    context = self.construct_context(Xcol, X, trans_idx)
                    concatenated_contexts = np.concatenate([self.params[sensor_name][transition]['context'], 
                                                                                    context])
                    if new_clustering:
                        self.params[sensor_name][transition]['context'] = self.trim_contexts(concatenated_contexts)
                    else:
                        self.params[sensor_name][transition]['context'] = self.remove_correlated(concatenated_contexts)
                    # self.params[sensor_name][transition]['context']  = concatenated_contexts
                    self.params[sensor_name][transition]['lower_bounds'] = np.min(np.concatenate(
                        [self.params[sensor_name][transition]['lower_bounds'].reshape(1, -1), 
                            context]), axis=0)
                    self.params[sensor_name][transition]['upper_bounds'] = np.max(np.concatenate(
                        [self.params[sensor_name][transition]['upper_bounds'].reshape(1, -1), 
                            context]), axis=0)
                else:
                    mask = np.concatenate([[False]*self.transition_length, np.array(sensor_all_transitions) == transition])
                    trans_idx = np.arange(Xcol.size)[mask]
                    context = self.construct_context(Xcol, X, trans_idx)
                    self.params[sensor_name][transition] = dict()
                    self.params[sensor_name][transition]['context'] = context
                    self.params[sensor_name]['transitions'] = np.concatenate([self.params[sensor_name]['transitions'], [transition]])

                    self.params[sensor_name][transition]['lower_bounds'] = np.min(context, axis=0)
                    self.params[sensor_name][transition]['upper_bounds'] = np.max(context, axis=0)

    def construct_score(self, res, return_aggregated = True):
        sensors = self.columns_data
        scaled_res = np.zeros((res.shape[0], 3*len(sensors)))
        for i_sensor, sensor_name in enumerate(sensors):
            scaled_res[:, i_sensor*3:i_sensor*3+2] = self.residual_scaler[sensor_name].transform(res[:, i_sensor*3:i_sensor*3+2])
            scaled_res[:, i_sensor*3+2] = np.ravel(res[:, i_sensor*3+2]*100)
            scaled_res = np.abs(scaled_res)
        if return_aggregated:
            score = scaled_res.max(axis=1)
            return score
        else:
            return scaled_res

    def partial_defit(self, X_raw, max_correlation):
        """Incremental Learning for Normality Space Reduction. See Chapter 6.4.2"""
        X_data = self.preprocess_data(X_raw)
        X = self.scaler.transform(X_data)
        for i_sensor, sensor_name in enumerate(self.columns_data):
            Xcol = X[:, i_sensor]
            sensor_all_transitions = self.construct_transitions(Xcol, sensor_name)
            sensor_transitions = list(set(sensor_all_transitions))
            for transition in sensor_transitions:
                if transition in self.params[sensor_name]['transitions']:
                    mask = np.concatenate([[False]*self.transition_length, np.array(sensor_all_transitions) == transition])
                    trans_idx = np.arange(Xcol.size)[mask]
                    context = self.construct_context(Xcol, X, trans_idx)
                    n1, n2 = len(self.params[sensor_name][transition]['context']), len(context)
                    M = np.vstack([self.params[sensor_name][transition]['context'], context])
                    correlated_context = np.arange(0, n1)[abs(np.corrcoef(M)[0:n1,n1:]).max(axis=1) > max_correlation]
                    context_to_keep = [i for i in np.arange(0, n1) if i not in correlated_context]
                    if len(context_to_keep) == 0:
                        print(sensor_name, transition)
                        self.params[sensor_name]['transitions'] = [trans for trans in self.params[sensor_name]['transitions'] if trans != transition]
                    else:
                        tmp = np.zeros((len(context_to_keep), self.params[sensor_name][transition]['context'].shape[1]))
                        for i, i_context in enumerate(context_to_keep):
                            tmp[i, :] = self.params[sensor_name][transition]['context'][i_context, :]
                        self.params[sensor_name][transition]['context'] = tmp
                    self.params[sensor_name][transition]['lower_bounds'] = np.min(self.params[sensor_name][transition]['context'], axis=0)
                    self.params[sensor_name][transition]['upper_bounds'] = np.max(self.params[sensor_name][transition]['context'], axis=0)


    def partial_defit_diagnosis(self, X_raw, X_diag_raw):
        X_data = self.preprocess_data(X_raw)
        X = self.scaler.transform(X_data)
        X_diag_data = self.preprocess_data(X_diag_raw)
        X_diag = self.scaler.transform(X_diag_data)
        transition_diagnosis = dict()
        for i_sensor, sensor_name in enumerate(self.columns_data):
            transition_diagnosis[sensor_name] = dict()
            Xcol = X[:, i_sensor]
            sensor_all_transitions = self.construct_transitions(Xcol, sensor_name)
            sensor_transitions = list(set(sensor_all_transitions))
            Xdiag_col = X_diag[:, i_sensor]
            sensor_diag_all_transitions = self.construct_transitions(Xdiag_col, sensor_name)
            for transition in sensor_transitions:
                transition_diagnosis[sensor_name][transition] = [1 if trans == str(transition) else 0 for trans in sensor_diag_all_transitions]
        return transition_diagnosis
    
    def forget_transition(self, sensor_name, transition):
        """Incremental Learning for Normality Space Reduction when the defit method does not output satisfactory results.
        See SKAB example in Chapter 6.4.2.4"""
        self.params[sensor_name]['transitions'] = [trans for trans in self.params[sensor_name]['transitions'] if trans != transition]