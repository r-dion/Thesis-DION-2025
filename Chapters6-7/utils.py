import numpy as np

meta_columns = np.array(['timestamp', 'item_id', 'label', 'context_id', ' ', ''])

def windowing(X, window_size=100, sliding=False):
    if sliding:
        n_windows = X.shape[0] - window_size
        Xwin = np.zeros((X.shape[1], n_windows, window_size))
        for i_col in range(X.shape[1]):
            for i_window in range(n_windows):
                Xwin[i_col, i_window, :] = np.ravel(X[i_window:i_window+window_size, i_col])
    else:
        if X.shape[0] % window_size:
            n_windows = int(X.shape[0] // window_size + 1)
        else:
            n_windows = int(X.shape[0] // window_size)
        Xwin = np.zeros((X.shape[1], n_windows, window_size))
        for i_col in range(X.shape[1]):
            for i_window in range(n_windows):
                if i_window < n_windows-1:
                    Xwin[i_col, i_window, :] = np.ravel(X[window_size*i_window:window_size*(i_window+1), i_col])
                else:
                    Xwin[i_col, i_window, :] = np.ravel(X[-window_size:, i_col])
    return Xwin

def safe_divide(a, b, constant=0):
    if (b):
        return a / b
    return np.ones(a.shape)*constant

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


def aggregate_score(res_test, res_train):
    n_sensors = res_train.shape[1] // 3
    scaled_res = np.zeros((res_test.shape[0], 3*n_sensors))
    for i_sensor in range(n_sensors):
        i_bound_train = np.ravel(res_train[:, i_sensor*3])
        i_corr_train = np.ravel(res_train[:, i_sensor*3+1])
        i_corr_th = np.quantile(i_corr_train, 0.99)

        i_bound_test = np.ravel(res_test[:, i_sensor*3])
        i_corr_test = np.ravel(res_test[:, i_sensor*3+1])
        i_nunexpected_test = np.ravel(res_test[:, i_sensor*3+2])
        
        scaled_res[:, i_sensor*3] = i_bound_test
        scaled_res[:, i_sensor*3+1] = i_corr_test / i_corr_th
        scaled_res[:, i_sensor*3+2] = i_nunexpected_test
        scaled_res = np.abs(scaled_res)
    score_new = scaled_res.max(axis=1)
    return score_new