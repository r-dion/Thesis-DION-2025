import numpy as np

def group_contiguous_anomalies(x, pos_label=1):
    """Constructs anomalous ranges from the discrete labels

    Parameters
    ----------
    x: np.array
        An array of labels
    pos_label : int
        Label value for the anomalous class

    Returns
    ----------
    anomalous_areas: list
        A list of tuples describing the start and end points of the anomalous ranges.
    """
    changes_index = np.concatenate([[0], np.where(np.diff(x) != 0)[0] + 1])
    anomalous_areas = list()
    for k_change, t_change in enumerate(changes_index):
        if x[t_change] == pos_label:
            if k_change < len(changes_index) - 1:
                anomalous_areas.append(
                    (
                        t_change,
                        changes_index[k_change + 1],
                    )
                )
            else:
                anomalous_areas.append(
                    (
                        t_change,
                        len(x),
                    )
                )
    return anomalous_areas

def sigmoid(x):
    """Standard sigmoid function."""
    return 1 / (1 + np.math.exp(-x))

def scaledSigmoid(relative_position_in_window, coef=-5):
    """Score function associated to each anomalous window"""
    if relative_position_in_window > 3.0:
        # FP well behind window
        val = -1.0
    else:
        val = 2 * sigmoid(coef * relative_position_in_window) - 1.0

    return val

def nab(y_true, y_pred, labels=None, pos_label=1, tp_weight=1, fp_weight=0.11, fn_weight=1, coef=-5):
    """Constructs anomalous ranges from the discrete labels

    Parameters
    ----------
    y_true : np.array
            An array of the ground truth binary labeling
    y_pred : np.array
        An array of the model predictions as binary output (inlier/outlier)
    pos_label : int
        The numerical value of an outlier label.
    labels : Not used, necessary for sklearn wrapping (default=None)
    tp_weight : float > 0, default=1
        weight associated to the TP score 
    fp_weight : float > 0, default=0.11
        weight associated to the FP score 
    fn_weight : float > 0, default=1
        weight associated to the FN score 
    coef : float < 0, default=-5
        coefficient for the score function
    Returns
    ----------
    scaled_nab: float in [0, 1]
        nab metric value, scaled against the worst possible prediction (only FP and FN)
    """
    nab_recomp = -np.ones(len(y_pred))

    events_labels = group_contiguous_anomalies(y_true)
    win_ano = 0.1*len(y_pred) / len(events_labels)

    events_label_ext = list()
    for i_e, event in enumerate(events_labels):
        events_label_ext.append([int(max(0, event[0] - win_ano // 2)), int(min(len(y_pred), event[1] + win_ano // 2))])

    for event in events_label_ext:
        len_ano = event[1] - event[0]
        for i in range(event[0],len(y_pred)):
            pos = -(event[1] - i + 1) / len_ano
            nab_recomp[i] = scaledSigmoid(pos, coef)
    
    counts_tp = np.zeros(len(events_label_ext))

    for i_e, event in enumerate(events_label_ext):
        for t in range(event[0], event[1]):
            if y_pred[t]:
                if nab_recomp[t] > counts_tp[i_e]:
                    counts_tp[i_e] = nab_recomp[t]

    counts_fp = np.zeros(len(y_pred))

    for i_e, event in enumerate(events_label_ext):
        if i_e == 0:
            for t in range(0, event[0]):
                if y_pred[t]:
                    counts_fp[t] = nab_recomp[i]
        else:
            for t in range(events_label_ext[i_e-1][1], event[0]):
                if y_pred[t]:
                    counts_fp[t] = nab_recomp[i]
        if i_e == len(events_label_ext) - 1:
            for t in range(event[1], len(y_pred)):
                if y_pred[t]:
                    counts_fp[t] = nab_recomp[t]


    counts_fn = np.array([0 if counts_tp[i] else 1 for i in range(len(events_label_ext))])

    raw_score = tp_weight*sum(counts_tp) + fp_weight*sum(counts_fp) - fn_weight*sum(counts_fn)


    best_possible_score = tp_weight*len(events_label_ext)
    worst_possible_score = -fn_weight*len(events_label_ext)
    for i_e, event in enumerate(events_label_ext):
        if i_e == 0:
            worst_possible_score += sum(nab_recomp[:event[0]])*fp_weight
        else:
            worst_possible_score += sum(nab_recomp[events_label_ext[i_e-1][1]:event[0]])*fp_weight
        if i_e == len(events_label_ext) - 1:
            worst_possible_score += sum(nab_recomp[event[1]:])*fp_weight

    scaled_nab = (raw_score - worst_possible_score) / (best_possible_score - worst_possible_score)
    return scaled_nab