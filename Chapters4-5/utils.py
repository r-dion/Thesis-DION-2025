import numpy as np

def sigmoid(x):
    """Standard sigmoid function."""
    return 1 / (1 + np.math.exp(-x))

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

def nested_depth(l):
    """Estimated depth of a list. Used for flattening lists

        Parameters
        ----------
        l : list
    

        Returns
        ----------
        int
            depth of list
    """
    if isinstance(l, list):
        return 1 + max(nested_depth(item) for item in l)
    else:
        return 0

def flatten(l):
    """Flatten a list

        Parameters
        ----------
        l : list
    

        Returns
        ----------
        list
            flattened list
    """
    if nested_depth(l) > 1:
        return [x for xs in l for x in xs]
    return l

def rank_with_scores(names, scores, round_prec = 3):
    """Rank in descending order predictions using their associated scores (given by a metric). Ties are possible.

        Parameters
        ----------
        names : list
            list of predictions names
        scores : list or np.array
            list of associated scores
        round_prec : int (default=3)
            rounding precision for ties detection
    
        Returns
        ----------
        ranking
            Decreasing ranking of predictions names
    """
    rounded_score = np.round(scores, round_prec)
    unique_values, counts = np.unique(rounded_score, return_counts=True)
    tied_values = unique_values[np.where(counts > 1)[0]]
    
    if len(tied_values):
        tied_tuples = dict()
        for val in tied_values:
            tied_tuples[val] = [i for i in range(len(rounded_score)) if rounded_score[i] == val]
        unique_index = flatten([i for i in range(len(rounded_score)) if rounded_score[i] not in tied_values] + [tied_tuples[val][0] for val in tied_values])
        ranking = np.flip([[names] for _, names in sorted(zip(rounded_score[unique_index], np.array(names)[unique_index]))])
        ranking = ranking.tolist()
        count_nested = 0
        for val in tied_values:
            if not count_nested:
                ranking[np.where(np.array(ranking) == names[tied_tuples[val][0]])[0][0]] = np.array(np.array(names)[tied_tuples[val]]).tolist()
                count_nested += 1
            else:
                for k_rank, rank in enumerate(ranking):
                    if rank == [names[tied_tuples[val][0]]]:
                        id_tie = k_rank
                ranking[id_tie] = np.array(np.array(names)[tied_tuples[val]]).tolist()
        return ranking
    else:
        return np.flip([[names] for _, names in sorted(zip(scores, names))])