import utils
import numpy as np

def overlap(S, T, depth):
    """Estimate overlap between 2 lists

        Parameters
        ----------
        S : list
        T : list
        depth : int
            overlap depth to estimate
    

        Returns
        ----------
        depth_overlap : list
            list of overlapping elements
    """
    Sd = utils.flatten(S[:depth])
    Td = utils.flatten(T[:depth])
    depth_overlap = [s for s in Sd if s in Td]
    return depth_overlap

def len_overlap(S, T, depth):
    """Len of overlap between two lists at specified depth"""
    return len(overlap(S, T, depth))

def agreement(S, T, depth):
    """Estimates agreement between two lists at specified depth"""
    return 2*len_overlap(S, T, depth) / (len(utils.flatten(S[:depth])) + len(utils.flatten(T[:depth])))

def rbo(S, T, p):
    """Estimate RBO between two rankings. See William Webber, Alistair Moffat, and Justin Zobel. 2010. 
        A similarity measure for indefinite rankings. ACM Trans. Inf. Syst. 28, 4, Article 20 (November 2010), 38 pages.
        https://doi.org/10.1145/1852102.1852106
        
        Parameters
        ----------
        S : list
        T : list
        p : float in ]0,1]
            cursor that indicates the ranking head importance (p close to 0 indicates that the ranking head is essential)
    

        Returns
        ----------
        float in [0, 1]
            RBO-similarity between two rankings
    """
    max_depth = max(len(S), len(T))
    rbo_list = np.zeros(max_depth)
    for depth in range(1, max_depth+1):
        rbo_list[depth-1] = (p**(depth-1))*agreement(S, T, depth)
    if p == 1:
        return np.mean(rbo_list)
    else:
        return (1-p) * np.sum(rbo_list)
    
