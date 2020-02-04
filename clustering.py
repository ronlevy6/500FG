import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt

def dist(v1, v2, min_vals_pct=0.4, penalty=10):
    """
    calculate euclidean distance between vectors containing NaNs based on shared values only
    :params v1, v2: vectors to calculate distance between
    :param min_vals_pct: minimal pct of shared values between vectors
    :param penalty: distance in case not enough shared non-NaN values
    :return: euclidean distance
    """
    vs = np.array([v1, v2])
    try:
        mask = np.isnan(vs[0]) | np.isnan(vs[1])
    except:
        mask = pd.isna(vs[0]) | pd.isna(vs[1])
    finally:
        filtered = vs[:,np.invert(mask)]
        if len(filtered[0]) < len(v1)*min_vals_pct:
            return penalty
        return np.sqrt(sum((filtered[0]-filtered[1])**2))


def get_cluster_from_clustermap_result(clustermap_res, is_row, labels,
                                       color_thresh_const=0.6, leaf_rotation=90, figsize=(15, 6)):
    """
    prints clustering based on clustermap results
    """
    plt.figure(figsize=figsize)
    if is_row:
        Z = clustermap_res.dendrogram_row.linkage
    else:
        Z = clustermap_res.dendrogram_col.linkage
    dn = hierarchy.dendrogram(Z, labels=labels,
                              color_threshold=color_thresh_const*max(Z[:,2]), leaf_rotation=leaf_rotation)
    plt.show()
    return dn
