import os
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import seaborn as sns
from utils import by_idx

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


def clustermap_with_color(states_df, idx_to_plot, patient_data, color_col, title=None, min_vals_pct=0.4,
                          palette='bwr', to_save=None, col_cluster=True):
    tissues_data = list(states_df.index)
    if len(tissues_data) == 1:
        col_cluster = False
    df_to_clustermap = states_df.transpose().merge(patient_data, left_index=True, right_index=True)
    color_data = df_to_clustermap[color_col]

    network_pal = sns.color_palette(palette, len(set(color_data)))
    network_lut = dict(zip(set(color_data.values), network_pal))
    network_colors = pd.Series(color_data).map(network_lut)
    if len(set(network_colors.values)) == 1:
        network_colors = None
    g = sns.clustermap(df_to_clustermap[tissues_data].applymap(lambda x: by_idx(x, idx_to_plot)),
                       metric=lambda u, v: dist(u, v, min_vals_pct), col_cluster=col_cluster,
                       figsize=(15, 10), method='complete', row_colors=network_colors)
    if title is not None:
        g.fig.suptitle(title, fontsize=20)
    for label in sorted(color_data.unique()):
        g.ax_col_dendrogram.bar(0, 0, color=network_lut[label], label=label, linewidth=0)
        g.ax_col_dendrogram.legend(loc="upper left", ncol=6, bbox_to_anchor=(-0.2, 0.75, 0.5, 0.5))

    if to_save is not None:
        plt.tight_layout()
        dir_to_save = os.path.join(to_save, title)
        g.savefig(dir_to_save+'.jpg')
        g.savefig(dir_to_save + '.pdf')
    plt.show()
    return g


def get_cluster_from_clustermap_result(clustermap_res, is_row, title,
                                       color_thresh_const=0.6, leaf_rotation=90, figsize=(15, 6), to_save=None):
    """
    prints clustering based on clustermap results
    :param to_save: if not None-directory in which to save figure
    """
    plt.figure(figsize=figsize)
    plt.title(title, fontdict={'fontsize':20})
    if is_row:
        Z = clustermap_res.dendrogram_row.linkage
        labels = clustermap_res.data.index
    else:
        Z = clustermap_res.dendrogram_col.linkage
        labels = clustermap_res.data.columns
    dn = hierarchy.dendrogram(Z, labels=labels,
                              color_threshold=color_thresh_const*max(Z[:,2]), leaf_rotation=leaf_rotation)
    if to_save is not None:
        plt.tight_layout()
        dir_to_save = os.path.join(to_save, title)
        plt.savefig(dir_to_save+'.jpg')
        plt.savefig(dir_to_save + '.pdf')
    plt.show()
    return dn
