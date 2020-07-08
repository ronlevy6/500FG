import os
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import seaborn as sns
from utils import by_idx
from plotting_utils import plot_missing_data
from misc_utils import save_and_show_figure


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
                          palette='bwr', to_save=None, col_cluster=True, to_show=True, save_pdf=False):
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

    if to_save is not None or to_show:
        dir_to_save = os.path.join(to_save, title)
        save_and_show_figure(dir_to_save, save_pdf=save_pdf, to_show=to_show, tight_layout=True, bbox_inches=None)
    return g


def get_cluster_from_clustermap_result(clustermap_res, is_row, title, color_thresh_const=0.6, leaf_rotation=90,
                                       figsize=(15, 6), to_save=None, to_show=True, save_pdf=False):
    """
    prints clustering based on clustermap results
    """
    plt.figure(figsize=figsize)
    plt.title(title, fontdict={'fontsize' : 20})
    if is_row:
        Z = clustermap_res.dendrogram_row.linkage
        labels = clustermap_res.data.index
    else:
        Z = clustermap_res.dendrogram_col.linkage
        labels = clustermap_res.data.columns
    dn = hierarchy.dendrogram(Z, labels=labels,
                              color_threshold=color_thresh_const*max(Z[:,2]), leaf_rotation=leaf_rotation)
    if to_save is not None or to_show:
        dir_to_save = os.path.join(to_save, title) if to_save is not None else None
        save_and_show_figure(dir_to_save, save_pdf=save_pdf, to_show=to_show, tight_layout=True, bbox_inches=None)
    return dn


def tissue_clustering(curr_data, main_title, patient_data, extract_cluster=False, hue_col='sex_name',
                      min_vals_pct=0.0, is_row=False, missing_data_figsize=(12, 8),
                      to_save=None, to_show=True, save_pdf=False):

    # plot missing data visualization
    plot_missing_data(curr_data, main_title, figsize=missing_data_figsize, to_save=to_save, to_show=to_show)

    s1_data = clustermap_with_color(curr_data, 0, patient_data, hue_col, min_vals_pct=min_vals_pct,
                                    title='s1 clustering - {}'.format(main_title), to_save=to_save, to_show=to_show,
                                    save_pdf=save_pdf)

    s2_data = clustermap_with_color(curr_data, 1, patient_data, hue_col, min_vals_pct=min_vals_pct,
                                    title='s2 clustering - {}'.format(main_title), to_save=to_save, to_show=to_show,
                                    save_pdf=save_pdf)
    if extract_cluster:
        s1_cluster = get_cluster_from_clustermap_result(s1_data, is_row=is_row,
                                                        title='s1 tissue clustering - {}'.format(main_title),
                                                        to_save=to_save, to_show=to_show, save_pdf=save_pdf)

        s2_cluster = get_cluster_from_clustermap_result(s2_data, is_row=is_row,
                                                        title='s2 tissue clustering - {}'.format(main_title),
                                                        to_save=to_save, to_show=to_show, save_pdf=save_pdf)
        return s1_data, s1_cluster, s2_data, s2_cluster
    return s1_data, s2_data


def corr_and_cluster_states(states_df, state_idx, main_title, tissues_del_thresh=5, extract_cluster=False, to_plot=True,
                            to_save=None, to_show=True, save_pdf=True, tight_layout=True, bbox_inches='tight'):
    curr_state_df = states_df.applymap(lambda val: by_idx(val, state_idx))
    corr_df = curr_state_df.transpose().corr()
    corr_df = corr_df[sorted(list(corr_df.columns))].sort_index()

    # remove all nans tissues
    to_del = corr_df[corr_df.isna().sum() == corr_df.shape[0]].index
    corr_df = corr_df.drop(columns=to_del).drop(index=to_del)

    curr_title = 'Correlation between tissues - {} s{}'.format(main_title, state_idx + 1)
    while corr_df.isna().sum().sum() and tissues_del_thresh > 0:
        tissues_del_thresh -= 1
        to_del = set(corr_df[corr_df.isna().sum() > tissues_del_thresh].index)
        corr_df = corr_df.drop(columns=to_del).drop(index=to_del)
    if to_plot:
        g = sns.clustermap(corr_df, cmap='bwr', vmin=-1, vmax=1)
        plt.title(curr_title)

        if to_save is not None or to_show:
            dir_to_save = os.path.join(to_save, curr_title) if to_save is not None else None
            save_and_show_figure(dir_to_save, save_pdf=save_pdf, to_show=to_show,
                                 tight_layout=tight_layout, bbox_inches=bbox_inches)
        if extract_cluster:
            corr_cluster = get_cluster_from_clustermap_result(g, is_row=True, title=curr_title + ' clusters',
                                                              to_save=to_save, to_show=to_show, save_pdf=save_pdf)
        else:
            corr_cluster = None
    else:
        g = None
        corr_cluster = None
    return corr_df, g, corr_cluster
