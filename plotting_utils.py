import math
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import itertools
from data_manipulation import filter_patient_df
from misc_utils import undo_print_pretty, fix_text, print_pretty, save_and_show_figure, filter_df, get_vmin_vmax
from utils import by_idx
from regression_analysis import get_bits_dist


def plot_df(df, x, y, to_filter=False, index_filter_vals=None, title=None, to_color=False, colors=None):
    """
    plots scatter of dataframe with optional coloring
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel(x, fontsize=15)
    ax.set_ylabel(y, fontsize=15)
    if title is not None:
        ax.set_title(title, fontsize=20)
    if colors is None:
        colors = ['r', 'g',]
    if to_filter:
        df = df[df.index.isin(index_filter_vals)]
    if to_color:
        ax.scatter(df[x], df[y], c=colors, s=50)
    else:
        ax.scatter(df[x], df[y], s=50)
    ax.grid()
    plt.show()


def plot_df_color_by_col_vals(df, x, y, col, title=None, is_grid=True, pallete='jet'):
    print(col)
    ax = df.plot.scatter(x=x, y=y, c=col, s=50, cmap=pallete, figsize=(10 ,8))
    ax.set_xlabel(x, fontsize=15)
    ax.set_ylabel(y, fontsize=15)
    if title:
        ax.set_title('{} colored by {}'.format(title, col), fontsize=20)
    if is_grid:
        ax.grid()
    plt.show()


def smart_print(is_mean, is_std, is_max, is_builtin, to_filter, evr):
    to_print = ''
    if is_mean:
        to_print += 'mean '
    if is_std:
        to_print += 'std '
    if is_max:
        to_print += 'max '
    if is_builtin:
        to_print += 'builtin '
    if to_filter:
        to_print += 'filtered '
    if to_print == '':
        to_print = 'original'
    to_print = '{} {} {}'.format(to_print, evr, sum(evr))
    print(to_print)


def calc_scatter_x_y_lims(input_data, patient_data, patients_to_use=None):
    s1_max = None
    s2_max = None
    s1_min = None
    s2_min = None
    for name, states_df, filtering_dict in input_data:
        df_to_use = filter_patient_df(patient_data, filtering_dict)
        if patients_to_use is not None:
            states_df = states_df[set(patients_to_use) & set(states_df.columns)]
        df = states_df[set(df_to_use.index) & set(states_df.columns)]
        s1 = df.applymap(lambda val: by_idx(val, 0))
        s2 = df.applymap(lambda val: by_idx(val, 1))
        curr_max_s1 = s1.max().max()
        if s1_max is None or curr_max_s1 > s1_max:
            s1_max = curr_max_s1

        curr_max_s2 = s2.max().max()
        if s2_max is None or curr_max_s2 > s2_max:
            s2_max = curr_max_s2

        curr_min_s1 = s1.min().min()
        if s1_min is None or curr_min_s1 < s1_min:
            s1_min = curr_min_s1

        curr_min_s2 = s2.min().min()
        if s2_min is None or curr_min_s2 < s2_min:
            s2_min = curr_min_s2
    return (s1_min, s1_max), (s2_min, s2_max)


def plot_states_scatter_subplots(main_title, input_data, patients_to_use, sub_tissues_to_use, patient_data,
                                  hue, hue_order, same_lims=True, lim_const=1.1, figsize=(25, 35), to_save=None,
                                 save_pdf=False, legend_loc=(0.4, 0.8), progress_print=False, to_show=True,
                                 title_fontsize=28, ylabel_fontsize=14, scatter_kwargs={}):
    """
    plots multiple scatter plots for states. Each state_df consists of tuples when data available, else empty (None)
    lst_of_states_df_with_names - list of tuples [(name, states_df),..]
    current sizes and locations fit 4 columns X 7 rows

    input_data - [(column title1, appropriate states df to use (normal/anchor/smoothen..), filtering condition dict1),
                 (column title2, df2, filtering_dict2)..]
    """
    if same_lims:
        x_lim, y_lim = calc_scatter_x_y_lims(input_data, patient_data, patients_to_use)
        x_lim = tuple(v*lim_const for v in x_lim)
        y_lim = tuple(v*lim_const for v in y_lim)
    fig, axs = plt.subplots(nrows=len(sub_tissues_to_use), ncols=len(input_data), figsize=figsize,
                            gridspec_kw={'hspace': 0.5}, sharex=True, sharey=True)
    fig.suptitle(main_title, fontsize=50)
    for idx, (name, states_df, filtering_dict) in enumerate(input_data):
        df_to_use = filter_patient_df(patient_data, filtering_dict)
        if progress_print:
            print(name)
        if patients_to_use is not None:
            states_df = states_df[set(patients_to_use) & set(states_df.columns)]
        df_to_use = states_df.transpose().merge(df_to_use, left_index=True, right_index=True)
        # TODO - fix axis to be the same for all figures. potential problems with outliers..
        #  there's example in data_manipulation.var_analysis
        for tissue_idx, sub_tissue in enumerate(sorted(sub_tissues_to_use)):
            curr_row = tissue_idx
            curr_col = idx
            curr_ax = axs[curr_row][curr_col]
            cols_to_use = set([sub_tissue, 'AGE', 'death_reason', hue]) & set(df_to_use.columns)
            curr_data = df_to_use[cols_to_use].dropna()
            curr_data['s1'] = curr_data[sub_tissue].apply(lambda x: x[0])
            curr_data['s2'] = curr_data[sub_tissue].apply(lambda x: x[1])
            # add lines for x=y=0
            curr_ax.axhline(y=0.0, color='black', linestyle='--', linewidth=0.8, )
            curr_ax.axvline(x=0.0, color='black', linestyle='--', linewidth=0.8, )

            a = sns.scatterplot(x='s1', y='s2', hue=hue, data=curr_data, ax=curr_ax, hue_order=hue_order,
                                **scatter_kwargs)
            if same_lims:
                curr_ax.set_xlim(x_lim)
                curr_ax.set_ylim(y_lim)
            handles, labels = curr_ax.get_legend_handles_labels()
            try:
                a.legend_.remove()
            except AttributeError:
                bla = 0
            curr_ax.title.set_text('{}'.format(name))
            curr_ax.title.set_fontsize(title_fontsize)
            if curr_col == 0:
                a.set_ylabel('{}'.format(sub_tissue), fontsize=ylabel_fontsize)

    leg = fig.legend(handles, labels, loc='center', prop={'size': 19}, bbox_to_anchor=legend_loc, ncol=2)
    if to_save is not None:
        plt.savefig(to_save + '.jpg')
        if save_pdf:
            plt.savefig(to_save + '.pdf')
    if to_show:
        plt.show()
    plt.close()
    return fig, axs, leg


def fix_labels_of_pca_map_plots(g, by_ind, obj_to_plot, col_to_state_dfs_d, num_of_cols, patient_data, colorbar_mode):
    for i, ax in enumerate(g.axes.flatten()):
        title, state_df_to_use = col_to_state_dfs_d[i % num_of_cols]
        if by_ind:
            s1, s2 = state_df_to_use[obj_to_plot][undo_print_pretty(g.row_names[i // num_of_cols])]
        else:  # by tissue
            if len(ax.texts):  # add more detailed patient data
                try:
                    text_idx = 1
                    curr_txt = ax.texts[text_idx]
                except IndexError:
                    text_idx = 0
                    curr_txt = ax.texts[text_idx]
                age, sex = patient_data.loc[curr_txt.get_text()][['AGE', 'sex_name']].values
                new_txt = '{}:\n {},{}'.format(curr_txt.get_text(), age, sex)
                plt.setp(ax.texts, text=new_txt, rotation=-90)
                # ax.text(curr_txt.get_unitless_position()[0] + 0.25, curr_txt.get_unitless_position()[1] - 0.7,
                #         s='{}:\n {},{}'.format(curr_txt.get_text(), age, sex), rotation=-90)
                # ax.texts[text_idx].remove()
            curr_ind = g.row_names[i // 4]
            s1, s2 = state_df_to_use[curr_ind][obj_to_plot]

        if i < num_of_cols:
            header = title + '\n'
        else:
            header = ''
        ax.set_title("{}{} {}".format(header, round(s1, 5), round(s2, 5)))

        if not colorbar_mode:
            if i % num_of_cols != num_of_cols - 1:  # rightmost column - different legend
                leg = ax.legend(loc='center right', bbox_to_anchor=(1.72, 0.61))
            else:
                leg = ax.legend(loc='center right', bbox_to_anchor=(2, 0.61))
            fix_text(leg)


def handle_suptitle_and_saving(g, by_ind, obj_to_plot, patient_data, to_save, save_pdf, to_show, colorbar_mode):
    # main title
    if by_ind:
        age, sex, death = patient_data.loc[obj_to_plot][['AGE', 'sex_name', 'death_reason']].values
        g.fig.suptitle("\n{} - {}, {}\n{}".format(obj_to_plot, sex, age, death), y=0.9)
    else:
        g.fig.suptitle("\n{}".format(obj_to_plot), y=0.9)
    # saving
    if to_save is not None:
        if colorbar_mode:
            full_path = os.path.join(to_save, '{}_GE_map_colorbar'.format(obj_to_plot))
        else:
            full_path = os.path.join(to_save, '{}_GE_map'.format(obj_to_plot))
        plt.savefig('{}.jpg'.format(full_path), bbox_inches='tight')
        if save_pdf:
            plt.savefig('{}.pdf'.format(full_path), bbox_inches='tight')
    if to_show:
        plt.show()
    plt.close()


def plot_ge_on_pca_space(stacked_df_ready_to_plot, by_ind, objs_to_plot, filter_col, grid_row, grid_col,
                         scatter_x_col, scatter_y_col, scatter_hue_col, col_to_state_dfs_d, patient_data,
                         gridspec_kws={"wspace": 0.7, "hspace": 0.5}, margin_titles=True, map_kwargs={},
                         sharex=True, sharey=True, to_show=True, to_save=None, save_pdf=False, palette='bwr'
                         ):
    """
    by_ind - Boolean. When False the plot is by tissue
    objs_to_plot - list of tissues/patients to plot
    col_to_state_dfs_d - {idx: (title, stated_df_to_use)}. e.g: {0:('Normal',states),1:('Smoothen',smoothen_states),...}
    to_save - if None, not saving, else to_save is a directory in which we save
    """
    for obj_to_plot in objs_to_plot:
        curr_plotabble = stacked_df_ready_to_plot[stacked_df_ready_to_plot[filter_col] == obj_to_plot]
        g = sns.FacetGrid(curr_plotabble, row=grid_row, col=grid_col, gridspec_kws=gridspec_kws,
                          margin_titles=margin_titles, sharex=sharex, sharey=sharey,
                          col_order=[v[0] for k, v in sorted(col_to_state_dfs_d.items())],
                          row_order=sorted(curr_plotabble[grid_row].unique())
                          )
        g = g.map(sns.scatterplot, scatter_x_col, scatter_y_col, scatter_hue_col, edgecolor=None, lw=0,
                  palette=palette, **map_kwargs)
        num_of_cols = len(g.col_names)

        # Fix the titles and labels in the grid
        [plt.setp(ax.texts, text="") for ax in g.axes.flat]  # remove the original texts
        g.set_titles(row_template='{row_name}', col_template='{col_name}')

        fix_labels_of_pca_map_plots(g, by_ind, obj_to_plot, col_to_state_dfs_d, num_of_cols,
                                    patient_data, colorbar_mode=False)
        handle_suptitle_and_saving(g, by_ind, obj_to_plot, patient_data, to_save, save_pdf, to_show, colorbar_mode=False)


def facet_scatter(x, y, c, **kwargs):
    # helper for plot_ge_on_pca_space_one_colorbar to workaround the params collision
    """Draw scatterplot with point colors from a faceted DataFrame columns."""
    kwargs.pop("color")
    plt.scatter(x, y, c=c, **kwargs)


def plot_ge_on_pca_space_one_colorbar(stacked_df_ready_to_plot, by_ind, objs_to_plot, filter_col, grid_row, grid_col,
                                      scatter_x_col, scatter_y_col, scatter_hue_col, col_to_state_dfs_d, patient_data,
                                      vmin=-7, vmax=7, gridspec_kws={"wspace": 0.7, "hspace": 0.5}, margin_titles=True,
                                      map_kwargs={}, sharex=True, sharey=True, to_show=True, to_save=None,
                                      save_pdf=False, palette='bwr'):
    for obj_to_plot in objs_to_plot:
        curr_plotabble = stacked_df_ready_to_plot[stacked_df_ready_to_plot[filter_col] == obj_to_plot]
        curr_plotabble.sort_values(by=[grid_row], inplace=True)
        g = sns.FacetGrid(curr_plotabble, row=grid_row, col=grid_col, gridspec_kws=gridspec_kws,
                          margin_titles=margin_titles, sharex=sharex, sharey=sharey,
                          col_order=[v[0] for k, v in sorted(col_to_state_dfs_d.items())],
                          row_order=sorted(curr_plotabble[grid_row].unique())
                          )
        g = g.map(facet_scatter, scatter_x_col, scatter_y_col, scatter_hue_col, edgecolor=None, linewidths=0,
                  cmap=palette, vmin=vmin, vmax=vmax, **map_kwargs)

        num_of_cols = len(g.col_names)

        # Fix the titles and labels in the grid
        [plt.setp(ax.texts, text="") for ax in g.axes.flat]  # remove the original texts
        g.set_titles(row_template='{row_name}', col_template='{col_name}')

        # enter one colorbar for all subfigures
        g.fig.subplots_adjust(right=.97)
        cax = g.fig.add_axes([1., .25, .02, .6])  # Define a new Axes where the colorbar will go
        points = plt.scatter([], [], c=[], vmin=vmin, vmax=vmax,
                             cmap=palette)  # Get a mappable object with the same colormap
        g.fig.colorbar(points, cax=cax)  # Draw the colorbar

        fix_labels_of_pca_map_plots(g, by_ind, obj_to_plot, col_to_state_dfs_d, num_of_cols,
                                    patient_data, colorbar_mode=True)
        handle_suptitle_and_saving(g, by_ind, obj_to_plot, patient_data, to_save, save_pdf, to_show, colorbar_mode=True)


def plot_missing_data(data, title, figsize=(12,8), to_save=None, to_show=True, save_pdf=False):
    plt.figure(figsize=figsize)
    sns.heatmap(data.applymap(lambda x: pd.notna(x)))
    plt.title("Missing data - {}".format(title), fontsize=20)
    path_to_save=os.path.join(to_save, "Missing data - {}".format(title))
    save_and_show_figure(path_to_save, to_show=to_show, save_pdf=save_pdf, tight_layout=True, bbox_inches=None)


def subplots_of_correlation(d, tissues_tup, bootstrap_mode, figsize=None, to_save=None, to_show=None, save_pdf=False):
    col_tissues, row_tissues = tissues_tup
    if figsize is None:
        figsize = len(col_tissues)*2.2, len(row_tissues) * 2.2
    # sharey only at correlation
    fig, axs = plt.subplots(nrows=len(row_tissues), ncols=len(col_tissues), figsize=figsize, sharey=not bootstrap_mode)
    for t1, t2 in itertools.product(row_tissues, col_tissues):
        curr_ax = axs[row_tissues.index(t1), col_tissues.index(t2)]
        if bootstrap_mode:
            if t1 == t2:
                corr_vals = []
            else:
                try:
                    corr_vals = [v for v in d[tuple(sorted([t1, t2]))].values()]
                except KeyError:
                    corr_vals = []
            curr_ax.boxplot(corr_vals, medianprops=dict(color='black'), patch_artist=True,
                            boxprops=dict(facecolor="lightblue"))
        else:
            if isinstance(d[tuple(sorted([t1, t2]))], dict):
                corr_vals = sorted(d[tuple(sorted([t1, t2]))].values())
            else:
                corr_vals = sorted(d[tuple(sorted([t1, t2]))])
            curr_ax.plot(corr_vals, marker='o', label=len(corr_vals))
            curr_ax.legend()

        curr_ax.set_ylim(-1, 1)
        curr_ax.axhline(y=0, color='black', linestyle='--', lw=1)

    for i in range(len(row_tissues)):
        axs[0, i].title.set_text(print_pretty(row_tissues[i]))
    for i in range(len(col_tissues)):
        axs[i, 0].set_ylabel(print_pretty(col_tissues[i]), rotation=90)
    if to_save or to_show:
        fig.tight_layout()
    if to_save:
        fig.savefig(to_save+'.jpg')
        if save_pdf:
            fig.savefig(to_save + '.pdf')
    if to_show:
        plt.show()
    plt.close()


def plot_reg_res(reg_res_df, main_title, state_idx,
                 tissues_del_thresh=5, heatmap_kws={'xticklabels': 1, 'yticklabels': 1},
                 to_save=None, to_show=False, save_pdf=False, tight_layout=True, bbox_inches='tight'):
    reg_res_df = filter_df(reg_res_df, tissues_del_thresh=tissues_del_thresh)

    g = sns.clustermap(reg_res_df, cmap='bwr', vmin=-1, vmax=1, **heatmap_kws)

    file_title = 'Regression between tissues - s{}'.format(state_idx + 1)
    curr_title = 'Regression between tissues - {} s{}'.format(main_title, state_idx + 1)
    plt.title(curr_title)

    if to_save is not None or to_show:
        dir_to_save = os.path.join(to_save, file_title) if to_save is not None else None
        save_and_show_figure(dir_to_save, save_pdf=save_pdf, to_show=to_show,
                             tight_layout=tight_layout, bbox_inches=bbox_inches)
    return g


def heatmap_and_save(df, attr_desc_d, curr_vmin, curr_vmax, curr_title, dir_to_save, file_title,
                     bbox_inches='tight', plot_readable_header=True, to_save=True):
    """
    plot heatmap and saves a csv of it
    """
    plt.figure(figsize=(20, 15))
    header_df = df.rename(index=attr_desc_d)
    if plot_readable_header:
        df = df.rename(index={k: v[:40] for k, v in attr_desc_d.items()})
    g = sns.heatmap(df.transpose(), cmap='bwr', xticklabels=1, yticklabels=1,
                    vmin=curr_vmin, vmax=curr_vmax)
    g.set_facecolor('xkcd:black')
    plt.title(curr_title)
    plt.savefig(os.path.join(dir_to_save, '{}.jpg'.format(file_title)), bbox_inches=bbox_inches)
    plt.clf()
    plt.cla()
    plt.close('all')
    if to_save:
        header_df.to_csv(os.path.join(dir_to_save, '{}.csv'.format(file_title)))


def plot_subplots(df_to_plot, title_baseline, attr_trim, tissue_trim, set_background, save_dir, show,
                  hspace, min_vmin=None, get_vmin_vmax_get_bigger=False, figsize=(50, 50), verbose=True):
    """
    gets a tuple-indexed DF, transforms each column to a normal DF and plots an heatmap.
    plots many heatmaps using plt.subplots
    """
    fig, axs = plt.subplots(nrows=math.ceil(df_to_plot.shape[1] / 2), ncols=2,
                            figsize=figsize, gridspec_kw={"wspace": 0.3, "hspace": hspace})
    axs_flat = axs.flatten()
    for idx, col in enumerate(df_to_plot.columns):
        d = df_to_plot[col].to_dict()
        dd = dict()
        for (k1, k2), v in d.items():
            dd.setdefault(k1, dict())[k2] = v
        inner_df = pd.DataFrame(dd).rename(index=lambda val: val[:attr_trim], columns=lambda val: val[:tissue_trim])
        vmin, vmax = get_vmin_vmax(inner_df, get_bigger=get_vmin_vmax_get_bigger)
        if min_vmin is not None and vmin < min_vmin:
            vmin = min_vmin
        g = sns.heatmap(inner_df, ax=axs_flat[idx], cmap='bwr', vmin=vmin, vmax=vmax, xticklabels=1, yticklabels=1)
        if set_background:
            g.set_facecolor('xkcd:black')
        if isinstance(col, tuple):
            title = '{}\n{}, {}'.format(title_baseline, col[0], col[1])
        else:
            title = '{}\n{}'.format(title_baseline, col)
        axs_flat[idx].set_title(title)
    fname = title_baseline
    for char in ["\n", " ", ",", "-"]:
        fname = fname.replace(char, "_")
    while "__" in fname:
        fname = fname.replace("__", "_")
    plt.savefig(os.path.join(save_dir, fname + ".pdf"))
    plt.savefig(os.path.join(save_dir, fname + ".jpg"))
    if show:
        plt.show()
    plt.close()
    if verbose:
        print(title_baseline)


def plot_correlation_scatters(curr_df, curr_bits_d, title_baseline, save_dir, is_same_reg_type, show=False,
                              verbose=True, hspace=0.3):
    """
    plots many scatter plots of pairs of columns in DF
    """
    cols_to_comp = list(curr_df.columns)
    pairs_to_plot = []
    for col_idx1, col_idx2 in list(itertools.combinations(list(range(len(cols_to_comp))), r=2)):
        if get_bits_dist(curr_bits_d[col_idx1], curr_bits_d[col_idx2]) == 1:
            if is_same_reg_type:
                if curr_bits_d[col_idx1][0] == curr_bits_d[col_idx2][0]:
                    pairs_to_plot.append((cols_to_comp[col_idx1], cols_to_comp[col_idx2]))
            else:
                if curr_bits_d[col_idx1][0] != curr_bits_d[col_idx2][0]:
                    pairs_to_plot.append((cols_to_comp[col_idx1], cols_to_comp[col_idx2]))
    if len(pairs_to_plot) == 0:
        print("No pairs for {}".format(title_baseline))
        return
    rows = math.ceil(len(pairs_to_plot) / 3)
    fig, axs = plt.subplots(ncols=3, nrows=rows, figsize=(40, 7 * rows), gridspec_kw={"hspace": hspace})
    flat_axs = axs.flatten()
    for idx, (col1, col2) in enumerate(pairs_to_plot):
        df_to_plot = curr_df[[col1, col2]].dropna()
        corr = np.corrcoef(df_to_plot[col1], df_to_plot[col2])[0, 1]
        sns.scatterplot(data=df_to_plot, x=col1, y=col2, ax=flat_axs[idx])
        curr_title = "{}\n {} VS {}\ncorr={:.3f}".format(title_baseline, col1, col2, corr)
        flat_axs[idx].set_title(curr_title)

    fname = os.path.join(save_dir, '{}.jpg'.format(title_baseline))
    plt.savefig(fname)
    plt.savefig(fname.replace("jpg", "pdf"))
    if show:
        plt.show()
    plt.close()
    if verbose:
        print(title_baseline)