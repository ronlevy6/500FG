import matplotlib.pyplot as plt
import seaborn as sns


def plot_df(df, x, y, to_filter=False, index_filter_vals=None, title=None, to_color=True, colors=None):
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


def plot_states_scatter_subplots(title, lst_of_states_df_with_names, patients_to_use, sub_tissues_to_use, patient_data,
                                  hue, hue_order, figsize=(25, 35), to_save=None, legend_loc=(0.4, 0.8),
                                  progress_print=False, to_show=True, title_fontsize=28, ylabel_fontsize=14):
    """
    plots multiple scatter plots for states. Each state_df consists of tuples when data available, else empty (None)
    lst_of_states_df_with_names - list of tuples [(name, states_df),..]
    current sizes and locations fit 4 columns X 7 rows
    """
    fig, axs = plt.subplots(nrows=len(sub_tissues_to_use), ncols=len(lst_of_states_df_with_names), figsize=figsize,
                            gridspec_kw={'hspace': 0.5}, sharex=True, sharey=True)
    fig.suptitle(title, fontsize=50)
    for idx, (name, df_to_use) in enumerate(lst_of_states_df_with_names):
        if progress_print:
            print(name)
        if patients_to_use is not None:
            df_to_use = df_to_use[set(patients_to_use)&set(df_to_use.columns)]
        df_to_use = df_to_use.transpose().merge(patient_data, left_index=True, right_index=True)
        for tissue_idx, sub_tissue in enumerate(sorted(sub_tissues_to_use)):
            curr_row = tissue_idx
            curr_col = idx
            curr_ax = axs[curr_row][curr_col]
            curr_data = df_to_use[[sub_tissue, 'AGE', 'death_reason', hue]].dropna()
            curr_data['s1'] = curr_data[sub_tissue].apply(lambda x: x[0])
            curr_data['s2'] = curr_data[sub_tissue].apply(lambda x: x[1])
            # add lines for x=y=0
            curr_ax.axhline(y=0.0, color='black', linestyle='--', linewidth=0.8, )
            curr_ax.axvline(x=0.0, color='black', linestyle='--', linewidth=0.8, )

            a = sns.scatterplot(x='s1', y='s2', hue=hue, data=curr_data, ax=curr_ax, hue_order=hue_order)
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
        plt.savefig(to_save + '.pdf')
    if to_show:
        plt.show()
    plt.close()
    return fig, axs, leg
