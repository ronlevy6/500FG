import matplotlib.pyplot as plt


def plot_df(df, x, y, to_filter, index_filter_vals, title, to_color=True, colors=None):
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


def plot_df_color_by_col_vals(df, x, y, col, title, is_grid=True, pallete='jet'):
    print(col)
    ax = df.plot.scatter(x=x, y=y, c=col, s=50, cmap=pallete, figsize=(10 ,8))
    ax.set_xlabel(x, fontsize=15)
    ax.set_ylabel(y, fontsize=15)
    if title:
        ax.set_title('{} colored by {}'.format(title, col), fontsize=20)
    else:
        ax.set_title('{} vs {}, colored by {}'.format(x, y, col), fontsize=20)
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