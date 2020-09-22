import uuid
import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


def create_tmp_file(curr_dir, suffix=None):
    filename = str(uuid.uuid4())
    if suffix is not None:
        filename += '.{}'.format(suffix)
    return os.path.join(curr_dir, filename)


def fix_directory(main_dir, sub_dir):
    assert os.path.isdir(main_dir)
    date_dir = datetime.today().strftime('%d_%m_%Y')
    full_dirpath = os.path.join(main_dir, date_dir, sub_dir)
    os.makedirs(full_dirpath, exist_ok=True)
    return full_dirpath


def chunks(obj, chunk_size=10, min_vals_in_chunk=1):
    """Yield successive n-sized chunks from lst."""
    to_break = False
    for i in range(0, len(obj), chunk_size):
        max_i = min(len(obj), i+chunk_size)
        if len(obj) - 1 - max_i < min_vals_in_chunk:
            max_i = len(obj)
            to_break = True
        # yield [obj[j] for j in range(i, max_i)]
        yield obj[i: max_i]
        if to_break:
            break


def fix_str(val):
    """
     runs on DF and makes str of tuple into a tuple of numbers
    """
    if pd.isna(val):
        return val
    else:
        return eval(val)


def fix_str_df(df):
    """
    run fix_str on df
    """
    return df.applymap(fix_str)


def drop_na_by_pct(df, axis, pct_of_not_nulls=0.5, to_print=False):
    thresh = df.shape[axis]*pct_of_not_nulls
    ret_df = df.dropna(axis=1-axis, thresh=thresh)
    if to_print:
        print(df.shape, ret_df.shape)
    return ret_df


# def make_pretty(val):
#     """
#     fit sub tissue value into subplots spacing..
#     """
#     sp = val.split(" ")
#     if len(sp) == 1:
#         return val
#     elif len(sp) >= 3:
#         return sp[0] + ' ' + sp[2][:10]
#     else:
#         return val
#

def print_pretty_old(txt):
    splitted = txt.split(' ')
    mid = len(splitted)//2
    return ' '.join(splitted[:mid]) + '\n' + ' '.join(splitted[mid:])


def print_pretty(txt, line_const=0.55):
    in_s1 = True
    total_len = len(txt)
    s1 = ''
    s2 = ''
    splitted = txt.split(' ')
    if len(splitted) == 3 and splitted[1] == '-':
        return ''.join(splitted[:2]) + '\n' + ''.join(splitted[2:])
    for word in splitted:
        if in_s1 and len(s1) < total_len / 2 and (len(s1) + len(word)) < total_len * line_const:
            s1 += ' {}'.format(word)
        else:
            in_s1 = False
            s2 += ' {}'.format(word)
    s1 = s1.replace(' - ', '-')
    s2 = s2.replace(' - ', '-')
    final_str = s1.strip() + '\n' + s2.strip()
    return final_str.strip()


def undo_print_pretty(txt):
    return txt.replace('-', ' - ', 1).replace('\n', ' ').replace('  ', ' ')


def fix_values_specific(val, min_val=-7, max_val=7):
    """
    changes boundaries of value to fit limits (prevent tail..)
    """
    if val < min_val:
        return min_val
    if val > max_val:
        return max_val
    return val


def fix_text(legend, entry_len=3):
    """
    shortens length of entries in legend
    :return:
    """
    for entry in legend.get_texts():
        txt = entry.get_text()
        try:
            ftxt = float(txt)
            entry.set_text(round(ftxt, entry_len))
        except ValueError:
            continue


def save_and_show_figure(full_path_without_extension, to_show=True, save_pdf=False, tight_layout=False, bbox_inches=None):
    # full_path_without_extension needs to be exist!
    if tight_layout:
        plt.tight_layout()
    if full_path_without_extension is not None:
        plt.savefig(full_path_without_extension+'.jpg', bbox_inches=bbox_inches)
        if save_pdf:
            plt.savefig(full_path_without_extension + '.pdf', bbox_inches=bbox_inches)
    if to_show:
        plt.show()
    plt.close()


def fix_query_txt(query_txt):
    query_txt = query_txt.strip()
    if len(query_txt) == 0:
        return query_txt
    if query_txt[0] == "&":
        return query_txt[1:].strip()
    return query_txt


def dropna_ndarray(arr):
    return arr[~np.isnan(arr)]


def filter_df(df, tissues_del_thresh=5):
    # remove all nans tissues
    to_del = df[df.isna().sum() == df.shape[0]].index
    df = df.drop(columns=to_del).drop(index=to_del)

    while df.isna().sum().sum() and tissues_del_thresh > 0:
        tissues_del_thresh -= 1
        to_del = set(df[df.isna().sum() > tissues_del_thresh].index)
        df = df.drop(columns=to_del).drop(index=to_del)

    return df
