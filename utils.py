import uuid
import os
import math
import pandas as pd
import numpy as np


def create_tmp_file(curr_dir, suffix=None):
    filename = str(uuid.uuid4())
    if suffix is not None:
        filename += '.{}'.format(suffix)
    return os.path.join(curr_dir, filename)


# center - x0,y0
def get_angle_in_circle(x, y, x0, y0):
    (dx, dy) = (x0-x, y-y0)
    # angle = atan(float(dy)/float(dx))
    angle = math.degrees(math.atan2(float(dy), float(dx)))
#     if angle < 0:
#         angle += 180
    return angle


def apply_angle(df, x_col, y_col, x0, y0,):
    return df.apply(lambda row: get_angle_in_circle(row[x_col], row[y_col], x0, y0), axis=1)


def find_center_sensitive(space_df, x_col, y_col, num_of_values_to_use=20):
    res = []
    for col in [x_col, y_col]:
        min_point = np.mean(sorted(space_df[col])[:num_of_values_to_use])
        max_point = np.mean(sorted(space_df[col], reverse=True)[:num_of_values_to_use])
        val = max_point + min_point
        res.append(val/2)
    return res


def get_amit_anchors():
    with open('/export/home/ronlevy/projects/500FG_GTEx/amits_anchors_genes/selectedGenesForLupus.txt') as f:
        amits_filtered_data = f.readlines()
        amits_filtered_data = set([f.strip() for f in amits_filtered_data])

    with open('/export/home/ronlevy/projects/500FG_GTEx/amits_anchors_genes/selectedGenesForLupusLarge.txt') as f:
        amits_full_data = f.readlines()
        amits_full_data = set([f.strip() for f in amits_full_data])

    return amits_full_data, amits_filtered_data


def chunks(obj, chunk_size=10):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(obj), chunk_size):
        max_i = min(len(obj), i+chunk_size)
        # yield [obj[j] for j in range(i, max_i)]
        yield obj[i: max_i]


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


def rename_patient_name(df):
    """
    format GTEX naming to shorten, so individuals from several tissues could be easily joined
    """
    df.rename(columns=lambda x: 'GTEX-{}'.format(x.split('-')[1]), inplace=True)
    return df


def drop_na_by_pct(df, axis, pct_of_not_nulls=0.5, to_print=False):
    thresh = df.shape[axis]*pct_of_not_nulls
    ret_df = df.dropna(axis=1-axis, thresh=thresh)
    if to_print:
        print(df.shape, ret_df.shape)
    return ret_df


def make_pretty(val):
    """
    fit sub tissue value into subplots spacing..
    """
    sp = val.split(" ")
    if len(sp) == 1:
        return val
    elif len(sp) >= 3:
        return sp[0] + ' ' + sp[2][:10]
    else:
        return val


def print_pretty(txt):
    splitted = txt.split(' ')
    mid = len(splitted)//2
    return ''.join(splitted[:mid]) + '\n' + ''.join(splitted[mid:])


def by_idx(val, idx):
    """
    return val[idx] in case val is not None
    :param val: indexed object - list/tuple
    :param idx: index to access
    :return: val[idx]
    """
    if pd.isna(val) or idx is None:
        return val
    return val[idx]


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


def assign_age_group(val):
    if val in ['50-59', '40-49']:
        return '40-60'
    elif val in ['60-69', '70-79']:
        return 'Above 60'
    elif val in ['30-39', '20-29']:
        return 'Below 40'
    else:
        raise


def dedup_space_df(space_df, subset_cols=None):
    idx_name = space_df.index.name
    if idx_name is None:
        idx_name = 'index'
    return space_df.reset_index().drop_duplicates(subset=subset_cols).set_index(idx_name)