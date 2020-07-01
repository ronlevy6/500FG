import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython

IS_NOTEBOOK = 'zmqshell' in str(get_ipython())


def get_angle_in_circle(x, y, x0, y0):
    # center - x0,y0
    (dx, dy) = (x0-x, y-y0)
    # angle = atan(float(dy)/float(dx))
    angle = math.degrees(math.atan2(float(dy), float(dx)))
#     if angle < 0:
#         angle += 180
    return angle


def apply_angle(df, x_col, y_col, x0, y0,):
    return df.apply(lambda row: get_angle_in_circle(row[x_col], row[y_col], x0, y0), axis=1)


def get_amit_anchors():
    with open('/export/home/ronlevy/projects/500FG_GTEx/amits_anchors_genes/selectedGenesForLupus.txt') as f:
        amits_filtered_data = f.readlines()
        amits_filtered_data = set([f.strip() for f in amits_filtered_data])

    with open('/export/home/ronlevy/projects/500FG_GTEx/amits_anchors_genes/selectedGenesForLupusLarge.txt') as f:
        amits_full_data = f.readlines()
        amits_full_data = set([f.strip() for f in amits_full_data])

    return amits_full_data, amits_filtered_data


def rename_patient_name(df):
    """
    format GTEX naming to shorten, so individuals from several tissues could be easily joined
    """
    df.rename(columns=lambda x: 'GTEX-{}'.format(x.split('-')[1]), inplace=True)
    return df


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


def compare_state_dfs(df1, df2, verbose=False, round_dig=5):
    # the fillna is to be able to compare nans
    lst = []
    shared_columns = list(set(df1.columns) & set(df2.columns))
    for idx in [0, 1]:
        df1_fixed = df1.applymap(lambda val: by_idx(val, idx)).fillna(666).applymap(lambda val: round(val, round_dig))
        df2_fixed = df2.applymap(lambda val: by_idx(val, idx)).fillna(666).applymap(lambda val: round(val, round_dig))
        df1_fixed.sort_index(inplace=True)
        df2_fixed.sort_index(inplace=True)
        df1_fixed = df1_fixed[shared_columns]
        df2_fixed = df2_fixed[shared_columns]
        lst.append(df1_fixed.equals(df2_fixed))
        if verbose:
            print("Compared idx-{}".format(idx + 1))
    return lst


def correlate_between_dfs(df1, df2, verbose=True, title=None, col_label='column', idx_label='index',
                          min_periods=25, hist_kwargs={}):
    shared_columns = list(set(df1.columns) & set(df2.columns))
    res_d = dict()
    for idx in [0, 1]:
        res_d[idx] = dict()
        df1_fixed = df1.applymap(lambda val: by_idx(val, idx)).sort_index()
        df2_fixed = df2.applymap(lambda val: by_idx(val, idx)).sort_index()
        df1_fixed = df1_fixed[shared_columns]
        df2_fixed = df2_fixed[shared_columns]
        corrs_col = dict()
        corrs_idx = dict()

        for col in df1_fixed.columns:
            res = df1_fixed[col].corr(df2_fixed[col], min_periods=min_periods)
            if pd.notna(res):
                corrs_col[col] = res

        for index in df1_fixed.index:
            res = df1_fixed.loc[index].corr(df2_fixed.loc[index], min_periods=min_periods)
            if pd.notna(res):
                corrs_idx[index] = res

        if verbose:
            for dict_to_use, label in [(corrs_col, col_label), (corrs_idx, idx_label)]:
                vals = list(dict_to_use.values())
                plt.hist(vals, **hist_kwargs)
                plt.title(
                    "s{} By {}\n{}\nmean: {}, std: {}".format(idx + 1, label, title, np.mean(vals), np.std(vals)))
                plt.show()
                plt.close()
        
        res_d[idx][col_label] = corrs_col
        res_d[idx][idx_label] = corrs_idx
    return res_d
