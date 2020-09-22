import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython
import os

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


def rotate_around_x0_y0(df, x_col, y_col, x0, y0, angle=np.pi/4, same_idx=True):
    # center - x0,y0
    n = len(df[x_col])
    x_data = df[x_col].values
    y_data = df[y_col].values
    x0_arr = x0 * np.ones(n)
    y0_arr = y0 * np.ones(n)
    angle_arr = angle * np.ones(n)
    x_new = x0_arr + (x_data - x0_arr)*np.cos(angle_arr) - (y_data - y0_arr)*np.sin(angle_arr)
    y_new = y0_arr + (x_data - x0_arr)*np.sin(angle_arr) + (y_data - y0_arr)*np.cos(angle_arr)
    ret_df = pd.DataFrame({x_col: x_new, y_col: y_new})
    if same_idx:
        ret_df.index = df.index
    return ret_df


def rotate(df, x_col, y_col, angle, in_radians, return_all_cols=True):
    df_to_rotate = df[[x_col, y_col]]
    if not in_radians:
        theta = np.radians(angle)
    else:
        theta = angle
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    comp1 = df_to_rotate.apply(lambda x: np.linalg.solve(R, x)[0], axis=1)
    comp2 = df_to_rotate.apply(lambda x: np.linalg.solve(R, x)[1], axis=1)
    df_res = pd.concat([comp1, comp2], axis=1)
    df_res.columns = [x_col, y_col]
    if return_all_cols:
        assert df_res.index.tolist() == df.index.tolist()
        for col in df.columns:
            if col not in df_res.columns:
                df_res[col] = df[col]
    return df_res


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


def read_df(data_dir, f):
    df = pd.read_pickle(os.path.join(data_dir, f))
    df = df.astype(np.float64)
    rename_patient_name(df)
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


def get_states(states_df):
    s1 = states_df.applymap(lambda val: by_idx(val, 0))
    s2 = states_df.applymap(lambda val: by_idx(val, 1))
    return s1, s2


def assign_age_group(val):
    if val in ['50-59', '40-49']:
        return '40-60'
    elif val in ['60-69', '70-79']:
        return 'Above 60'
    elif val in ['30-39', '20-29']:
        return 'Below 40'
    else:
        raise


def assign_death_group(row):
    if row.DTHHRDY in (1,2):
        return 'Good'
    else:
        return 'Bad'


def dedup_space_df(space_df, subset_cols=None):
    idx_name = space_df.index.name
    if idx_name is None:
        idx_name = 'index'
    return space_df.reset_index().drop_duplicates(subset=subset_cols).set_index(idx_name)


def compare_state_dfs(df1, df2, verbose=False, round_dig=5, cols_to_use=None):
    # the fillna is to be able to compare nans
    lst = []
    shared_columns = list(set(df1.columns) & set(df2.columns))
    if cols_to_use is not None:
        shared_columns = list(set(shared_columns) & set(cols_to_use))
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
    shared_indexes = list(set(df1.index) & set(df2.index))
    res_d = dict()
    for idx in [0, 1]:
        res_d[idx] = dict()
        df1_fixed = df1.applymap(lambda val: by_idx(val, idx)).sort_index()
        df2_fixed = df2.applymap(lambda val: by_idx(val, idx)).sort_index()
        df1_fixed = df1_fixed.loc[shared_indexes, shared_columns]
        df2_fixed = df2_fixed.loc[shared_indexes, shared_columns]
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


def extract_correlations(clustering_res, corr_clustering_res_idx=None):
    # in case more than one clustering method si running, corr_clustering_res_idx will hold it's index in the dictionary
    state0_corrs = dict()
    state1_corrs = dict()
    for k_tup in clustering_res:
        if corr_clustering_res_idx is not None:
            inner_d_to_use = clustering_res[k_tup][corr_clustering_res_idx]
        else:
            inner_d_to_use = clustering_res[k_tup]
        for state_idx, (curr_corr_df, _, _) in inner_d_to_use.items():
            tmp_d = curr_corr_df.to_dict()
            for t1, t1_corrs in tmp_d.items():
                for t2, t1_t2_corr in t1_corrs.items():
                    if state_idx == 0:
                        state0_corrs.setdefault(tuple(sorted([t1,t2])), dict())[k_tup] = t1_t2_corr
                    else:
                        state1_corrs.setdefault(tuple(sorted([t1,t2])), dict())[k_tup] = t1_t2_corr
    return state0_corrs, state1_corrs


def make_subdir_from_key(k):
    subdir = ""
    gender, age_group, death_group, states_df_type, to_filter = k
    subdir += "{} ".format(states_df_type)

    if gender == 'All':
        subdir += "Both sexes, "
    else:
        subdir += "{}, ".format(gender)

    if age_group == 'All':
        subdir += "all ages, "
    else:
        subdir += "{}, ".format(age_group)

    subdir += "{} death types".format(death_group)
    if to_filter:
        subdir += ' filtered'
    return subdir
