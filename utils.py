import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython
import os
from misc_utils import fix_query_txt, drop_na_by_pct, check_from_option
from scipy.stats import zscore

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


def get_data_by_groups(patient_data, states_df_td, genders_to_use=['Male', 'Female', 'All'],
                       death_type_to_use=['Good', 'Bad', 'All'], ages_to_use=['Old', 'Normal', 'Young', 'All'],
                       states_df_type_to_use=['Anchor'], create_dir=None, verbose=True, pct_of_not_nulls=0.2):
    ret_d = dict()

    for gender in genders_to_use:
        if verbose:
            print(gender)

        if create_dir is not None:
            new_dir = os.path.join(create_dir, gender)
            os.makedirs(new_dir, exist_ok=True)

        if gender == 'All':
            query_gender = ''
            title_gender = 'Both sexes,'
        else:
            query_gender = "sex_name=='{}'".format(gender)
            title_gender = '{},'.format(gender)

        for age_group in ages_to_use:
            if age_group == 'All':
                query_gender_age = query_gender
                title_gender_age = title_gender + ' all ages,'
            else:
                query_gender_age = query_gender + " & AGE_group=='{}'".format(age_group)
                title_gender_age = title_gender + ' {},'.format(age_group)

            for death_group in death_type_to_use:
                if death_group == 'All':
                    query_gender_age_death = query_gender_age
                    title_gender_age_death = title_gender_age + ' all death types,'
                else:
                    query_gender_age_death = query_gender_age + " & Death_group=='{}'".format(death_group)
                    title_gender_age_death = title_gender_age + ' {} death type,'.format(death_group)

                final_query = fix_query_txt(query_gender_age_death)

                if len(final_query) == 0:  # no filtering
                    patient_data_to_use = patient_data
                else:
                    patient_data_to_use = patient_data.query(final_query)
                for states_df_type, states_df_to_use in states_df_td.items():
                    if states_df_type not in states_df_type_to_use:
                        continue
                    for to_filter in [True, False]:
                        filtered_states_df_to_use = states_df_to_use[patient_data_to_use.index]
                        final_title = states_df_type + ', ' + title_gender_age_death
                        if to_filter:
                            filtered_states_df_to_use = drop_na_by_pct(filtered_states_df_to_use, 1,
                                                                       pct_of_not_nulls=pct_of_not_nulls, to_print=False)
                            final_title += ' filtered'

                        if create_dir is not None:
                            final_dir = os.path.join(new_dir, final_title,)
                            os.makedirs(final_dir, exist_ok=True)
                        else:
                            final_dir = None

                        curr_k = gender, age_group, death_group, states_df_type, to_filter
                        ret_d[curr_k] = filtered_states_df_to_use, patient_data_to_use, final_title, final_dir
    return ret_d


def zscore_attr_df(attr_df, attr_metadata, only_continous=True, merge_all_cols=True):
    if only_continous:
        cols_to_standart = attr_metadata[
            (attr_metadata.to_use > 0) & (attr_metadata.calculated_type.isin(['integer', 'decimal']))].index.tolist()
    else:
        cols_to_standart = attr_df.select_dtypes(include=[np.number]).columns.tolist()

    attr_df_zscore = dict()
    for col in cols_to_standart:
        data_no_na = attr_df[[col]].dropna()
        if data_no_na.shape[0] > 0 and len(set(data_no_na.values.flatten())) > 1:
            attr_df_zscore[col] = data_no_na.apply(zscore).to_dict()[col]

    attr_df_zscore_df = pd.DataFrame(attr_df_zscore)
    if not attr_df_zscore_df.index.tolist() == attr_df.index.tolist():
        assert set(attr_df_zscore_df.index.tolist()) == set(attr_df.index.tolist())
        attr_df_zscore_df = attr_df_zscore_df.reindex(attr_df.index)

    if merge_all_cols:
        missing_cols = set(attr_df.columns) - set(attr_df_zscore_df.columns)
        attr_df_zscore_df = attr_df_zscore_df.merge(attr_df[missing_cols], left_index=True, right_index=True)
    return attr_df_zscore_df


def get_group_key_from(pickle_path, idx_pos=-8):
    gender = check_from_option(['Male', 'Female'], pickle_path, 'All', 'Both_sexes')

    state_df_type = check_from_option(['Normal', 'Anchor', 'Smoothen', 'Smoothen_Anchor'], pickle_path, 'Anchor')

    age = check_from_option(['Old', 'Normal', 'Young'], pickle_path, "All", 'all_ages')

    death_type = check_from_option(['Good', 'Bad'], pickle_path, "All", 'all_death_types')

    is_filtered = "filtered" in pickle_path

    idx = pickle_path[idx_pos]
    return (gender, age, death_type, state_df_type, is_filtered), idx


def key_to_dirname(key_tup):
    gender, age_group, death_type, states_df_type, is_filtered = key_tup
    ret_str = "{}, {}, {}, {}".format(states_df_type.strip(), gender.strip(), age_group.strip(), death_type.strip())
    if is_filtered:
        ret_str += " filtered"

    if 'Both' in gender:
        ret_gender = "All"
    else:
        ret_gender = gender.strip()

    return ret_str, ret_gender


def save_attr_df(attr_df, attr_metadata, full_saving_path, to_csv=True, ret=False):
    if len(set(attr_df.columns) & set(attr_metadata.index)) < 10:
        attr_df = attr_df.transpose()
    old_order = attr_df.index.tolist()
    col_to_add = dict()
    for col in attr_df.columns:
        if col in attr_metadata.index:
            col_to_add[col] = attr_metadata.loc[col]['description']
        else:
            col_to_add[col] = None
    attr_df = attr_df.append(pd.DataFrame(col_to_add, index=['full_name']))
    attr_df = attr_df.reindex(['full_name']+old_order)
    if full_saving_path:
        if to_csv:
            attr_df.to_csv(full_saving_path)
        else:
            attr_df.to_pickle(full_saving_path)
    if ret:
        return attr_df


def remove_outliers(attr_df, attr_metadata=None, attr_cols=None, outlier_const=3):
    # find and remove outliers
    if attr_metadata is not None:
        continuous_cols = attr_metadata[attr_metadata.calculated_type.isin(['decimal', 'integer'])].index.tolist()
        continuous_cols = list(set(continuous_cols) & attr_cols)
    else:
        continuous_cols = attr_df.columns
    low_lim = attr_df[continuous_cols].mean() - outlier_const * attr_df[continuous_cols].std()
    high_lim = attr_df[continuous_cols].mean() + outlier_const * attr_df[continuous_cols].std()

    tmp = pd.DataFrame(index=attr_df.index)
    for col in continuous_cols:
        tmp[col] = np.where((attr_df[col] > high_lim[col]) | (attr_df[col] < low_lim[col]), np.nan, 0)
        # TODO: consider, if a lot of nans, remove the whole column values
    # mask - int + np.nan = np.nan
    attr_df[continuous_cols] = attr_df[continuous_cols] + tmp
    return attr_df


def clean_attributes_data(attr_df, attr_metadata, outlier_const=3, unknown_code_vals=[99, 98, 97, 96],
                          remove_unknown=True, remove_outlier=True, zscore=True, strict_removal=None):
    attr_cols = set(attr_df.columns)
    enum_cols = None
    continuous_cols = attr_metadata[attr_metadata.calculated_type.isin(['decimal', 'integer'])].index.tolist()
    continuous_cols = list(set(continuous_cols) & attr_cols)
    if remove_unknown:
        # remove unknown/non reported values
        enum_cols = attr_metadata[(attr_metadata.calculated_type.str.contains("enum")) & (
                    attr_metadata.reported_type != 'string')].index.tolist()
        enum_cols = list(set(enum_cols) & attr_cols)
        attr_df[enum_cols] = attr_df[enum_cols].replace(unknown_code_vals, np.nan)
        if strict_removal is not None:
            attr_df[strict_removal] = attr_df[strict_removal].replace(unknown_code_vals, np.nan)

    if remove_outlier:
        attr_df = remove_outliers(attr_df, attr_metadata, attr_cols, outlier_const)

    if zscore:
        attr_df = zscore_attr_df(attr_df, attr_metadata)
    return attr_df, enum_cols, continuous_cols
