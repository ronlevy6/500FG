import numpy as np
import pandas as pd
from IPython import display
from utils import IS_NOTEBOOK


def translate_key(key):
    """
    make string based on booleans of each regression setting
    :param key:
    :return:
    """
    model, remove_outliers, covariates_mode, ischemic_mode, sex = key
    ret = model
    if remove_outliers:
        ret += " no outliers"
    if covariates_mode:
        ret += " covariates"
    if ischemic_mode:
        ret += " ischemic"
    ret += " {}".format(sex)
    return ret


def create_many_dfs(d_to_be_table):
    d_of_dfs = dict()
    for k in d_to_be_table:
        d_of_dfs[k] = pd.DataFrame(d_to_be_table[k]).transpose().dropna(axis=1, thresh=1)
    return d_of_dfs


def make_levels_in_dfs(d_of_dfs):
    """
    since some of the DFs had levels in columns we organize it to be the same
    """
    for k, df_to_use in d_of_dfs.items():
        pretty_k = translate_key(k)
        try:
            lvls = df_to_use.columns.__getattribute__("levels")
            if lvls == 1:
                df_to_use.columns = pd.MultiIndex.from_product([[pretty_k], df_to_use.columns])
        except AttributeError:
            df_to_use.columns = pd.MultiIndex.from_product([[pretty_k], df_to_use.columns])
    return d_of_dfs


def unite_to_one_df(d_of_dfs):
    """
    unite many DFs into one and add missing columns
    """
    idx_to_comp = None
    for df in d_of_dfs.values():
        if idx_to_comp is None:
            idx_to_comp = df.index
        else:
            # make sure all indexes are inside the first one chosen
            if len(set(df.index.tolist()) - set(idx_to_comp.tolist())) != 0:
                to_add = list(set(df.index) - set(idx_to_comp))
                print("to add", len(to_add))
                idx_to_comp = idx_to_comp.append(pd.MultiIndex.from_tuples(to_add))

    final_df = pd.DataFrame(index=idx_to_comp)

    for df in d_of_dfs.values():
        try:
            final_df = final_df.merge(df, right_index=True, left_index=True, how='outer')
        except UserWarning as e:
            assert 0 in final_df.shape
            print("merge warning when final df is empty")
            final_df = df

    final_df.columns = pd.MultiIndex.from_tuples(final_df.columns)
    return final_df


def out_of_list(val):
    if isinstance(val, np.ndarray):
        assert len(val) == 1
        return val[0]
    return val


def get_k_by_attr_and_tissue(row, k_per_attr_tissue_d):
    row_idx = row.name
    t = row_idx[0]
    attr = row_idx[1]
    return k_per_attr_tissue_d[t].get(attr, None)


def order_columns(df, reg_type_order=["k", 'OLS', 'RLM', 'LinearSVR', 'Logit'],
                  reg_config_order=["natural", "intercept", "covariates", "ischemic", "covariates intercept",
                                    "covariates ischemic"],
                  sub_cols=['beta', 'intercept', 'num', 'pval', 'r2', 'aic'], no_outlier_order=[False, True]):
    """
    order columns of df by regression type and other settings
    """
    new_col_order = []
    all_cols_to_order = []
    cols = list(df.columns)
    outlier_order = ["no outliers" * int(no_outlier_first) for no_outlier_first in no_outlier_order]
    for reg_config in reg_config_order:
        for reg_type in reg_type_order:
            if reg_type == 'k' and reg_config == reg_config_order[0]:
                tmp_lst = [c for c in cols if c[0] == 'k']
                if len(tmp_lst):
                    new_col_order.append(tmp_lst[0])
            else:
                for outlier_txt in outlier_order:
                    curr_col_name = reg_type
                    curr_col_name = '{} {}'.format(curr_col_name, outlier_txt)
                    if reg_config != 'natural':
                        curr_col_name = '{} {}'.format(curr_col_name, reg_config)
                    curr_col_name = curr_col_name.replace("  ", " ").strip()
                    cols_to_add = [(curr_col_name, sub_col) for sub_col in sub_cols]
                    for col_to_add in cols_to_add:
                        all_cols_to_order.append(col_to_add)
                        if col_to_add in cols:
                            new_col_order.append(col_to_add)
    #     return new_col_order
    df2 = df.reindex(columns=new_col_order, )
    return df2, new_col_order, all_cols_to_order


def flat_table(df):
    df_flatten = df.copy()
    df_flatten.columns = df.columns.to_flat_index()
    return df_flatten


def extract_val_from_tup(tup, idx):
    if isinstance(tup, list):
        if len(tup) == 1:
            if idx == 0:
                return
            else:
                return tup[0]
        else:
            return tup[idx]


def add_minor_and_major_vals_with_counts(df_to_change, k_labels_and_nums):
    data_to_insert = dict()
    for idx, row in df_to_change.iterrows():
        tissue, attr = idx
        curr_val = k_labels_and_nums.loc[attr, tissue]
        if isinstance(curr_val, pd.Series):
            curr_val = curr_val.apply(eval)
            sums_per_dataset = curr_val.apply(lambda row: sum(tup[1] for tup in row))
            try:
                curr_n = row['num, OLS']
            except KeyError:
                curr_n = row['num, Logit']
            row_to_take = None
            for row_idx, (_, num_data) in enumerate(sums_per_dataset.iteritems()):
                if num_data == curr_n:
                    row_to_take = row_idx
            try:
                assert row_to_take is not None
            except AssertionError:
                print(sums_per_dataset)
                print(curr_n)
                print(tissue, attr)
                raise
            data_to_insert[idx] = curr_val.iloc[row_to_take]
        else:
            if pd.isna(curr_val):
                continue
            curr_val = eval(curr_val)
            data_to_insert[idx] = curr_val

    if 'discrete_vals' not in df_to_change.columns:
        df_to_change.insert(1, 'discrete_vals', None)
        df_to_change.insert(1, 'major_val', None)
        df_to_change.insert(1, 'minor_val', None)

    df_to_change['discrete_vals'] = df_to_change.index.map(data_to_insert)
    df_to_change["minor_val"] = df_to_change['discrete_vals'].apply(lambda row: extract_val_from_tup(row, 0))
    df_to_change["major_val"] = df_to_change['discrete_vals'].apply(lambda row: extract_val_from_tup(row, 1))
    df_to_change.drop(columns=['discrete_vals'], inplace=True)
    # remove rows with k=1
    df_to_change = df_to_change[~(df_to_change['k, k'] == 1)]
    return df_to_change


def str_to_tup(val):
    """
    change the columns of (val, #apperances) read as str to truple
    :param val:
    :return:
    """
    if pd.isna(val):
        return val
    return eval(val)


def remove_rows_with_small_minor_group(df_to_fix, minor_group_min_samples=5):
    """
    remove analyses with less than minor_group min samples(5) samples
    meaning, if there are 50 healthy and 3 sick this row will be deleted
    """
    try:
        cols_to_fix = df_to_fix.select_dtypes(include=['object']).columns
        df_to_fix[cols_to_fix] = df_to_fix[cols_to_fix].applymap(str_to_tup)
    except TypeError as e:
        print("Did it run by now?")
    df_to_fix = df_to_fix[df_to_fix['minor_val'].apply(lambda tup: pd.isna(tup) or tup[1] > minor_group_min_samples)]
    return df_to_fix


def extract_cols(df, regression_types_to_use=[], regression_configs_to_use=[], subcols_to_use=[],
                 cols_txt_to_remove=[], spacer="(.*)", add_k=False):
    """
    df is in the format of excel file of attr,tissue and many regression results and attributes
    it extracts
    """
    types_as_str = "|".join(regression_types_to_use)
    configs_as_str = "|".join(regression_configs_to_use)
    subcols_as_str = "|".join(subcols_to_use)

    regex_query = ""
    for curr_str_to_use in [subcols_as_str, types_as_str, configs_as_str, ]:
        if len(curr_str_to_use) > 2:
            regex_query += '({}){}'.format(curr_str_to_use, spacer)

    dff = df.filter(regex=regex_query)

    if len(cols_txt_to_remove):
        to_remove_as_str = "|".join(cols_txt_to_remove)
        cols_to_remove = set(dff.filter(regex=to_remove_as_str).columns)
        dff = dff[[col for col in dff.columns if col not in cols_to_remove]]

    if add_k:
        dff[("k", "k")] = dff.index.map(df[("k", "k")].to_dict())

    return dff


def mask_df_by_sample_size(num_df, df_to_mask, min_n=20):
    assert [col.split(",")[1] for col in num_df.columns] == [col.split(",")[1] for col in df_to_mask.columns]
    assert num_df.index.tolist() == df_to_mask.index.tolist()
    min_n_mask = np.ma.masked_where(num_df > min_n, df_to_mask).mask
#     min_n_mask = pd.DataFrame(min_n_mask, columns=df_to_mask.columns, index=df_to_mask.index)
    return df_to_mask.where(min_n_mask)


def naive_mask_by_sample_size(original_df, df_to_mask=None, min_n=30, subcols_to_extract=None,
                              max_diff_threshold=7, max_pct_diff_threshold=0.9, verbose=True):
    """
    remove samples with less than min_n samples
    """
    if df_to_mask is None:
        df_to_mask = extract_cols(original_df, subcols_to_use=subcols_to_extract, add_k=False)

    num_df = extract_cols(original_df, subcols_to_use=["num"], add_k=False)
    num_df['min_num'] = num_df.min(axis=1)

    num_diff = num_df.max(axis=1) - num_df.min(axis=1)
    num_diff = num_diff[num_diff > 0]
    try:
        assert num_diff.shape[0] == 0 or num_diff.max() < max_diff_threshold
    except AssertionError:
        print('num diff', num_diff.max(), (num_df.min(axis=1) / num_df.max(axis=1)).min())
        assert max_pct_diff_threshold <= (num_df.min(axis=1) / num_df.max(axis=1)).min()

    #         return None, num_df, None
    if verbose:
        print(num_diff.index.get_level_values('attr').value_counts())
        print(num_diff.value_counts())

    too_low = num_df.min_num < min_n
    enough_samples = num_df.min_num >= min_n
    if verbose:
        print("num of too low:", too_low.sum())

    enough_samples_idx = list()
    for idx, val in enumerate(enough_samples.values):
        if val:
            enough_samples_idx.append(idx)
    masked_df = df_to_mask.iloc[enough_samples_idx]
    return masked_df, num_df, enough_samples_idx


def make_bits(df, pattern_to_look_for=['no outliers', 'covariates', 'ischemic'],
              models=['OLS', 'RLM', 'LinearSVR', 'Logit']):
    """
    Gets a df and returns a list of tuples, each tuple represents a column.
    Each tuple contains booleans marking if the column has a specific property or not. (By column name)
    (regression_model, outliers, covariates, ischemic)
    """
    ret_lst = []
    for col in df.columns:
        lst_to_be_tup = []
        for model in models:
            if model in col:
                lst_to_be_tup.append(model)
                continue
        for p in pattern_to_look_for:
            lst_to_be_tup.append(p in col)
        ret_lst.append(tuple(lst_to_be_tup))
    return ret_lst


def compare_tuple_bits(tup1, tup2, diff_bit):
    for i in range(len(tup1)):
        if tup1[i] != tup2[i] and i != diff_bit:
            return False
    return True


def get_bits_dist(tup1, tup2):
    dist = 0
    for i in range(len(tup1)):
        if tup1[i] != tup2[i]:
            dist += 1
    return dist


def get_delta_df_by_bits(df, bits_data, diff_bit, validate_multi=True, return_multi=False):
    d = dict()
    d_multi = dict()

    cols_lst = df.columns.tolist()
    for i in range(len(cols_lst)):
        col1 = cols_lst[i]
        for j in range(i + 1, len(cols_lst)):
            col2 = cols_lst[j]
            if compare_tuple_bits(bits_data[i], bits_data[j], diff_bit):
                d[cols_lst[i]] = df.iloc[:, i] - df.iloc[:, j]
                d_multi[cols_lst[i], cols_lst[j]] = df.iloc[:, i] - df.iloc[:, j]

    ret_df = pd.DataFrame(d).rename(columns=lambda x: x.replace(", ", "_").replace(" ", "_"))
    df_multi = pd.DataFrame(d_multi)
    if validate_multi:
        if IS_NOTEBOOK:
            display.display(df_multi.head(n=3))
        else:
            print(df_multi.head(n=3))
    if return_multi:
        return df_multi
    return ret_df
