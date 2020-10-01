import pandas as pd
import statsmodels.api as sm
from utils import IS_NOTEBOOK
if IS_NOTEBOOK:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm
from misc_utils import flatten_dict


def fit_tissues(x, y, fit_intercept=True):

    x = x.dropna().astype(float)
    for col in x.columns:
        if len(set(x[col].values)) == 1:
            x = x.drop(columns=[col])
    y = y.dropna().astype(float)

    rows_to_use = set(x.index) & set(y.index)
    if len(rows_to_use) == 0:
        return None

    x = x.loc[rows_to_use]
    y = y.loc[rows_to_use]

    assert x.index.tolist() == y.index.tolist()

    if fit_intercept:
        x = sm.add_constant(x, has_constant='add')
    model_sm = sm.OLS(y, x).fit()
    return model_sm


def make_df(dict_to_use, make_symmetric):
    dict_of_dfs = dict()
    for k_tup in dict_to_use:
        tmp = dict_to_use[k_tup]
        new_tmp = dict()
        for k1 in tmp:
            for k2, v in tmp[k1].items():
                new_tmp.setdefault(k1, dict())[k2] = v
                if make_symmetric and (k2 not in tmp or k1 not in tmp[k2]):
                    new_tmp.setdefault(k2, dict())[k1] = v
        tmp_df = pd.DataFrame.from_dict(new_tmp, orient='index')
        tmp_df = tmp_df[sorted(tmp_df.columns)]
        tmp_df = tmp_df.reindex(tmp_df.columns)
        dict_of_dfs[k_tup] = tmp_df
    return dict_of_dfs


def filter_reg_dict(reg_dict, use_attr=False):
    filtered_reg_dict = dict()
    for k, v in tqdm(reg_dict.items()):
        if v is not None:
            new_v = v.params, v.pvalues, v.f_pvalue
            filtered_reg_dict[k] = new_v
    return filtered_reg_dict


def filter_reg_dict_cluster(outer_key, inner_reg_futs_dict, use_attr=False):
    filtered_reg_dict = dict()
    for (t1, t2, attr_col), reg_sm in inner_reg_futs_dict.items():
        if use_attr:
            gender, age_group, death_group, states_df_type, to_filter, idx = outer_key
            new_k = gender, age_group, death_group, states_df_type, to_filter, t1, t2, idx, attr_col
        else:
            new_k = outer_key
        if reg_sm is not None:
            new_v = reg_sm.params, reg_sm.pvalues, reg_sm.f_pvalue
            filtered_reg_dict[new_k] = new_v
    return filtered_reg_dict


def organize_reg_dict(filtered_reg_dict, separate_s1_s2=True, create_dfs=True, make_symmetric=True,
                      use_attr=False, to_merge=True, assert_key=True):
    if isinstance(filtered_reg_dict, list) and to_merge:
        flatten_reg_dict = flatten_dict(filtered_reg_dict, assert_disjoint=assert_key)
    else:
        flatten_reg_dict = filtered_reg_dict
    if use_attr:
        key_tup_to_use = 4
    else:
        key_tup_to_use = 3

    if separate_s1_s2:
        filtered_reg_dict_s1 = dict()
        filtered_reg_dict_s2 = dict()
        for k, v in tqdm(flatten_reg_dict.items()):
            keys = k[:-key_tup_to_use]
            if use_attr:
                t1, t2, idx, attr = k[-key_tup_to_use:]
            else:
                t1, t2, idx = k[-key_tup_to_use:]
            params, variables_pvalues, f_pvalue = v
            if t1 not in params and t2 not in params:
                continue
            try:
                curr_val = params[t1]
            except KeyError:
                curr_val = params[t2]
            if idx == 0:
                curr_d = filtered_reg_dict_s1
            else:
                curr_d = filtered_reg_dict_s2

            if use_attr:
                curr_d.setdefault(keys, dict()).setdefault(t1, dict()).setdefault(t2, dict())[attr] = curr_val
            else:
                curr_d.setdefault(keys, dict()).setdefault(t1, dict())[t2] = curr_val

        if create_dfs:
            filtered_reg_dict_s1_dfs = make_df(filtered_reg_dict_s1, make_symmetric)
            filtered_reg_dict_s2_dfs = make_df(filtered_reg_dict_s2, make_symmetric)
            return filtered_reg_dict_s1_dfs, filtered_reg_dict_s2_dfs, flatten_reg_dict
        else:
            return filtered_reg_dict_s1, filtered_reg_dict_s2, flatten_reg_dict
    else:
        return flatten_reg_dict

