from misc_utils import flatten_dict
import pandas as pd
import statsmodels.api as sm
from utils import IS_NOTEBOOK
if IS_NOTEBOOK:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def fit_tissues(x, y, fit_intercept=True):

    x = x.dropna().astype(float)
    for col in x.columns:
        if len(set(x[col].values)) == 1:
            x = x.drop(columns=[col])
    if x.shape[1] == 0:
        return None
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


def make_df(dict_to_use, make_symmetric, from_dict_orient='columns'):
    dict_of_dfs = dict()
    for k_tup in dict_to_use:
        tmp = dict_to_use[k_tup]
        new_tmp = dict()
        for k1 in tmp:
            for k2, v in tmp[k1].items():
                new_tmp.setdefault(k1, dict())[k2] = v
                if make_symmetric and (k2 not in tmp or k1 not in tmp[k2]):
                    new_tmp.setdefault(k2, dict())[k1] = v
        tmp_df = pd.DataFrame.from_dict(new_tmp, orient=from_dict_orient)
        if make_symmetric:
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
    for (t1, attr_col), reg_sm in inner_reg_futs_dict.items():
        if use_attr:
            gender, age_group, death_group, states_df_type, to_filter, idx = outer_key
            new_k = gender, age_group, death_group, states_df_type, to_filter, t1, idx, attr_col
        else:
            new_k = outer_key
        if reg_sm is not None:
            new_v = reg_sm.params, reg_sm.pvalues, reg_sm.f_pvalue
            filtered_reg_dict[new_k] = new_v
    return filtered_reg_dict


def organize_reg_dict(filtered_reg_dict, separate_s1_s2=True, create_dfs=True, make_symmetric=True,
                      use_attr=False, to_merge=True, assert_key=True, from_dict_orient='columns'):
    if isinstance(filtered_reg_dict, list) and to_merge:
        flatten_reg_dict = flatten_dict(filtered_reg_dict, assert_disjoint=assert_key)
    else:
        flatten_reg_dict = filtered_reg_dict
    if use_attr:
        key_tup_to_use = 3
    else:
        key_tup_to_use = 2

    if separate_s1_s2:
        filtered_reg_dict_s1 = dict()
        filtered_reg_dict_s2 = dict()
        for k, v in tqdm(flatten_reg_dict.items()):
            keys = k[:-key_tup_to_use]
            if use_attr:
                tissue, idx, attr = k[-key_tup_to_use:]
            else:
                tissue, idx = k[-key_tup_to_use:]
            params, variables_pvalues, f_pvalue = v
            if attr not in params:
                continue
            curr_val = params[attr]
            if idx == 0:
                curr_d = filtered_reg_dict_s1
            elif idx == 1:
                curr_d = filtered_reg_dict_s2
            else:
                raise ValueError("Invalid state index")

            if use_attr:
                curr_d.setdefault(keys, dict()).setdefault(tissue, dict())[attr] = curr_val
            else:
                curr_d.setdefault(keys, dict())[tissue] = curr_val

        if create_dfs:
            filtered_reg_dict_s1_dfs = make_df(filtered_reg_dict_s1, make_symmetric, from_dict_orient=from_dict_orient)
            filtered_reg_dict_s2_dfs = make_df(filtered_reg_dict_s2, make_symmetric, from_dict_orient=from_dict_orient)

            return filtered_reg_dict_s1_dfs, filtered_reg_dict_s2_dfs, flatten_reg_dict
        else:
            return filtered_reg_dict_s1, filtered_reg_dict_s2, flatten_reg_dict
    else:
        return flatten_reg_dict

