import pandas as pd
import statsmodels.api as sm
from utils import IS_NOTEBOOK
if IS_NOTEBOOK:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def fit_tissues(x, y, fit_intercept=True):

    x = x.dropna()
    for col in x.columns:
        if len(set(x[col].values)) == 1:
            x = x.drop(columns=[col])
    y = y.dropna()

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


def filter_and_organize_reg_dict(reg_dict, separate_s1_s2=True, create_dfs=True, make_symmetric=True):
    filtered_reg_dict = dict()
    for k, v in tqdm(reg_dict.items()):
        if v is not None:
            new_v = v.params, v.pvalues, v.f_pvalue
            filtered_reg_dict[k] = new_v
    if separate_s1_s2:
        filtered_reg_dict_s1 = dict()
        filtered_reg_dict_s2 = dict()
        for k, v in tqdm(filtered_reg_dict.items()):
            keys = k[:-3]
            t1, t2, idx = k[-3:]
            params, variables_pvalues, f_pvalue = v
            if t1 not in params and t2 not in params:
                continue
            try:
                curr_val = params[t1]
            except KeyError:
                curr_val = params[t2]
            if idx == 0:
                filtered_reg_dict_s1.setdefault(keys, dict()).setdefault(t1, dict())[t2] = curr_val
            else:
                filtered_reg_dict_s2.setdefault(keys, dict()).setdefault(t1, dict())[t2] = curr_val

        if create_dfs:
            filtered_reg_dict_s1_dfs = make_df(filtered_reg_dict_s1, make_symmetric)
            filtered_reg_dict_s2_dfs = make_df(filtered_reg_dict_s2, make_symmetric)
            return filtered_reg_dict_s1_dfs, filtered_reg_dict_s2_dfs
        else:
            return filtered_reg_dict_s1, filtered_reg_dict_s2
    else:
        return filtered_reg_dict

