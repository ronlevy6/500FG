import pandas as pd
from collections import Counter
from sklearn.linear_model import LinearRegression
from data_manipulation import fit_GE_to_space
from utils import IS_NOTEBOOK
if IS_NOTEBOOK:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def handle_pca_df_dups(pca_df, ge_df, ge_index_col):
    dup_genes = pca_df.index[pca_df.index.duplicated(keep=False)].to_list()
    new_df = pd.DataFrame(columns=ge_df.columns)
    for dup_gene, num_of_appearances in Counter(dup_genes).items():
        for insertion in range(num_of_appearances - 1):
            # the minus 1 is because it appears once in the ge_df already
            new_df = new_df.append(ge_df.loc[ge_df.index.get_level_values(ge_index_col).isin([dup_gene])])

    ge_df = ge_df.append(new_df)
    return ge_df


def filter_df(df, to_filter):
    if isinstance(to_filter[0], str):
        df = df.loc[to_filter]
    elif isinstance(to_filter[0], int):
        df = df.iloc[to_filter]
    else:
        print(type(to_filter[0]))
        raise
    return df


def calc_states(tissue_dict, pca_df, x_col='pc1', y_col='pc2', ge_index_col='name',
                fit_to_500FG=True, to_filter=None, should_handle_pca_df_dups=True, fit_ge_to_space_kwargs={}):
    if to_filter is not None:
        pca_df = filter_df(pca_df, to_filter)
    states_by_all = dict()
    for tissue in tqdm(tissue_dict):
        for sub_tissue in tissue_dict[tissue]:
            df = tissue_dict[tissue][sub_tissue]

            if fit_to_500FG:
                df = fit_GE_to_space(df, pca_df.index, **fit_ge_to_space_kwargs)

            vals_to_use = list(set(df.index.get_level_values(ge_index_col)) & set(pca_df.index))
            curr_pca_df = pca_df.loc[vals_to_use].dropna().sort_index()
            if should_handle_pca_df_dups:
                df = handle_pca_df_dups(curr_pca_df, df, ge_index_col)
            df = df[df.index.get_level_values(ge_index_col).isin(vals_to_use)]
            df.sort_values(by=[ge_index_col], inplace=True)
            for ind in df.columns:
                y = df[ind]
                assert (y.index.get_level_values(ge_index_col) == curr_pca_df.index).all()
                model = LinearRegression(fit_intercept=False).fit(curr_pca_df[[x_col, y_col]], y)
                assert model.intercept_ is None or model.intercept_ == 0
                if ind not in states_by_all:
                    states_by_all[ind] = dict()
                states_by_all[ind][sub_tissue] = (model.coef_[0], model.coef_[1])
    return states_by_all


def calc_states_combined(tissue_dict, pca_df, x_col='pc1', y_col='pc2', ge_index_col='name',
                         to_filter_pca_df=None, need_to_merge=False, query_to_filter_merged_df=None):

    if to_filter_pca_df is not None:
        pca_df = filter_df(pca_df, to_filter_pca_df)

    states_by_all = dict()
    for tissue in tqdm(tissue_dict):
        for sub_tissue in tissue_dict[tissue]:
            df = tissue_dict[tissue][sub_tissue]
            if need_to_merge:
                df_with_pca = pca_df.merge(df, left_index=True, right_on=ge_index_col)
            else:
                df_with_pca = df
            if query_to_filter_merged_df:
                df_with_pca = df_with_pca.query(query_to_filter_merged_df)
            for ind in df.columns:  # use individuals from original GE df
                y = df_with_pca[ind]
                model = LinearRegression(fit_intercept=False).fit(df_with_pca[[x_col, y_col]], y)
                assert model.intercept_ is None or model.intercept_ == 0
                if ind not in states_by_all:
                    states_by_all[ind] = dict()
                states_by_all[ind][sub_tissue] = (model.coef_[0], model.coef_[1])
    return states_by_all
