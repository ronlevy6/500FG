import pandas as pd
import numpy as np
from numpy.linalg import lstsq
from collections import Counter
from sklearn.linear_model import LinearRegression

from data_manipulation import fit_GE_to_space


def calc_states(tissue_dict, pca_df, fit_to_500FG=True, to_filter=None):
    states_by_all = dict()
    for tissue in tissue_dict:
        for sub_tissue in tissue_dict[tissue]:
            df = tissue_dict[tissue][sub_tissue]
            if to_filter is not None:
                pca_df = pca_df.loc[to_filter]
            x1 = pca_df['pc1']
            x2 = pca_df['pc2']
            if fit_to_500FG:
                df = fit_GE_to_space(df, x1.index)

            x1 = x1.loc[df.index.get_level_values('name')]
            x2 = x2.loc[df.index.get_level_values('name')]

            A1 = np.vstack([x1, np.zeros(len(x1))]).T  # to make sure no intercept
            A2 = np.vstack([x2, np.zeros(len(x2))]).T  # to make sure no intercept

            for ind in df.columns:
                y = df[ind]
                res1 = lstsq(A1, y, rcond=None)
                res2 = lstsq(A2, y, rcond=None)
                coef1 = res1[0]
                coef2 = res2[0]
                assert coef1[1] == coef2[1] == 0
                if ind not in states_by_all:
                    states_by_all[ind] = dict()
                states_by_all[ind][sub_tissue] = (coef1[0], coef2[0])
    return states_by_all


def handle_pca_df_dups(pca_df, ge_df, ge_index_col):
    dup_genes = pca_df.index[pca_df.index.duplicated(keep=False)].to_list()
    new_df = pd.DataFrame(columns=ge_df.columns)
    for dup_gene, num_of_appearances in Counter(dup_genes).items():
        for insertion in range(num_of_appearances - 1):
            # the minus 1 is because it appears once in the ge_df already
            new_df = new_df.append(ge_df.loc[ge_df.index.get_level_values(ge_index_col).isin([dup_gene])])

    ge_df = ge_df.append(new_df)
    return ge_df


def calc_states_combined(tissue_dict, pca_df, x_col='pc1', y_col='pc2', ge_index_col='name',
                         fit_to_500FG=True, to_filter=None, should_handle_pca_df_dups=True, fit_ge_to_space_kwargs={}):
    pca_df = pca_df[[x_col, y_col]]
    if to_filter is not None:
        if isinstance(to_filter[0], str):
            pca_df = pca_df.loc[to_filter]
        elif isinstance(to_filter[0], int):
            pca_df = pca_df.iloc[to_filter]
        else:
            print(type(to_filter[0]))
            raise
    states_by_all = dict()
    for tissue in tissue_dict:
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
                model = LinearRegression(fit_intercept=False).fit(curr_pca_df, y)
                assert model.intercept_ is None or model.intercept_ == 0
                if ind not in states_by_all:
                    states_by_all[ind] = dict()
                states_by_all[ind][sub_tissue] = (model.coef_[0], model.coef_[1])
    return states_by_all
