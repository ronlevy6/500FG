import pandas as pd
import numpy as np
from numpy.linalg import lstsq
from collections import Counter
from sklearn.linear_model import LinearRegression


def fit_GE_to_500fg(ge_df, fg500_genes):
    ge_df.set_index(pd.MultiIndex.from_tuples(ge_df.index, names=('symbol', 'name')), inplace=True)
    ge_df = ge_df[ge_df.index.get_level_values('name').isin(fg500_genes)]  # take from GE only genes in PCA
    many_times = [x[0] for x in Counter(ge_df.index.get_level_values('name')).items() if x[1] > 1]
    for gene in many_times:
        med = ge_df[ge_df.index.get_level_values('name') == gene].median(axis=1)
        to_del = med.sort_values().index[0]
        ge_df.drop(to_del, inplace=True)
    return ge_df


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
                df = fit_GE_to_500fg(df, x1.index)

            x1 = x1[x1.index.isin(df.index.get_level_values('name'))]
            x2 = x2[x2.index.isin(df.index.get_level_values('name'))]
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


def calc_states_combined(tissue_dict, pca_df, fit_to_500FG=True, to_filter=None):
    pca_df = pca_df[['pc1','pc2']]
    states_by_all = dict()
    for tissue in tissue_dict:
        for sub_tissue in tissue_dict[tissue]:
            df = tissue_dict[tissue][sub_tissue]
            if to_filter is not None:
                pca_df = pca_df.loc[to_filter]
            if fit_to_500FG:
                df = fit_GE_to_500fg(df, pca_df.index)

            curr_pca_df=pca_df[pca_df.index.isin(df.index.get_level_values('name'))]

            for ind in df.columns:
                y = df[ind]
                model = LinearRegression(fit_intercept=False).fit(curr_pca_df,y)
                assert model.intercept_ is None or model.intercept_ == 0
                if ind not in states_by_all:
                    states_by_all[ind] = dict()
                states_by_all[ind][sub_tissue] = (model.coef_[0],model.coef_[1])
    return states_by_all
