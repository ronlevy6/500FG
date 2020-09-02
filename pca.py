from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from plotting_utils import smart_print,plot_df


def normalize(df, is_mean, is_std, is_max, is_biultin):
    if is_biultin:
        assert not is_std
        assert not is_max
        assert not is_mean
        df_normalized = pd.DataFrame(StandardScaler().fit_transform(df), index=df.index, columns=df.columns)

    else:
        if is_mean:
            assert not is_biultin
            g = np.mean
        else:
            g = lambda x: 0

        f = lambda x: 1
        if is_std:
            assert not is_max
            assert not is_biultin
            f = np.std

        if is_max:
            assert not is_std
            assert not is_biultin
            f = np.max
        df_normalized = df.apply(lambda col: (col - g(col)) / f(col), axis='columns')

    return df_normalized


def do_pca(df_normalized):
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(df_normalized)
    principalDf = pd.DataFrame(data=principalComponents
                               , columns=['pc1', 'pc2'])

    return principalDf, pca.explained_variance_ratio_, pca


def run_it_all(df, is_mean=False, is_std=True, is_max=False, is_builtin=False, to_filter=False, to_color=False,
               to_plot=True):
    try:
        normalized_df = normalize(df, is_mean, is_std, is_max, is_builtin)
        normalized_df.dropna(axis=0, inplace=True)
        print("after normalize")
        principalDf, evr, pca = do_pca(normalized_df)
        print("after pca")
        principalDf.index = normalized_df.index
        smart_print(is_mean, is_std, is_max, is_builtin, to_filter, evr)
        if to_plot:
            plot_df(principalDf, x='pc1', y='pc2', to_filter=to_filter, to_color=to_color)
        return principalDf, evr, pca
    except AssertionError as error:
        pass


def pca_tissue(states_df, tissues_to_pca):
    """
    run pca on states df for specific tissue
    """
    ret_d = dict()
    for tissue in tissues_to_pca:
        data = pd.DataFrame(states_df[tissue].dropna())
    #     print(df.shape,data.shape)
        data['s1'] = data[tissue].apply(lambda val: val[0])
        data['s2'] = data[tissue].apply(lambda val: val[1])
        del data[tissue]
        pca = PCA(n_components=1)
        principalComponents = pca.fit_transform(data)
        principalDf = pd.DataFrame(data=principalComponents, columns=[tissue], index=data.index)
        ret_d[tissue] = principalDf, pca
    return ret_d


def merge_tissue_pcas(pca_d, patient_lst, min_explained_variance_ratio=0.3):
    res_df = pd.DataFrame(columns=list(pca_d.keys()), index=patient_lst)
    for tissue, (curr_pca_df, curr_pca) in pca_d.items():
        if curr_pca.explained_variance_ratio_ > min_explained_variance_ratio:
            res_df[tissue] = curr_pca_df
    return res_df
