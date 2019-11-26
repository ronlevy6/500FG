from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


def normalize(df, is_mean, is_std, is_max, is_biultin):
    if is_biultin:
        assert not is_std
        assert not is_max
        assert not is_mean
        df_normalized = StandardScaler().fit_transform(df)

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

        #         df_normalized = pd.DataFrame(columns=df.columns, index=df.index)

        #         for col in tqdm(df.columns):
        #             vals = df[col]
        #             std_val = f(vals)
        #             mean_val = g(vals)

        #             df_normalized[col] = (vals-mean_val)/std_val
        df_normalized = df.apply(lambda col: (col - g(col)) / f(col), axis='columns')

    return df_normalized


def do_pca(df_normalized):
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(df_normalized)
    principalDf = pd.DataFrame(data=principalComponents
                               , columns=['pc1', 'pc2'])

    return principalDf, pca.explained_variance_ratio_

