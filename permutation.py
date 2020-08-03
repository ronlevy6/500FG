import pandas as pd
from random import shuffle
from utils import IS_NOTEBOOK
if IS_NOTEBOOK:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def permute_idx(orig_df, verbose=False):
    if verbose:
        print("permute_idx")
    original_idx = orig_df.index.tolist()
    permutated_idx = original_idx.copy()
    shuffle(permutated_idx)
    assert not permutated_idx == original_idx
    assert set(permutated_idx) == set(original_idx)
    return permutated_idx


def permute_df(orig_df, permutated_idx, multi_index=True, index_names=['symbol', 'name']):
    perm_df = orig_df.copy()
    if multi_index:
        perm_df.set_index(pd.MultiIndex.from_tuples(permutated_idx, names=index_names), inplace=True)
    else:
        perm_df.set_index(pd.Index(permutated_idx), inplace=True)
    return perm_df


def permute_all_ge_data(ge_dict, permutated_idx=None, verbose=False, multi_index=True, index_names=['symbol', 'name']):
    perm_ge_dict = dict()
    for tissue in tqdm(ge_dict):
        for sub_tissue in ge_dict[tissue]:
            orig_df = ge_dict[tissue][sub_tissue]
            if permutated_idx is None:
                permutated_idx = permute_idx(orig_df, verbose=verbose)
            perm_df = permute_df(orig_df, permutated_idx, multi_index=multi_index, index_names=index_names)
            perm_ge_dict.setdefault(tissue, dict())[sub_tissue] = perm_df
    return perm_ge_dict, permutated_idx


def permute_df_from_scratch(df, multi_index=False, index_names=None):
    perm_idx = permute_idx(df)
    perm_df = permute_df(df, perm_idx, multi_index=multi_index, index_names=index_names)
    return perm_df, perm_idx
