import numpy as np
from tqdm import tqdm
import pickle

def calc_row_col_dist(arr_2d, node_idx, k=50, with_distance=False,axis=0):
    if isinstance(arr_2d, str):
        with open(arr_2d, 'rb') as f:
            arr_2d = pickle.load(f)

    dist_2 = np.sum((arr_2d - arr_2d[node_idx])**2, axis=axis)
    idx_dist_lst = sorted([(i, dist) for i, dist in enumerate(dist_2)],key=lambda x:x[1])
    if with_distance:
        return idx_dist_lst[:k]
    else:
        return [x[0] for x in idx_dist_lst[:k]]


def calc_dist_in_df(df, by_idx=True, k=50, with_distance=False, executor=None):
    arr_2d = df.to_numpy()
    dist_lsts = dict()
    if by_idx:
        vals_to_check = df.index
        axis=1
    else:
        vals_to_check = df.columns
        axis=0
    for idx, val in tqdm(enumerate(vals_to_check)):
        if executor is not None:
            nearest_points = executor.submit(calc_row_col_dist,arr_2d, idx, axis=axis, k=k, with_distance=with_distance)
        else:
            nearest_points = calc_row_col_dist(arr_2d, idx, axis=axis, k=k, with_distance=with_distance)
        dist_lsts[val] = nearest_points
    return dist_lsts


def smooth_data(df, idx, nns_data):
    nns = nns_data[idx]
    fdf = df.iloc[nns]
    return fdf.mean()
