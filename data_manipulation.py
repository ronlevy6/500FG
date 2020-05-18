import numpy as np
from tqdm import tqdm
import pickle
import pandas as pd

from utils import create_tmp_file, rename_patient_name, apply_angle


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


def calc_dist_in_df(df, by_idx=True, k=None, with_distance=False, executor=None, tmp_dir=None):
    tmp = None
    arr_2d = df.to_numpy()


    if executor is not None:
        assert tmp_dir is not None
        tmp = create_tmp_file(tmp_dir, 'pickle')
        f = open(tmp, 'wb')
        pickle.dump(arr_2d, f)
        f.close()

    dist_lsts = dict()
    if by_idx:
        if k is None:
            k = df.shape[0]
        vals_to_check = df.index
        axis=1
    else:
        if k is None:
            k = df.shape[1]
        vals_to_check = df.columns
        axis=0
    for idx, val in tqdm(enumerate(vals_to_check)):
        if executor is not None:
            nearest_points = executor.submit(calc_row_col_dist, tmp, idx, axis=axis, k=k, with_distance=with_distance)
        else:
            nearest_points = calc_row_col_dist(arr_2d, idx, axis=axis, k=k, with_distance=with_distance)
        dist_lsts[val] = nearest_points
    return dist_lsts, tmp


def smooth_data(df, idx, nns_data, by_name=False):
    nns = nns_data[idx]
    if by_name:
        fdf = df.loc[nns]
    else:
        fdf = df.iloc[nns]
    return fdf.mean()


def norm_gene(row, eps=1):
    row = row - row.mean()
    std = row.std()
    if std == 0:
        divide_by = eps
    else:
        divide_by = std
    row = row / divide_by
    return row


def norm_df(pickle_path, output_path=None):
    curr_df = pd.read_pickle(pickle_path)
    curr_df = curr_df.astype(np.float64)
    rename_patient_name(curr_df)
    normed_df = curr_df.apply(norm_gene,axis=1)
    if output_path is None:
        output_path = pickle_path  # overwrite existing df
    normed_df.to_pickle(output_path)
    return output_path


def get_anchors(space_df, x_col, y_col, center_x, center_y, number_of_sections=16,
                anchor_pct_of_data=0.065, num_of_center_genes=10):
    # anchor_pct_of_data between 0-1, based on lp map

    if space_df[(space_df[x_col] == center_x) & (space_df[y_col] == center_y)].shape[0] == 0:
        # adding the center of the space
        center_df = pd.DataFrame(data=[center_x, center_y], columns=['center'], index=[x_col, y_col]).transpose()
        space_df = space_df.append(center_df)

    space_df['point_angle'] = apply_angle(space_df, x_col, y_col, center_x, center_y)
    space_df['section_num'] = space_df['point_angle'] // (360 / number_of_sections)
    space_df['overall_idx'] = list(range(space_df.shape[0]))
    center_idx = space_df[(space_df[x_col] == center_x) & (space_df[y_col] == center_y)]['overall_idx']

    dist_from_center = calc_row_col_dist(space_df[[x_col, y_col]].to_numpy(), center_idx,
                                         k=space_df.shape[0], with_distance=True, axis=1)
    dist_from_center_by_idx = [v[1] for v in sorted(dist_from_center)]
    space_df['dist_from_center'] = dist_from_center_by_idx

    center_genes_idx = [v[0] for v in dist_from_center[1:num_of_center_genes+1]]
    anchor_genes_idx = list()

    for map_section in space_df['section_num'].unique():
        curr_df = space_df[space_df['section_num'] == map_section]
        curr_df.sort_values(by=['dist_from_center'], inplace=True, ascending=False)
        anchor_genes_idx += curr_df.head(n=int(curr_df.shape[0] * anchor_pct_of_data))['overall_idx'].to_list()

    return center_genes_idx, anchor_genes_idx
