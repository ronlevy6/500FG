from collections import Counter

import numpy as np
from tqdm import tqdm
import pickle
import pandas as pd

from misc_utils import create_tmp_file
from utils import apply_angle, rename_patient_name


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


def filter_patient_df(patient_df, filtering_dict, verbose=False):
    """
    :param patient_df: has the columns SEX, AGE, DTHHRDY, sex_name, death_reason, and maybe cardiac, cerebrovascular
    :param filtering_dict: key is column in patient_df. value is tuple
    :return:
    """
    filtered_df = patient_df.copy()
    for col, values in filtering_dict.items():
        if col not in patient_df.columns:
            if verbose:
                print("{} Doesen't exist in DF".format(col))
            continue
        filtered_df = filtered_df[filtered_df[col].isin(values)]
    return filtered_df


def find_center_sensitive(space_df, x_col, y_col, num_of_values_to_use=20):
    res = []
    for col in [x_col, y_col]:
        min_point = np.mean(sorted(space_df[col])[:num_of_values_to_use])
        max_point = np.mean(sorted(space_df[col], reverse=True)[:num_of_values_to_use])
        val = max_point + min_point
        res.append(val/2)
    return res


def fit_GE_to_500fg(ge_df, fg500_genes):
    ge_df.set_index(pd.MultiIndex.from_tuples(ge_df.index, names=('symbol', 'name')), inplace=True)
    ge_df = ge_df[ge_df.index.get_level_values('name').isin(fg500_genes)]  # take from GE only genes in PCA
    many_times = [x[0] for x in Counter(ge_df.index.get_level_values('name')).items() if x[1] > 1]
    for gene in many_times:
        med = ge_df[ge_df.index.get_level_values('name') == gene].median(axis=1)
        to_del = med.sort_values().index[0]
        ge_df.drop(to_del, inplace=True)
    return ge_df


def make_ordered_d(d):
    """
    return keys sorted by values
    """
    ret_d = {k: loc for loc,(k,v) in enumerate(sorted(d.items(),key=lambda x:x[1]))}
    return {k: loc/len(ret_d) for k,loc in ret_d.items()}


def make_as_pct(df, ind):
    """
    per ind - change values to pct when lowest is 0 and hightest is 1
    """
    curr_order = make_ordered_d(df[ind].dropna().to_dict())
    df[ind] = df.index.map(curr_order)


def stack_per_ind(d, sub_tissue, ind, condition, is_anchor, anchor_genes, ge_index_name='name',
                                col_to_stack_by='GE', to_pct=False, ):
    df = None
    for tissue in d:
        if sub_tissue in d[tissue]:
            df = pd.DataFrame(d[tissue][sub_tissue][ind])
    assert df is not None
    if to_pct:
        make_as_pct(df, ind)
    if is_anchor:
        df = df[df.index.get_level_values(ge_index_name).isin(anchor_genes)]
    stacked = pd.DataFrame(df.stack(), columns=[col_to_stack_by])
    stacked['sub'] = '{}'.format(sub_tissue)
    stacked['condition'] = condition
    return stacked


def stack_GE_dict_for_map_plots(patients, ge_data, smoothen_ge_data, anchor_genes, condition_dict, sub_tissues_to_use=None):
    all_sub_tissues = []
    for v in ge_data.values():
        # nested dict
        for sub_tissue in v.keys():
            all_sub_tissues.append(sub_tissue)
    curr_df_by_ind = dict()
    for ind in tqdm(patients):
        curr_df = None
        for sub_tissue in all_sub_tissues:
            if sub_tissues_to_use is not None and sub_tissue not in sub_tissues_to_use:
                continue
            for condition_name, params in condition_dict.items():
                is_anchor = params.get('is_anchor', False)
                to_pct = params.get('to_pct', False)
                ge_index_name = params.get('ge_index_name', 'name')
                col_to_stack_by = params.get('col_to_stack_by', 'GE')

                if params['ge_type'] == 'normal':
                    ge_dict_to_use = ge_data
                elif params['ge_type'] == 'smoothen':
                    ge_dict_to_use = smoothen_ge_data
                else:
                    raise ValueError("Unknown type of ge_type")
                stacked = stack_per_ind(ge_dict_to_use, sub_tissue, ind, condition=condition_name,
                                        is_anchor=is_anchor, to_pct=to_pct, ge_index_name=ge_index_name,
                                        col_to_stack_by=col_to_stack_by, anchor_genes=anchor_genes)
                if curr_df is None:
                    curr_df = stacked
                else:
                    curr_df = curr_df.append(stacked)
        curr_df_by_ind[ind] = curr_df
    return curr_df_by_ind
