from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from misc_utils import create_tmp_file, chunks
from utils import apply_angle, rename_patient_name, IS_NOTEBOOK
if IS_NOTEBOOK:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


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


def calc_dist_in_df_cluster(df, by_idx=True, k=None, with_distance=False, executor=None, tmp_dir=None):
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


def calc_dist_in_df(df, subset_cols=[], metric='euclidean', return_df=True, col_names=True, idx_names=False):
    if len(subset_cols):
        np_arr = df[subset_cols].to_numpy()
    else:
        np_arr = df.to_numpy()
    distances = pdist(np_arr, metric=metric)
    dist_mat = squareform(distances)
    if return_df:
        dist_mat = pd.DataFrame(dist_mat)
        if col_names:
            dist_mat.columns = df.index.tolist()
        if idx_names:
            dist_mat.index = df.index.tolist()
    return dist_mat


def smooth_data_by_df(df, idx, nns_data, by_name=False):
    nns = nns_data[idx]
    if by_name:
        fdf = df.loc[nns]
    else:
        fdf = df.iloc[nns]
    return fdf.mean()


def get_gene_closest(gene_dist, k, max_dist=1000):
    """
    :param gene_dist: A series with distances from gene
    :param k: number of nearest neighbors to return
    :param max_dist: max valid distance
    :return: knn
    """
    sorted_by_dist = gene_dist.sort_values().iloc[:k]
    return sorted_by_dist[sorted_by_dist < max_dist]


def get_closest(dist_df, k=None, max_dist=0.1):
    if k is None:
        k = int(dist_df.shape[0]*0.01)  # 1% of data
    closet_genes_dict = dict()
    for col_num in tqdm(range(len(dist_df.columns))):
        res = get_gene_closest(dist_df.iloc[:, col_num], k=k, max_dist=max_dist)
        closet_genes_dict[col_num] = res
    return closet_genes_dict, isinstance(res[0], int)


def smooth_data(ge_dict, pca_space, dist_df, x_col, y_col, col_to_change_to_original=['is_far', 'early', 'late'],
                need_to_merge=True, validate_distance_df=False, ge_index_col='name', get_closest_kwargs={}):
    if dist_df is not None:
        closet_genes_dict, closest_by_idx = get_closest(dist_df, **get_closest_kwargs)
    smoothen_ge_space_data = dict()
    for tissue in tqdm(ge_dict):
        for sub_tissue in ge_dict[tissue]:
            smoothen_ge_space_data.setdefault(tissue, dict())[sub_tissue] = dict()

            # The merge is done to keep order
            curr_ge_df = ge_dict[tissue][sub_tissue]
            if need_to_merge:
                pca_space_with_ge = pca_space.merge(curr_ge_df, left_index=True, right_on='name')
            else:
                pca_space_with_ge = curr_ge_df
            if dist_df is None:
                dist_df = calc_dist_in_df(pca_space_with_ge, [x_col, y_col], idx_names=False)
                closet_genes_dict, closest_by_idx = get_closest(dist_df, **get_closest_kwargs)

            # fill in dict is faster than DataFrame
            smoothen_d = dict()
            if validate_distance_df:
                # SLOW!!
                curr_dist_df = calc_dist_in_df(pca_space_with_ge, [x_col, y_col,], idx_names=False)
                curr_closest_genes_dict = get_closest(curr_dist_df, **get_closest_kwargs)
                assert pca_space_with_ge.index.tolist() == curr_dist_df.columns.tolist()
                assert (dist_df.values.flatten() == curr_dist_df.values.flatten()).all()
                if isinstance(dist_df[col], tuple):
                    assert [col[1] for col in dist_df.columns] == [col[1] for col in curr_dist_df.columns]
                else:
                    assert [col for col in dist_df.columns] == [col for col in curr_dist_df.columns]
                assert curr_closest_genes_dict == closet_genes_dict
            if closest_by_idx:
                # only need to make sure the order is the same when comparing by indices
                assert pca_space_with_ge.index.get_level_values(ge_index_col).tolist() == \
                       [col[1] if isinstance(col, tuple) else col for col in dist_df.columns]  #fit case columns aren't tuple
            # The smoothing itself
            for col_num in tqdm(range(len(dist_df.columns))):
                res = closet_genes_dict[col_num]
                if closest_by_idx:
                    smoothen_d[col_num] = pca_space_with_ge.iloc[res.index].mean()
                else:
                    smoothen_d[col_num] = \
                        pca_space_with_ge[pca_space_with_ge.index.get_level_values(ge_index_col).isin(res.index)].mean()

            smoothen_df_t = pd.DataFrame(smoothen_d)
            smoothen_df_t.columns = dist_df.columns
            for col in col_to_change_to_original:
                smoothen_df_t.loc[col] = pca_space_with_ge[col].values.tolist()
            smoothen_ge_space_data[tissue][sub_tissue] = smoothen_df_t.transpose()
    return smoothen_ge_space_data


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

    # extract gene names
    space_df_genes = list(space_df.index)
    anchor_gene_names = [space_df_genes[idx] for idx in anchor_genes_idx]
    center_gene_names = [space_df_genes[idx] for idx in center_genes_idx]

    return center_genes_idx, center_gene_names, anchor_genes_idx, anchor_gene_names


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


def fit_GE_to_space(ge_df, space_genes, index_names=('symbol', 'name'), ge_index_col='name', handle_variants=True):
    """
    filter genes in GE dataframe that don't appear in the space and in case of variants keeps the highest one
    :return:
    """
    ge_df.set_index(pd.MultiIndex.from_tuples(ge_df.index, names=index_names), inplace=True)
    ge_df = ge_df[ge_df.index.get_level_values(ge_index_col).isin(space_genes)]  # take from GE only genes in PCA
    many_times = [x[0] for x in Counter(ge_df.index.get_level_values(ge_index_col)).items() if x[1] > 1]
    if handle_variants:
        for gene in many_times:
            med = ge_df[ge_df.index.get_level_values(ge_index_col) == gene].median(axis=1)
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
            try:
                df = pd.DataFrame(d[tissue][sub_tissue][ind])
            except KeyError:
                return None
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
                if stacked is None:
                    continue
                if curr_df is None:
                    curr_df = stacked
                else:
                    curr_df = curr_df.append(stacked)
        curr_df_by_ind[ind] = curr_df
    return curr_df_by_ind


def bootstrap(d, iter_num=50, bootstrap_pct=0.8, min_vals=10):
    # the keys of d is two tissues
    not_enough = 0
    bootstrap_res = dict()
    for k, val in tqdm(d.items()):
        if k[0] == k[1]:
            continue
        # Fit case of nested dict
        if isinstance(val, dict):
            curr_lst = list(val.values())
        else:
            curr_lst = list(val)
        if len(curr_lst) < min_vals:
            not_enough += 1
            continue
        for i in range(iter_num):
            np.random.shuffle(curr_lst)
            lst_to_use = curr_lst[:int(len(curr_lst)*bootstrap_pct)]
            bootstrap_res.setdefault(k, list()).append((np.mean(lst_to_use), np.std(lst_to_use)))
    return bootstrap_res, not_enough


def var_analysis_tmp_df(x_data, y_data, min_num_of_points, x_label, y_label, fill_missing_vals=False):
    tmp_x = x_data.dropna(thresh=min_num_of_points, axis=0).var(axis=1).sort_index()
    tmp_y = y_data.dropna(thresh=min_num_of_points, axis=0).var(axis=1).sort_index()
    if list(tmp_x.index) != list(tmp_y.index):
        if fill_missing_vals:
            # one side
            missing_idx = set(tmp_y.index) - set(tmp_x.index)
            tmp_x = tmp_x.append(pd.Series(index=missing_idx, data=[0] * len(missing_idx)))
            # other side
            missing_idx = set(tmp_x.index) - set(tmp_y.index)
            tmp_y = tmp_y.append(pd.Series(index=missing_idx, data=[0] * len(missing_idx)))
        else:
            shared = set(tmp_x.index) & set(tmp_y.index)
            tmp_x = tmp_x.loc[shared]
            tmp_y = tmp_y.loc[shared]
    tmp_x.sort_index(inplace=True)
    tmp_y.sort_index(inplace=True)
    assert list(tmp_x.index) == list(tmp_y.index)
    tmp_df = pd.DataFrame(data=[tmp_x.index, tmp_x.values, tmp_y.values]).transpose()
    tmp_df.columns = ['tissue', x_label, y_label]
    return tmp_df


def var_analysis(x_data, y_data, patient_data, x_label='x', y_label='y', col_to_sort_by=None, chunk_size=10,
                 min_num_of_points=10, title=None, fill_missing_vals=False, to_plot=True, to_save=None, to_show=True):
    full_markers = ['o', '<', '>']
    # Prepare the variance df
    df_to_use = None
    if col_to_sort_by is not None:
        curr_markers = dict()
        for idx, val in enumerate(patient_data[col_to_sort_by].unique()):
            curr_markers[val] = full_markers[idx]

        #         curr_markers = full_markers[:len(patient_data[col_to_sort_by].unique())]
        for col_val in patient_data[col_to_sort_by].unique().tolist():
            curr_patients = patient_data[patient_data[col_to_sort_by] == col_val].index.tolist()
            tmp_df = var_analysis_tmp_df(x_data[curr_patients], y_data[curr_patients],
                                         min_num_of_points, x_label, y_label, fill_missing_vals)
            tmp_df[col_to_sort_by] = col_val

            if df_to_use is None:
                df_to_use = tmp_df
            else:
                df_to_use = df_to_use.append(tmp_df)
    else:
        df_to_use = var_analysis_tmp_df(x_data, y_data, min_num_of_points, x_label, y_label,
                                        fill_missing_vals=fill_missing_vals)
        df_to_use['col'] = 'all'

    tissues = sorted(df_to_use['tissue'].unique())
    max_x = df_to_use[x_label].max() * 1.2
    max_y = df_to_use[y_label].max() * 1.2
    if to_plot:
        for chunk_idx, chunk in enumerate(chunks(tissues, chunk_size)):
            curr_df_to_plot = df_to_use[df_to_use['tissue'].isin(chunk)]
            if col_to_sort_by is None:
                sns.scatterplot(data=curr_df_to_plot, x=x_label, y=y_label, hue='tissue')
            else:
                sns.scatterplot(data=curr_df_to_plot, x=x_label, y=y_label, hue='tissue', style=col_to_sort_by,
                                markers=curr_markers, )

            plt.xlim(0, max_x)
            plt.ylim(0, max_y)
            plt.legend(bbox_to_anchor=(1.1, 1.05))
            if title is not None:
                plt.title(title + ' chuck {}'.format(chunk_idx))
            if to_save is not None:
                plt.savefig(to_save + '_chunk {}.jpg'.format(chunk_idx), bbox_inches='tight')
            if to_show:
                plt.show()
            plt.close()
    return df_to_use
