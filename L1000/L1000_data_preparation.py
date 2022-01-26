import numpy as np
import pandas as pd
import os
from utils import IS_NOTEBOOK
from sklearn.preprocessing import StandardScaler
if IS_NOTEBOOK:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def normalize_by_cell_id(crtl_key, crtl_df, trt_cell_id_d, with_mean=True, with_std=True):
    ret_d = dict()

    scaler1 = StandardScaler(with_mean=with_mean, with_std=with_std)
    scaler2 = StandardScaler(with_mean=with_mean, with_std=with_std)

    ref_data_scaled = pd.DataFrame(scaler1.fit_transform(crtl_df),
                                   index=crtl_df.index, columns=crtl_df.columns)
    scaler2.fit(ref_data_scaled.T)

    for trt_g, trt_g_df in tqdm(trt_cell_id_d.items()):
        data_scaled = pd.DataFrame(scaler1.fit_transform(trt_g_df), index=trt_g_df.index, columns=trt_g_df.columns)
        data_scaled = pd.DataFrame(scaler2.transform(data_scaled.T).T,
                                   index=data_scaled.index, columns=data_scaled.columns)
        ret_d[trt_g] = data_scaled

    ref_data_scaled = pd.DataFrame(scaler2.transform(ref_data_scaled.T).T,
                                   index=ref_data_scaled.index, columns=ref_data_scaled.columns)
    ret_d[crtl_key] = ref_data_scaled
    return ret_d


def pickle_d_of_futs(outer_k, d_of_futs, dir_to_pickle, filename):
    """
    re-concating the normalized datasset
    """
    ret_d = dict()
    for k, df in d_of_futs.items():
        #         ret_d[k] = outer_k[0], outer_k[1], k[0], k[1]
        ret_d[k] = pickle_fut_df(df, dir_to_pickle, filename,
                                 filename_format=[outer_k[0], outer_k[1].replace("'", "").replace("/", ""),
                                                  k[0], k[1].replace("'", "").replace("/", "")])
    return ret_d


def pickle_fut_df(fut_df, dir_to_pickle, filename, filename_format):
    full_path_to_pickle = os.path.join(dir_to_pickle, filename.format(*filename_format))
    assert not os.path.exists(full_path_to_pickle)
    fut_df.to_pickle(full_path_to_pickle)
    return full_path_to_pickle


def make_full_dirname(log_mode, std_mode, real_gene_mode, homedir):
    if log_mode:
        log_text = '_log'
    else:
        log_text = ''
    if std_mode:
        std_text = ''
    else:
        std_text = '_no_std'
    if real_gene_mode:
        real_genes_text = '_real_genes'
    else:
        real_genes_text = ''
    inner_dir = 'normalized{}{}_data{}'.format(log_text, std_text, real_genes_text)
    return os.path.join(homedir, inner_dir)


def concat_files(input_dir, input_files, output_file):
    dfs = []
    for f in input_files:
        curr_df = pd.read_pickle(os.path.join(input_dir, f))
        dfs.append(curr_df)
    super_df = pd.concat(dfs, axis=1)
    super_df.to_pickle(output_file)
    return super_df.shape


def flat_d(d):
    flat_d = dict()
    for inner_d in tqdm(d.values()):
        for k in inner_d:
            assert k not in flat_d
            flat_d[k] = dict()
            flat_d[k]['s1'] = inner_d[k][0]
            flat_d[k]['s2'] = inner_d[k][1]
    return flat_d


def decapitalize(s):
    if not s:  # check that s is not empty string
        return s
    s_lst = s.split(" ")
    s_lst = [v[0].lower() + v[1:] for v in s_lst]
    return str.join("-", s_lst)


def make_pretty_tab(curr_pert, full_res_by_dose, full_num_res_by_dose, full_res, full_num_res, all_corrs,
                    query="pert_iname_x=='{}'", filter_regex='real_genes_t_pva',
                    log_pval_func=lambda val: -np.log10(val), shorten_cols=True, rename_cols=True):
    """
    format results of a perturbator to a single table
    """
    query = query.format(curr_pert)
    pretty_tab = full_res_by_dose.query(query).filter(regex=filter_regex)
    cols_to_change = [col for col in pretty_tab.columns if "pval" in col]
    pretty_tab[cols_to_change] = pretty_tab[cols_to_change].applymap(log_pval_func)
    pretty_tab = pretty_tab.merge(full_num_res_by_dose.query(query)['trt_samples'], left_index=True, right_index=True)

    no_dose_calc = full_res.query(query).filter(regex=filter_regex)
    cols_to_change = [col for col in no_dose_calc.columns if "pval" in col]
    no_dose_calc[cols_to_change] = no_dose_calc[cols_to_change].applymap(log_pval_func)
    no_dose_calc = no_dose_calc.merge(full_num_res.query(query)['trt_samples'], left_index=True, right_index=True)

    no_dose_calc['pert_dose'] = 'combined'
    no_dose_calc.set_index(['pert_dose'], append=True, inplace=True)

    corrs_df = all_corrs.query(query)
    corrs_df['pert_dose'] = 'correlation'
    corrs_df.set_index(['pert_dose'], append=True, inplace=True)

    if shorten_cols or rename_cols:
        assert not shorten_cols and rename_cols
        for df in [pretty_tab, no_dose_calc, corrs_df]:
            if shorten_cols:
                col_names = [col[:13] for col in df.columns]  # len("s1_real_genes")
            if rename_cols:
                col_names = [col[:13] if "t_stat" in col else col for col in df.columns]
            df.columns = col_names

    pretty_tab = pretty_tab.append(no_dose_calc)
    pretty_tab = pretty_tab.sort_values(by=['cell_id', 'pert_dose'], )

    return pretty_tab, corrs_df


def gather_data(pert, factory_res_d_to_use, metadata_with_states_d_to_use, sum_up_tabs_to_use):
    pretty_data_d = dict()
    raw_states_d = dict()

    pert_sum_up, corr_sum_up = make_pretty_tab(pert, factory_res_d_to_use['full_res_by_dose'],
                                               factory_res_d_to_use['full_num_res_by_dose'],
                                               factory_res_d_to_use['full_res'],
                                               factory_res_d_to_use['full_num_res'],
                                               sum_up_tabs_to_use['all_corrs'],
                                               filter_regex='real_genes_t', shorten_cols=False)

    pretty_data_d[pert] = pert_sum_up, corr_sum_up

    raw_states_d[pert] = metadata_with_states_d_to_use['trt'].query("pert_iname_x=='{}'".format(pert))

    pert_analysis_summary, pert_analysis_corr = pretty_data_d[pert]

    pert_analysis_summary['dosage'] = pert_analysis_summary.index.get_level_values("pert_dose")
    pert_analysis_summary['dosage'].replace({"combined": -1}, inplace=True)
    pert_analysis_summary['row_type'] = ['overall' if v == -1 else 'by_dose' for v in pert_analysis_summary.dosage]
    replace_d = pert_analysis_summary.groupby(["cell_id"])['dosage'].max().to_dict()

    return pert_analysis_summary, pert_analysis_corr, raw_states_d[pert], replace_d
