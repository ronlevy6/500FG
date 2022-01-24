import pandas as pd
import itertools
from scipy.stats import f_oneway, ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from utils import IS_NOTEBOOK
if IS_NOTEBOOK:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

def get_closest_crtl(curr_crtl_df, trt_times):
    """
    fetch the control data which fits most the treatment times
    """
    if len(set(curr_crtl_df.pert_time) & trt_times) > 0:
        return curr_crtl_df[curr_crtl_df.pert_time.isin(trt_times)]
    else:
        times_to_query = []
        for time in trt_times:
            curr_best = \
                sorted(itertools.product(curr_crtl_df.pert_time.unique(), [time]), key=lambda t: abs(t[0] - t[1]))[0]
            times_to_query.append(curr_best[0])
        return curr_crtl_df[curr_crtl_df.pert_time.isin(times_to_query)]


def test_and_plot_pert_states(cell_id, pert, curr_trt_df, crtl_df, cols_to_comp, to_save=None, to_show=False,
                              specific_vars_to_ret=None, calc_anova=False, calc_t_test=False, ):
    """
    calculates ANOVA and t-test to find difference significance between control samples from same tissues as treatment
    dataset.
    In addition, plots the results
    """
    #     curr_trt_df = curr_trt_df.reset_index().set_index(["cell_id","pert_iname_x"])
    #     curr_crtl_df = crtl_df.loc[crtl_df.index.get_level_values("cell_id")==cell_id]
    #     curr_crtl_df = crtl_df
    curr_crtl_df = get_closest_crtl(crtl_df, set(curr_trt_df.pert_time.values))
    valid_cols_to_comp = set(curr_trt_df.columns) & set(curr_crtl_df.columns) & set(cols_to_comp)

    # test for difference
    anova_res_d = dict()
    t_res_d = dict()
    for col in valid_cols_to_comp:
        if calc_anova:
            anova_res_d[col] = f_oneway(curr_trt_df[col], curr_crtl_df[col])
        if calc_t_test:
            t_res_d[col] = ttest_ind(curr_trt_df[col], curr_crtl_df[col])

    concat = None

    if to_save is not None or to_show:
        # plots
        concat = pd.concat([curr_crtl_df, curr_trt_df])
        if "s1" in concat.columns:
            assert "s2" in concat.columns
            x_col, y_col = "s1", "s2"
        else:
            assert "s1_real_genes" in concat.columns and "s2_real_genes" in concat.columns
            x_col, y_col = "s1_real_genes", "s2_real_genes"

        g = sns.scatterplot(data=concat, x=x_col, y=y_col, size="pert_dose", style="pert_time", hue="pert_type")
        plt.legend(bbox_to_anchor=(1.02, 1), loc=2)
        if x_col in anova_res_d:
            pval_title = '\ns1 pval {:.3f}, s2 pval {:.3f}'.format(-np.log(anova_res_d[x_col].pvalue),
                                                                   -np.log(anova_res_d[y_col].pvalue))
        elif x_col in t_res_d:
            pval_title = '\ns1 pval {:.3f}, s2 pval {:.3f}'.format(-np.log(t_res_d[x_col].pvalue),
                                                                   -np.log(t_res_d[y_col].pvalue))
        else:
            pval_title = ''
        plt.title("{} {}{}".format(cell_id, pert, pval_title))
        if to_save is not None:
            fname = os.path.join(to_save, '{}_{}.jpg'.format(cell_id, pert).replace("/", "_"))
            plt.savefig(fname, bbox_inches='tight')
        if to_show:
            plt.show()
        plt.close()
    ret_d = {'curr_trt_df': curr_trt_df, 'curr_crtl_df': curr_crtl_df, "concat": concat, "anova_res_d": anova_res_d,
             "t_res_d": t_res_d, 'crtl_samples': curr_crtl_df.shape[0], 'trt_samples': curr_trt_df.shape[0]}
    if specific_vars_to_ret is not None:
        to_ret = dict()
        for key in specific_vars_to_ret:
            to_ret[key] = ret_d[key]
        return to_ret
    return ret_d


def make_df_from_res(ret_d, use_t_test=True, use_anova=True, use_numbers=True, use_statistic=True, use_pval=True):
    """
    flat results dict into DataFrame
    """
    d_to_be_df = dict()
    num_d_to_be_df = dict()
    for k in ret_d:
        for prefix, to_use in [("t", use_t_test), ("anova", use_anova)]:
            if to_use:
                test_res_d = ret_d[k]['{}_res_d'.format(prefix)]
                for inner_k, test_res in test_res_d.items():
                    if use_statistic:
                        d_to_be_df.setdefault(k, dict())['{}_{}_stat'.format(inner_k, prefix)] = test_res.statistic
                    if use_pval:
                        d_to_be_df.setdefault(k, dict())['{}_{}_pval'.format(inner_k, prefix)] = test_res.pvalue
        if use_numbers:
            num_d_to_be_df.setdefault(k, dict())['trt_samples'] = ret_d[k]['trt_samples']
            num_d_to_be_df.setdefault(k, dict())['crtl_samples'] = ret_d[k]['crtl_samples']
    print("after loop, before df")
    res_df = pd.DataFrame(d_to_be_df).T
    num_df = pd.DataFrame(num_d_to_be_df).T
    return res_df, num_df


def calc_anova_crtl_trt(metadata_with_states, output_dir, verbose,
                        cols_to_comp=['s1', 's2', 's1_anchor', 's2_anchor', 's1_real_genes', 's2_real_genes'],
                        to_show=False, vars_to_ret=['anova_res_d', 't_res_d', 'crtl_samples', 'trt_samples'],
                        calc_anova=True, calc_t_test=True, to_save=None, overwrite_files=False
                        ):
    """
    calculates ANOVA and t-test values for ALL dataset, flats the results and pickles it
    """
    gb_agg_trt = metadata_with_states.query("pert_type.str.startswith('trt')")
    gb_agg_crtl = metadata_with_states.query("pert_type.str.startswith('ctl')")

    crtl_by_dose_and_time = gb_agg_crtl.fillna(0).groupby(['cell_id', 'pert_iname_x', 'pert_dose', 'pert_time'])
    # sizes_super_elborated = pd.DataFrame(crtl_by_dose_and_time.size())

    crtl = gb_agg_crtl.fillna(0).groupby(['cell_id', 'pert_iname_x'])
    # sizes_less_elborated = pd.DataFrame(crtl.size())

    gb_agg_crtl_reindexed = gb_agg_crtl.reset_index().set_index(["cell_id", "pert_iname_x"])
    gb_agg_crtl_reindexed['pert_type'] = 'control'
    gb_agg_crtl_reindexed.fillna(0, inplace=True)

    gb_agg_crtl_reindexed_final_to_use = gb_agg_crtl_reindexed.reset_index().groupby("cell_id")

    # crtl_df_by_cid = dict()
    # print("crtl d tqdm")
    # for cell_id in tqdm(gb_agg_crtl_reindexed.index.get_level_values("cell_id")):
    #     crtl_df_by_cid[cell_id] = gb_agg_crtl_reindexed.loc[gb_agg_crtl_reindexed.index.get_level_values("cell_id")==cell_id]

    for by_dose in [True, False]:
        if verbose:
            print("by dose-{}".format(by_dose))

        if by_dose:
            trt_gb = gb_agg_trt.groupby(["cell_id", "pert_iname_x", "pert_dose"])
            file_addition = "_by_dose"
            curr_to_save = None  # too many to plot so we pass
        else:
            trt_gb = gb_agg_trt.groupby(["cell_id", "pert_iname_x"])
            file_addition = ""
            curr_to_save = to_save

        if not overwrite_files and \
                os.path.exists(os.path.join(output_dir, 'full_res{}.pickle'.format(file_addition))) and \
                os.path.exists(os.path.join(output_dir, 'full_num_res{}.pickle'.format(file_addition))):
            print("file exists")
            continue

        ret_d = dict()
        for k_trt in tqdm(trt_gb.groups.keys()):
            if by_dose:
                curr_cell_id, curr_pert, curr_pert_dose = k_trt
            else:
                curr_cell_id, curr_pert = k_trt
            curr_trt_df = trt_gb.get_group(k_trt)
            if curr_trt_df.shape[0] == 1:
                continue
            ret_d[k_trt] = test_and_plot_pert_states(curr_cell_id, curr_pert, curr_trt_df,
                                                     gb_agg_crtl_reindexed_final_to_use.get_group(curr_cell_id),
                                                     cols_to_comp, to_show=to_show, specific_vars_to_ret=vars_to_ret,
                                                     calc_anova=calc_anova, calc_t_test=calc_t_test,
                                                     to_save=curr_to_save
                                                     )
        full_res, full_num_res = make_df_from_res(ret_d)
        full_res.to_pickle(os.path.join(output_dir, 'full_res{}.pickle'.format(file_addition)))
        full_num_res.to_pickle(os.path.join(output_dir, 'full_num_res{}.pickle'.format(file_addition)))


def factory(metadata_with_state_path, pert_type_d, output_dir, verbose=True,
            cols_to_comp=['s1', 's2', 's1_anchor', 's2_anchor', 's1_real_genes', 's2_real_genes'],
            to_show=False, calc_anova=True, calc_t_test=True, to_save=None
            ):
    metadata_with_states = pd.read_pickle(metadata_with_state_path)
    metadata_with_states['pert_type'] = metadata_with_states.pert_iname_x.map(pert_type_d)
    #     if verbose:
    #         print("pert_type values")
    #         print(metadata_with_states.pert_type.value_counts())
    calc_anova_crtl_trt(metadata_with_states, output_dir, verbose=verbose, cols_to_comp=cols_to_comp,
                        to_show=to_show, to_save=to_save, calc_anova=calc_anova, calc_t_test=calc_t_test)
