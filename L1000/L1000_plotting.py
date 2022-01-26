import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from utils import IS_NOTEBOOK
from L1000.L1000_data_preparation import gather_data
if IS_NOTEBOOK:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def plot_states_t_val(tmp_summary, states_t_val_ax, pert, cid, curr_cid_data_d, x='s1_real_genes', y='s2_real_genes',
                      palette='Paired_r', size='dosage', verbose=False):
    sns.scatterplot(data=tmp_summary, x=x, y=y, palette=palette,
                    size=size, legend='full', hue='row_type', ax=states_t_val_ax)
    states_t_val_ax.legend(bbox_to_anchor=(1.0, 1.07), title="dosage")
    states_t_val_ax.set_xlabel("s1 t-values")
    states_t_val_ax.set_ylabel("s2 t-values")
    states_t_val_ax.set_title("{}\nt-values for cell-{}\n{}-{} from {}".format(pert.capitalize(), cid,
                                                                               curr_cid_data_d['sample_type'],
                                                                               curr_cid_data_d['subtype'],
                                                                               curr_cid_data_d['primary_site']))
    if verbose:
        print("finished plot_states_t_val")


def plot_corr(tmp_summary, tmp_corr, corr_ax, cid, curr_cid_data_d, pert_analysis_summary, verbose=False):
    tmp_df = tmp_summary[['s1_real_genes_t_pval', 's2_real_genes_t_pval']].melt(ignore_index=False)
    sns.barplot(x=tmp_df.index.get_level_values('pert_dose'), y=tmp_df['value'], hue=tmp_df['variable'],
                ax=corr_ax, palette='bwr')
    corr_ax.set_ylabel("-log(pvalue)")
    xlabels = corr_ax.get_xticklabels()
    xlabels2 = [v.get_text()[:6] if v != 'combined' else v for v in xlabels]
    corr_ax.set_xticklabels(xlabels2, rotation=-45)
    corr_ax.axhline(y=-np.log10(0.0025), color='black', linestyle='--', label='-log(0.0025)')

    corr_ax.set_title("ttest-pval for cell-{}\n{}-{} from {}".format(cid, curr_cid_data_d['sample_type'],
                                                                     curr_cid_data_d['subtype'],
                                                                     curr_cid_data_d['primary_site']))

    try:
        c1, c2 = tmp_corr[['s1_real_genes', 's2_real_genes']].values.flatten()
    except ValueError:
        #         print(pert, cid, tmp_corr.shape)
        if pert_analysis_summary.query('cell_id==@cid').shape[0] > 3:
            raise
        else:
            c1, c2 = 0, 0
    corr_ax.text(s="corr: {:.3f}, {:.3f}".format(c1, c2), ha='left', va='top', x=0.05, y=0.95,
                 transform=corr_ax.transAxes, size=12,
                 bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'))
    corr_ax.legend(bbox_to_anchor=(1.02, 1.03), loc='upper left')

    if verbose:
        print("finished plot_corr")


def plot_states(crtl_states_by_cid, tmp_states, states_ax, cid, curr_cid_data_d, x='s1_real_genes', y='s2_real_genes',
                cols_to_take=['cell_id', 'pert_dose', 'pert_iname_x', 'pert_time', 's1_real_genes', 's2_real_genes',
                              'pert_type'], elaborate=False, verbose=False, full_res_to_use=None):
    sns.scatterplot(data=crtl_states_by_cid[cid][cols_to_take], x=x, y=y,
                    label='control', color='red', ax=states_ax, alpha=0.3)

    sns.scatterplot(data=tmp_states[cols_to_take], x=x, y=y, label='trt',
                    size='pert_dose', legend='full', color='darkblue', ax=states_ax, alpha=0.8)

    states_ax.legend(bbox_to_anchor=(1.0, 1.07), title="dosage")
    states_ax.set_xlabel("s1 states")
    states_ax.set_ylabel("s2 states")
    title = "states for cell-{}\n{}-{} from {}".format(cid, curr_cid_data_d['sample_type'],
                                                       curr_cid_data_d['subtype'],
                                                       curr_cid_data_d['primary_site'])
    if elaborate:
        pert = tmp_states['pert_iname_x'].unique()
        assert len(pert) == 1
        pert = pert[0]
        comb_pval_df = full_res_to_use.query("(cell_id==@cid)&(pert_iname_x==@pert)").filter(regex='pval').applymap(
            lambda val: -np.log10(val))
        s1_pval, s2_pval = comb_pval_df[['s1_real_genes_t_pval', 's2_real_genes_t_pval']].values.flatten()
        title = '{}\n{}'.format(pert.capitalize(), title)
        states_ax.text(s="-log(pval): {:.3f}, {:.3f}".format(s1_pval, s2_pval),
                       ha='left', va='top', x=0.05, y=0.95,
                       transform=states_ax.transAxes, size=12,
                       bbox=dict(facecolor='none', edgecolor='black', boxstyle='round'))

    states_ax.set_title(title)
    if verbose:
        print("finished plot_states")


def plot_perturbators_data(perts, curr_metadata_with_states_d, curr_factory_res_d, curr_sum_up_data, cid_data_d,
                           cids_to_use_per_pert, crtl_states_by_cid, to_save, output_file_path, save_pdf):
    num_of_rows = len(perts)
    raw_states_d = dict()
    fig, axs = plt.subplots(ncols=3, nrows=num_of_rows,
                            figsize=(20, 4 * num_of_rows),
                            gridspec_kw={"wspace": 0.6, "hspace": 0.65})
    row_id = 0
    for pert in tqdm(perts):
        pert_analysis_summary, pert_analysis_corr, curr_raw_states_d, replace_d = gather_data(pert,
                                                                                              curr_factory_res_d,
                                                                                              curr_metadata_with_states_d,
                                                                                              curr_sum_up_data)

        raw_states_d[pert] = curr_raw_states_d

        for k, v in replace_d.items():
            cid = k
            if pert not in cids_to_use_per_pert or cid not in cids_to_use_per_pert[pert]:
                continue
            pert_analysis_summary.loc[(k, pert, 'combined'), 'dosage'] = v

            tmp_summary = pert_analysis_summary.query("cell_id==@cid")
            tmp_corr = pert_analysis_corr.query("cell_id==@cid")
            tmp_states = raw_states_d[pert].query("cell_id==@cid")

            states_ax = axs[row_id, 0]
            states_t_val_ax = axs[row_id, 1]
            corr_ax = axs[row_id, 2]

            plot_states_t_val(tmp_summary, states_t_val_ax, pert, cid, cid_data_d[cid])

            plot_corr(tmp_summary, tmp_corr, corr_ax, cid, cid_data_d[cid])

            plot_states(crtl_states_by_cid, tmp_states, states_ax, cid, cid_data_d[cid])

            row_id += 1
    if to_save:
        plt.savefig(output_file_path, bbox_inches='tight')
        if "jpg" in output_file_path and save_pdf:
            plt.savefig(output_file_path.replace("jpg","pdf"), bbox_inches='tight')
    plt.close()
