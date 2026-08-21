import os
from pathlib import Path

# import importlib
# import umap
import numpy as np
import pandas as pd
import seaborn as sns
# import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.ticker
# from joblib import dump, load
from .plot_gene import compare_exp_between_group
from ..utility import read_cancer_purity, check_dir, read_df, log2_transform, set_fig_style
from sklearn.metrics import median_absolute_error
# sns.set()
# sns.set(font_scale=1.5)
# plt.rcParams.update({'font.size': 20})
set_fig_style()


def plot_loss(
    history_df,
    output_dir: Path = None,
    x_label="Epoch",
    y_label="Loss",
    file_name=None,
    x_col="epoch",
    aggregate_same_x=True,
    agg_func="last",
    figsize=(8, 6),
    log_y: bool = True,
    metric_pairs=None,
):
    """
    Plot loss curves from a metrics DataFrame.

    Args:
        history_df (pd.DataFrame):
            DataFrame containing logged metrics.
        output_dir (str, optional):
            Directory to save the figure.
        x_label (str):
            Label for x-axis.
        y_label (str):
            Label for y-axis.
        file_name (str, optional):
            Output figure name. Default is 'loss.png'.
        x_col (str):
            Column to use as x-axis, usually 'epoch' or 'step'.
        aggregate_same_x (bool):
            Whether to aggregate duplicated x values.
        agg_func (str):
            Aggregation for duplicated x values: 'last', 'mean', 'min', 'max'.
        figsize (tuple):
            Figure size.
        log_y (bool):
            Whether to use a log scale on the y-axis.
        metric_pairs (list[tuple[str, str]] | None):
            Optional explicit metric columns and legend labels to plot.
            When omitted, the default loss columns are used.

    Returns:
        (fig, ax) if output_dir is None, otherwise None.
    """
    if history_df is None or len(history_df) == 0:
        raise ValueError("history_df is empty.")

    if x_col not in history_df.columns:
        raise ValueError(f"Column '{x_col}' not found in history_df.")

    df = history_df.copy()

    candidate_metrics = metric_pairs if metric_pairs is not None else [
        ("loss", "loss"),
        ("train_loss_epoch", "train loss"),
        ("train_loss", "train loss"),
        ("val_loss", "val loss"),
        ("total_loss", "total loss"),
        ("val_total_loss", "val total loss"),
    ]

    metrics_to_plot = [(col, label) for col, label in candidate_metrics if col in df.columns]

    if len(metrics_to_plot) == 0:
        raise ValueError(
            "None of the expected loss columns were found. "
            f"Available columns: {list(df.columns)}"
        )

    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")

    fig, ax = plt.subplots(figsize=figsize)
    if log_y:
        ax.set_yscale("log", nonpositive="clip")

    all_y_values = []  # collect all y values for range-aware tick formatting

    plotted_lines = []  # store (x_last, y_last, label, color) for end annotations

    for metric_col, metric_label in metrics_to_plot:
        plot_df = df[[x_col, metric_col]].copy()
        plot_df[metric_col] = pd.to_numeric(plot_df[metric_col], errors="coerce")
        plot_df = plot_df.dropna(subset=[x_col, metric_col])

        if len(plot_df) == 0:
            continue

        if aggregate_same_x:
            if agg_func == "last":
                plot_df = plot_df.groupby(x_col, as_index=False).last()
            elif agg_func == "mean":
                plot_df = plot_df.groupby(x_col, as_index=False).mean()
            elif agg_func == "min":
                plot_df = plot_df.groupby(x_col, as_index=False).min()
            elif agg_func == "max":
                plot_df = plot_df.groupby(x_col, as_index=False).max()
            else:
                raise ValueError(f"Unsupported agg_func: {agg_func}")

        plot_df = plot_df.sort_values(by=x_col)

        if log_y:
            y = plot_df[metric_col].to_numpy(dtype=float)
            if np.any(y <= 0):
                plot_df[metric_col] = np.where(y > 0, y, np.nan)
                plot_df = plot_df.dropna(subset=[metric_col])
                if len(plot_df) == 0:
                    continue

        line, = ax.plot(
            plot_df[x_col],
            plot_df[metric_col],
            marker="o",
            markersize=4,
            linewidth=2,
            label=metric_label,
        )

        all_y_values.extend(plot_df[metric_col].tolist())

        # Record the final point of each curve for annotation
        x_last = plot_df[x_col].iloc[-1]
        y_last = plot_df[metric_col].iloc[-1]
        plotted_lines.append((x_last, y_last, metric_label, line.get_color()))

    # ── Y-axis tick improvements ──────────────────────────────────────────────

    if log_y:
        # Major ticks: one per decade, formatted as decimals (e.g. 0.01, 0.1, 1.0)
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(
                lambda val, _: f"{val:.4g}"
            )
        )
        # Minor ticks: 8 subdivisions per decade (2~9 × 10^n), no labels
        ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.tick_params(axis="y", which="major", labelsize=10, length=6)
        ax.tick_params(axis="y", which="minor", length=3)
    else:
        # Linear scale: auto major ticks + 5 minor subdivisions
        ax.yaxis.set_major_locator(matplotlib.ticker.AutoLocator())
        ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(5))
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(
                lambda val, _: f"{val:.4g}"
            )
        )
        ax.tick_params(axis="y", which="major", labelsize=10, length=6)
        ax.tick_params(axis="y", which="minor", length=3)

    # Grid: major solid, minor dotted
    ax.grid(True, which="major", alpha=0.4, linestyle="-")
    ax.grid(True, which="minor", alpha=0.15, linestyle=":")

    # ── Annotate the final value of each curve on the right side ─────────────
    for x_last, y_last, label, color in plotted_lines:
        ax.annotate(
            f"{y_last:.4g}",
            xy=(x_last, y_last),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=8,
            color=color,
            va="center",
        )

    # ── Mark global minimum on each curve ────────────────────────────────────
    for metric_col, metric_label in metrics_to_plot:
        plot_df = df[[x_col, metric_col]].copy()
        plot_df[metric_col] = pd.to_numeric(plot_df[metric_col], errors="coerce")
        plot_df = plot_df.dropna(subset=[x_col, metric_col])
        if len(plot_df) == 0:
            continue
        if aggregate_same_x:
            plot_df = plot_df.groupby(x_col, as_index=False).agg(agg_func if agg_func != "last" else "last")
        plot_df = plot_df.sort_values(by=x_col)
        idx_min = plot_df[metric_col].idxmin()
        x_min = plot_df.loc[idx_min, x_col]
        y_min = plot_df.loc[idx_min, metric_col]
        ax.axhline(y=y_min, linestyle="--", linewidth=0.8, alpha=0.5, color="gray")
        ax.annotate(
            f"min={y_min:.4g}",
            xy=(x_min, y_min),
            xytext=(-4, -14),
            textcoords="offset points",
            fontsize=7.5,
            color="gray",
            ha="center",
        )

    ax.legend(loc="upper right", fontsize=9, framealpha=0.7)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("Training History")
    fig.tight_layout()

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        save_name = file_name if file_name is not None else "loss.png"
        fig.savefig(str(output_dir / save_name), dpi=200, bbox_inches="tight")
        plt.close(fig)
        return None

    return fig, ax


def plot_corr_two_columns(df: pd.DataFrame, output_dir: str, col_name1: str = 'CPE',
                          col_name2: str = 'cancer_cell', cancer_type: str = '', diagonal: bool = True,
                          predicted_by: str = None, font_scale: float = 1.5, scale_exp=False, update_figures=False):
    """
    Plot the relation between two columns in DataFrame `df`

    :param df: a dataFrame which contains CPE (cancer purity) and cancer_fraction

    :param output_dir: result folder

    :param col_name1: column name, such as CPE (cancer purity), x axis

    :param col_name2: column name, such as cancer cell fraction (predicted cancer purity), y axis

    :param cancer_type: mark x axis / y axis label

    :param diagonal: if plot diagonal

    :param predicted_by: model name

    :param font_scale: scale font size

    :param scale_exp: if scale all expression values to range [0, 10] by x_i/max(x) * 10

    :param update_figures: if update figures in output_dir

    :return: None
    """
    check_dir(output_dir)
    result_file_path = os.path.join(output_dir, '{}_vs_predicted_{}_proportion.png'.format(col_name1, col_name2))
    if (not os.path.exists(result_file_path)) or update_figures:
        # sns.set(font_scale=font_scale)
        # plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # show Chinese characters
        plt.figure(figsize=(8, 8))
        df_col1 = df[col_name1].copy()
        df_col2 = df[col_name2].copy()
        if np.any(df_col1 > 2) and scale_exp:
            df_col1 = df_col1 / df_col1.max() * 10
        if np.any(df_col2 > 2) and scale_exp:
            df_col2 = df_col2 / df_col2.max() * 10

        corr = np.corrcoef(df_col1, df_col2)
        # print(corr)
        if np.isnan(corr[0, 1]):
            corr[0, 1] = 0  # when all predicted cell fraction are 0, set corr to 0
        # corr2 = stats.pearsonr(df[col_name1], df[col_name2])
        mae = median_absolute_error(y_true=df_col1, y_pred=df_col2)
        plt.scatter(df_col1, df_col2, s=8)
        plt.xlabel('{} ({})'.format(col_name1, cancer_type))
        # plt.xlabel('{} (头颈癌)'.format(col_name1))
        # plt.ylabel('预测 {} 比例 (样本数={})'.format(col_name2, df.shape[0]))
        x_left, x_right = plt.xlim()
        y_bottom, y_top = plt.ylim()
        if '_true' in col_name2:
            plt.ylabel('{} prop. (n={})'.format(col_name2, df.shape[0]))
        elif predicted_by:
            plt.ylabel('Predicted {} prop. by {} (n={})'.format(col_name2, predicted_by, df.shape[0]))
            # plt.ylabel('{}预测值 (样本数={})'.format(predicted_by, df.shape[0]))
        else:
            plt.ylabel('{} prop. (n={})'.format(col_name2, df.shape[0]))
        if 'CPE' in [col_name1, col_name2]:
            plt.text(0.05, 0.95, 'corr = {:.3f}'.format(corr[0, 1]))
            plt.text(0.05, 0.90, '$MAE$ = {:.3f}'.format(mae))
        elif ('CD8A' in [col_name1, col_name2]) or ('CD8A+CD8B' in [col_name1, col_name2]):
            plt.text(x_left + 1.5, y_top * 0.92, 'corr = {:.3f}'.format(corr[0, 1]))
        elif 'CD3E' in [col_name1, col_name2]:
            plt.text(x_left + 1.5, y_top * 0.92, 'corr = {:.3f}'.format(corr[0, 1]))
        elif 'y_pred' in [col_name1, col_name2]:
            plt.text(0.2, 0.92, 'corr = {:.3f}'.format(corr[0, 1]))
            plt.text(0.2, 0.85, '$MAE$ = {:.3f}'.format(mae))
            # plt.title()
        elif '_true' in col_name1 or '_true' in col_name2:
            plt.text(0.1, 0.92, 'corr = {:.3f}'.format(corr[0, 1]))
            plt.text(0.1, 0.85, '$MAE$ = {:.3f}'.format(mae))
        elif ('_marker_mean' in col_name1) or ('_marker_max' in col_name1):
            # compare mean expression of marker genes and predicted cell fraction
            plt.text(x_right * 0.05, y_top * 0.92, 'corr = {:.3f}'.format(corr[0, 1]))
        elif '_gene_signature_score' in col_name1:
            # compare mean expression of marker genes and predicted cell fraction
            plt.text(x_right * 0.05, y_top * 0.92, 'corr = {:.3f}'.format(corr[0, 1]))
        if diagonal:
            plt.plot([0, 1], [0, 1], linestyle='--', color='tab:gray')
        plt.tight_layout()
        plt.savefig(result_file_path, dpi=200)
        plt.close('all')
    else:
        print(f'   Using previous figure, {result_file_path}')


def plot_predicted_result(cell_frac_result_fp, bulk_exp_fp, cancer_type,
                          model_name, result_dir, cancer_purity_fp: str = None,
                          font_scale=2.0, update_figures=False):
    """
    Plot and evaluate predicted results of DeSide or Scaden model for TCGA data

    :param cell_frac_result_fp: the file path of predicted cell fraction

    :param bulk_exp_fp: the file path of bulk cell expression profile or pd.Dataframe, TPM, gene by sample

    :param cancer_type: only for naming or mark x / y label when plotting

    :param model_name: model name, DeSide or Scaden

    :param result_dir: where to save result

    :param cancer_purity_fp: estimated tumor purity for TCGA, download from
        Aran, D. et al., Nat Commun 6, 8971 (2015), Supplementary Data 1

    :param font_scale: scale font size

    :param update_figures: whether to update figures

    :return: None
    """
    y_pred = read_df(cell_frac_result_fp)  # cell fraction, sample by cell type
    # sep = get_sep(bulk_exp_fp)
    bulk_exp_cpm = read_df(bulk_exp_fp)
    # bulk_exp_cpm = log_exp2cpm(bulk_exp_log2cpm1p)

    # plot CD8 T cell fraction against CD8A expression value
    merged_df1 = y_pred.merge(bulk_exp_cpm.T, left_index=True, right_index=True)
    plot_corr_two_columns(df=merged_df1, col_name2='CD8 T', col_name1='CD8A',
                          predicted_by=model_name, font_scale=font_scale,
                          output_dir=result_dir, diagonal=False, cancer_type=cancer_type, update_figures=update_figures)

    if cancer_purity_fp is not None:
        # read cancer purity file
        cancer_purity = read_cancer_purity(cancer_purity_fp, sample_names=list(y_pred.index))
        merged_df = y_pred.merge(cancer_purity, left_index=True, right_index=True)
        # plot CPE vs cell fraction of cancer cell / 1-others
        if merged_df.shape[0] > 0:
            merged_df.to_csv(os.path.join(result_dir,
                                          f'cancer_purity_merged_{model_name}_predicted_result.csv'))
            plot_corr_two_columns(df=merged_df, col_name1='CPE', col_name2='Cancer Cells',
                                  output_dir=result_dir, font_scale=font_scale,
                                  cancer_type=cancer_type, predicted_by=model_name, update_figures=update_figures)
            # plot_corr_two_columns(df=merged_df, col_name1='CPE', col_name2='1-others',
            #                       output_dir=result_dir, font_scale=font_scale,
            #                       cancer_type=cancer_type, predicted_by=model_name)
        else:
            print('   There is no any samples in cancer purity about this cancer type ({})'.format(cancer_type))
    # plot cell fraction of each cell type before decon_cf
    y_pred['labels'] = 1
    cell_types = sorted(y_pred.columns.to_list())
    cell_types = [i for i in cell_types if i not in ['1-others', 'labels']]
    print('   Cell types: ', ', '.join(cell_types))
    compare_exp_between_group(exp=y_pred, group_list=tuple(cell_types),
                              result_dir=result_dir, xlabel=f'Cell Type ({cancer_type})',
                              ylabel=f'Cell prop. predicted by {model_name}',
                              file_name='pred_cell_prop_before_decon.png', font_scale=font_scale - 0.4,
                              xticks_rotation=50)


def plot_paras(paras_file_path, vae_cla_model, latent_z_pos,
               current_cell_types, sampled_sc_id_file=None, sample_id: str = None, result_file=None):
    """
    plot parameters of regression model (deconvolved GEP) in latent z space
    :param paras_file_path: w, weights of regression model which represent valid GEPs for each cell type
    :param vae_cla_model:
    :param latent_z_pos: latent z for all training set of VAEClassifier model (encoder)
    :param current_cell_types
    :param sample_id: only provide if plot this sample
    :param sampled_sc_id_file: selected single cell id for each simulated GEP
    :param result_file:
    """

    latent_z_pos = read_df(latent_z_pos)
    paras = read_df(paras_file_path)
    if paras.shape[0] > paras.shape[1]:  # sample by cell type
        paras = paras.loc[:, current_cell_types].T
    else:
        paras = paras.loc[current_cell_types, :]
    paras = log2_transform(paras)

    _, _, latent_z_paras, _ = vae_cla_model.encoder_predict(paras.values)

    plt.figure(figsize=(8, 8))
    plt.scatter(latent_z_pos.loc[:, 'z1'], latent_z_pos.loc[:, 'z2'], color='gray')
    if sample_id is not None:  # the location of ground truth
        if sampled_sc_id_file is None:
            raise FileNotFoundError('sampled_sc_id_file should be provided with sample_id to plot this sample')
        sampled_sc_id_file = read_df(sampled_sc_id_file)
        current_sc_ids = sampled_sc_id_file.loc[sample_id, :].copy()
        sc_ids = dict(zip(current_sc_ids['cell_prop'], current_sc_ids['selected_cell_id']))
        sc_id_list = [sc_ids[_] for _ in current_cell_types]
        plt.scatter(latent_z_pos.loc[sc_id_list, 'z1'], latent_z_pos.loc[sc_id_list, 'z2'], marker='x', color='red')
    plt.scatter(latent_z_paras[:, 0], latent_z_paras[:, 1], marker='*', color='green')
    if result_file is not None:
        plt.savefig(result_file, dpi=200)
    plt.close()


def plot_paras_all_cell_types(latent_z_paras_file, latent_z_pos_file, current_cell_types,
                              sampled_sc_id_file=None, sample_id: str = None, result_file=None):
    """
    plot parameters of regression model (deconvolved GEP) in latent z space
    :param latent_z_paras_file: latent z of all cell types which represent valid GEPs for each cell type
        - generated by VAEDecon model (encoder), n_cell_type x latent_dim
    :param latent_z_pos_file: latent z for all training set of VAEClassifier model (encoder)
    :param current_cell_types:
    :param sample_id: only provide if plot this sample
    :param sampled_sc_id_file: selected single cell id for each simulated GEP
    :param result_file:
    """

    latent_z_pos_file = read_df(latent_z_pos_file)
    if type(latent_z_paras_file) == str:
        paras = pd.read_csv(latent_z_paras_file, index_col=[0, 1])
    else:  # pd.Dataframe
        paras = latent_z_paras_file
    sample_inx = [(sample_id, ct) for ct in current_cell_types]
    latent_z_paras = paras.loc[sample_inx, :].copy()  # n_cell_type x latent_dim
    plt.figure(figsize=(8, 8))
    col_names = latent_z_pos_file.columns.to_list()
    plt.scatter(latent_z_pos_file.iloc[:, 0], latent_z_pos_file.iloc[:, 1], color='gray')
    if sample_id is not None:  # the location of ground truth
        if sampled_sc_id_file is None:
            raise FileNotFoundError('sampled_sc_id_file should be provided with sample_id to plot this sample')
        if type(sampled_sc_id_file) == str:
            sampled_sc_id_file = pd.read_csv(sampled_sc_id_file, index_col=[0, 1])
        # sample_inx = list(zip([sample_id] * len(current_cell_types), current_cell_types))
        current_sc_ids = sampled_sc_id_file.loc[sample_inx, 'selected_cell_id'].to_list()
        # sc_ids = dict(zip(current_sc_ids['cell_prop'], current_sc_ids['selected_cell_id']))
        # sc_id_list = [sc_ids[_] for _ in current_cell_types]
        plt.scatter(latent_z_pos_file.loc[current_sc_ids, col_names[0]],
                    latent_z_pos_file.loc[current_sc_ids, col_names[1]],
                    marker='x', color='red')
    plt.scatter(latent_z_paras.loc[:, col_names[0]], latent_z_paras.loc[:, col_names[1]], marker='*', color='green')
    plt.xlabel(f'{col_names[0]} of latent space')
    plt.ylabel(f'{col_names[1]} of latent space')
    if result_file is not None:
        plt.savefig(result_file, dpi=200)
    plt.close()
