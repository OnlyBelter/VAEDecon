import os
from typing import List

import numpy as np
import pandas as pd

from .pub_func import calculate_rmse, get_ccc, get_corr


def get_core_zone_of_pca(pca_data: pd.DataFrame, col_x, col_y, q_lower, q_upper):
    """
    get core zone of TCGA in PCA plot
    :param pca_data: PCA data
    :param col_x: x axis column name
    :param col_y: y axis column name
    :param q_lower: lower quantile
    :param q_upper: upper quantile
    :return: coordinate_core_zone, n_tcga_in_core_zone, n_non_tcga_in_core_zone
    """
    assert 'TCGA' in pca_data['class'].unique()
    x_tcga = pca_data.loc[pca_data['class'] == 'TCGA', col_x]
    y_tcga = pca_data.loc[pca_data['class'] == 'TCGA', col_y]
    x_q_lower = np.quantile(x_tcga, q_lower)
    x_q_upper = np.quantile(x_tcga, q_upper)
    y_q_lower = np.quantile(y_tcga, q_lower)
    y_q_upper = np.quantile(y_tcga, q_upper)
    core_zone = pca_data.loc[(pca_data[col_x] >= x_q_lower) & (pca_data[col_x] <= x_q_upper) &
                             (pca_data[col_y] >= y_q_lower) & (pca_data[col_y] <= y_q_upper), :]
    coordinate_core_zone = {'x_lower': x_q_lower, 'x_upper': x_q_upper, 'y_lower': y_q_lower, 'y_upper': y_q_upper}
    n_tcga_in_core_zone = core_zone.loc[core_zone['class'] == 'TCGA', :].shape[0]
    n_non_tcga_in_core_zone = core_zone.loc[core_zone['class'] != 'TCGA', :].shape[0]
    return coordinate_core_zone, n_tcga_in_core_zone, n_non_tcga_in_core_zone


def calculate_single_cell_gep_metrics_per_sample(
    *,
    sc_gep_result_dir: str,
    cell_types: List[str],
    n_samples: int,
    selected_sample2cell_id_file_path: str,
) -> pd.DataFrame:
    selected_sample2cell_id = pd.read_csv(selected_sample2cell_id_file_path, index_col=0)
    rows: list[dict] = []

    for cell_type in cell_types:
        selected_sample2cell_id_mapping = selected_sample2cell_id.loc[
            selected_sample2cell_id["cell_type"] == cell_type, "selected_cell_id"
        ].to_dict()
        sample_ids = list(selected_sample2cell_id_mapping.keys())

        y_true_fp = os.path.join(
            sc_gep_result_dir, f"sct_gep_{cell_type}_from_{n_samples}_bulksamples.csv"
        )
        y_pred_fp = os.path.join(
            sc_gep_result_dir, f"recon_sct_gep_{cell_type}_from_{n_samples}_bulksamples.csv"
        )
        if not (os.path.exists(y_true_fp) and os.path.exists(y_pred_fp)):
            continue

        y_true = pd.read_csv(y_true_fp, index_col=0)
        y_true = y_true.loc[:, [selected_sample2cell_id_mapping[i] for i in sample_ids]]
        y_true.columns = sample_ids
        y_pred = pd.read_csv(y_pred_fp, index_col=0)

        common_genes = [g for g in y_true.index if g in y_pred.index]
        if not common_genes:
            continue
        y_true = y_true.loc[common_genes, :]
        y_pred = y_pred.loc[common_genes, :]

        common_samples = [s for s in sample_ids if s in y_pred.columns and s in y_true.columns]
        for s in common_samples:
            yt = y_true[s].values
            yp = y_pred[s].values
            corr, p_value = get_corr(yp, yt, return_p_value=True)
            rmse = calculate_rmse(y_true=yt, y_pred=yp)
            ccc = get_ccc(x=yp, y=yt)
            rows.append(
                {
                    "cell_type": cell_type,
                    "sample_id": s,
                    "corr": corr,
                    "p_value": float(p_value),
                    "rmse": rmse,
                    "ccc": ccc,
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["cell_type", "sample_id"]).reset_index(drop=True)
    return df
