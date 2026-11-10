import anndata
import numpy as np
import pandas as pd
import torch

from vaedecon.configs.default_config import GEPDatasetConfig
from vaedecon.data import GEPDataset
from vaedecon.utility.read_file import ReadH5AD


def test_h5ad_labels_preserve_small_proportions_through_training_dataset(tmp_path):
    path = tmp_path / "bulk.h5ad"
    labels = pd.DataFrame(
        {"Rare": [0.0004, 0.0006], "Other": [0.9996, 0.9994]},
        index=["sample_1", "sample_2"],
    )
    data = anndata.AnnData(
        X=np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float32),
        obs=labels,
        var=pd.DataFrame(index=["gene_1", "gene_2"]),
    )
    data.write_h5ad(path)

    reader = ReadH5AD(path)
    loaded = reader.get_cell_fraction()
    pd.testing.assert_frame_equal(loaded, labels)
    loaded.iloc[0, 0] = 0.5
    assert reader.get_cell_fraction().iloc[0, 0] == 0.0004
    reader.close()

    dataset = GEPDataset(
        config=GEPDatasetConfig(
            file_paths=[path],
            processed_data_dir=tmp_path / "processed",
            persist_processed_data=False,
            remove_low_var_genes=False,
        )
    )
    torch.testing.assert_close(
        dataset[0].labels, torch.tensor([0.0004, 0.9996], dtype=torch.float32)
    )
    torch.testing.assert_close(
        dataset[1].labels, torch.tensor([0.0006, 0.9994], dtype=torch.float32)
    )
