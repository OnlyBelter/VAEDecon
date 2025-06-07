import logging
import os
import sys
import numpy as np
import pandas as pd
import anndata as an
from typing import Union, Optional, List
from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn import preprocessing as pp
from .pub_func import (log_exp2cpm, read_df, non_log2log_cpm,
                       non_log2cpm, get_inx2cell_type, log_message)

logger = logging.getLogger(__name__)


class ReadH5AD(object):
    """
    Read .h5ad file and provides methods to access and process its data.
    Values are typically log2 transformed (e.g., log2(TPM+1) or log2(CPM+1)).

    :param file_path: Path to the .h5ad file.
    :param show_info: Whether to print dataset information upon loading.
    """

    def __init__(self, file_path: Union[str, Path], show_info: bool = False):
        self.file_path = Path(file_path)
        try:
            # Consider using backed mode for very large files if you only ever access slices
            # self.dataset = an.read_h5ad(self.file_path, backed='r')
            self.dataset = an.read_h5ad(self.file_path)
            logger.info(f"Successfully loaded: {self.file_path}")
        except FileNotFoundError:
            logger.error(f"H5AD file not found: {self.file_path}")
            raise
        except Exception as e:
            logger.error(f"Error reading H5AD file {self.file_path}: {e}")
            raise

        if show_info:
            print(self.dataset)

    def get_df(self,
               obs_names: Optional[List[str]] = None,
               result_file_path: Optional[Union[str, Path]] = None,
               convert_to_tpm: bool = False,  # Note: This implies input is log-transformed
               scaling_by_sample: bool = False) -> pd.DataFrame:
        """
        Extracts data for specified observations (samples/cells), performs optional
        transformations, and returns a Pandas DataFrame (samples by genes).

        Args:
            obs_names: Optional list of observation names (sample/cell IDs) to fetch.
                       If None, all observations from self.dataset are processed.
            result_file_path: Optional path to save the resulting DataFrame as a CSV.
            convert_to_tpm: If True, attempts to convert data (assumed to be log-transformed
                              counts like log2(X+1)) to TPM. Requires a valid log_exp2cpm function.
            scaling_by_sample: If True, scales the expression values of each sample
                               (row-wise) to the range [0, 1] using MinMaxScaler.

        Returns:
            A Pandas DataFrame. If obs_names are specified and none are found,
            or if the initial dataset is empty, an empty DataFrame with appropriate
            columns is returned.
        """

        if self.dataset.n_obs == 0:
            logger.warning(f"Source AnnData object from {self.file_path} has 0 observations.")
            return pd.DataFrame(columns=self.dataset.var_names.to_list())

        adata_slice: an.AnnData

        if obs_names is not None:
            if not obs_names:  # Handle empty list explicitly
                logger.warning("Empty obs_names list provided to get_df. Returning empty DataFrame.")
                return pd.DataFrame(columns=self.dataset.var_names.to_list())

            # Filter provided obs_names to those actually present in the AnnData object's index
            valid_obs_mask = self.dataset.obs_names.isin(obs_names)
            actual_obs_to_slice = self.dataset.obs_names[valid_obs_mask].to_list()

            if not actual_obs_to_slice:
                requested_preview = obs_names[:min(5, len(obs_names))]
                logger.warning(
                    f"None of the {len(obs_names)} provided obs_names (e.g., {requested_preview}) "
                    f"were found in the dataset from {self.file_path}. Returning empty DataFrame."
                )
                return pd.DataFrame(columns=self.dataset.var_names.to_list())

            logger.info(f"Subsetting AnnData for {len(actual_obs_to_slice)} requested observations.")
            # Slicing AnnData usually returns a view. .copy() makes it an independent object in memory.
            # This is crucial if you are only working with this subset for transformations.
            adata_slice = self.dataset[actual_obs_to_slice, :].copy()
        else:
            # Process the entire dataset.
            # Making a copy ensures that self.dataset remains unchanged by downstream processing.
            adata_slice = self.dataset.copy()
            logger.info(f"Processing all {self.dataset.n_obs} observations from {self.file_path}.")

        if adata_slice.n_obs == 0:  # Safeguard if slicing resulted in empty
            logger.warning("Resulting AnnData slice has 0 observations. Returning empty DataFrame.")
            return pd.DataFrame(columns=self.dataset.var_names.to_list())

        # Extract expression data .X
        # Convert to dense NumPy array if sparse, ensure float32
        if isinstance(adata_slice.X, csr_matrix):
            x_data = adata_slice.X.toarray().astype(np.float32)
        else:
            # Ensure it's a new numpy array, not a view, and correct dtype
            x_data = np.array(adata_slice.X, dtype=np.float32)

        # Apply transformations
        if convert_to_tpm:
            # Ensure log_exp2cpm function is available and correctly implemented
            if 'log_exp2cpm' not in globals() and not hasattr(self, 'log_exp2cpm'):
                logger.error("Function 'log_exp2cpm' is not defined but convert_to_tpm=True.")
                raise NameError("Function 'log_exp2cpm' is not defined.")
            try:
                x_data = log_exp2cpm(x_data)  # This function needs to be robust
            except Exception as e:
                logger.error(f"Error during log_exp2cpm conversion: {e}")
                raise

        if scaling_by_sample:
            logger.info("Applying MinMax scaling per sample (each row scaled to [0,1]).")
            # MinMaxScaler scales features (columns). To scale samples (rows) using sklearn's
            # scaler, we transpose, scale columns (which are now samples), and transpose back.
            if x_data.shape[0] > 0 and x_data.shape[1] > 0:  # Scaler needs data
                scaler = pp.MinMaxScaler(feature_range=(0, 1), copy=True)
                x_data = scaler.fit_transform(x_data.T).T
            elif x_data.shape[0] == 0:
                logger.warning("No data to scale (0 samples).")
            else:  # x_data.shape[1] == 0 (0 genes)
                logger.warning("No data to scale (0 genes).")

        # Create final DataFrame
        df = pd.DataFrame(
            data=x_data,
            index=adata_slice.obs_names.to_list(),  # Use obs_names from the slice
            columns=adata_slice.var_names.to_list()  # Use var_names from the slice (or original if not subsetting vars)
        ).round(3)

        if result_file_path is not None:
            save_path = Path(result_file_path)
            try:
                save_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
                df.to_csv(save_path, float_format='%.3f')
                logger.info(f"DataFrame (shape: {df.shape}) saved to {save_path}")
            except Exception as e:
                logger.error(f"Error saving DataFrame to {save_path}: {e}")

        return df

    def get_cell_fraction(self) -> Optional[pd.DataFrame]:
        """
        Get cell fraction from .obs, (samples by cell types).
        Returns None if .obs is empty or has no columns.
        """
        if self.dataset.n_obs > 0 and self.dataset.obs.shape[1] > 0:
            return self.dataset.obs.copy().round(3)  # Return a copy
        else:
            logger.warning('No observation metadata (cell fractions) found in .obs or dataset is empty.')
            return None

    def get_h5ad(self) -> an.AnnData:
        """
        Get the underlying AnnData object.
        """
        return self.dataset


class ReadExp(object):
    """
    Read gene expression file, and convert to specific format (TPM / CPM, log2cpm1p)

    - TPM: transcript per million

    - CPM: UMI reads per million (3' end sc-RNA seq), same as TPM in the full-length RNA-seq of bulk cells

    - log_space: log2(CPM + 1), or log2(TPM + 1)

    - non_log: non log space, could be normalized to TPM

    - Data from full-length protocols may benefit from normalization methods that take into account gene length
      (e.g. Patelet al, 2014; Kowalczyket al,2015; Soneson & Robinson, 2018), while 3' enrichment data do not.

    - A commonly used normalization method for full-length scRNA-seq data is TPM normalization (Liet al, 2009),
      which comes from bulk RNA-seq analysis. (Luecken, M. D. & Theis, F. J., Mol. Syst. Biol. 15, e8746 (2019))

    :param exp_file: file path or DataFrame, samples by genes
    :param exp_type: TPM / CPM, log_space, non_log
    :param transpose: transpose if exp_file formed as genes (index) by samples (columns)
    """

    def __init__(self, exp_file, exp_type='TPM', transpose: bool = False):
        """
        """
        assert exp_type in ['TPM', 'CPM', 'log_space', 'non_log']
        self.file_type = exp_type
        self.exp = read_df(exp_file)
        if transpose:
            self.exp = self.exp.T
        self.scaled_by_sample = False

    def to_tpm(self):
        """
        Convert to TPM
        """
        if self.file_type == 'non_log':
            self.exp = non_log2cpm(self.exp)
        elif self.file_type == 'TPM' or self.file_type == 'CPM':
            pass
        elif self.file_type == 'log_space':
            self.exp = log_exp2cpm(self.exp)
        self.file_type = 'TPM'

    def to_log2cpm1p(self):
        """
        Convert to log2(TPM + 1)
        """
        if self.file_type != 'log_space':
            self.exp = non_log2log_cpm(self.exp, transpose=False)
        else:
            print('   This file has already log2 transformed.')
        self.file_type = 'log_space'

    def get_file_type(self) -> str:
        """
        Get the file type
        """
        return self.file_type

    def get_exp(self) -> pd.DataFrame:
        """
        Get the expression matrix
        """
        return self.exp.round(3)

    def save(self, file_path, sep=',', transpose: bool = False):
        """
        Save the expression matrix to file

        :param file_path: file path
        :param sep: separator, default is ','
        :param transpose: transpose index and columns
        """
        if transpose:
            self.exp = self.exp.T.copy()
        self.exp.to_csv(file_path, sep=sep, float_format='%.3f')

    def do_scaling(self):
        """
        Scaling GEPs by sample to [0, 1], same as Scaden
        """
        if not self.scaled_by_sample:
            scaler = pp.MinMaxScaler(feature_range=(0, 1), copy=True)
            x_scaled = scaler.fit_transform(self.exp.T).T  # scaling by column (sample), so T is needed here
            self.scaled_by_sample = True
            self.exp = pd.DataFrame(data=x_scaled, index=self.exp.index,
                                    columns=self.exp.columns).round(3)
        # return self.exp

    def do_scaling_by_constant(self, divide_by=20):
        """
        Scaling GEPs by dividing a constant in log space (20 by default), ensures all expression values are in [0, 1)
        """
        if self.file_type != 'log_space':
            raise ValueError('   This file is not in log space')
        if np.any(self.exp.values > 1.0):
            self.exp = self.exp / divide_by

    def align_with_gene_list(self, gene_list: list = None, fill_not_exist=False, pathway_list: bool = False):
        """
        Align the expression matrix with a gene list

        :param gene_list: gene list
        :param fill_not_exist: fill 0 if gene not exist in the provided gene_list when True
        :param pathway_list: gene list contains pathway names, so TPM normalization is not suitable
        """
        common_genes = [i for i in gene_list if i in self.exp.columns]
        not_exist_in_gene_list = [i for i in gene_list if i not in common_genes]
        removed_genes = [i for i in self.exp.columns if i not in common_genes]
        print(f'   {len(common_genes)} common genes will be used, {len(removed_genes)} genes will be removed.')
        self.exp = self.exp.loc[:, common_genes]
        if fill_not_exist and (len(not_exist_in_gene_list) != 0):
            print(f'   {len(not_exist_in_gene_list)} genes are not in current dataset, 0 will be filled')
            _not_exist_exp = pd.DataFrame(np.zeros((self.exp.shape[0], len(not_exist_in_gene_list))),
                                          index=self.exp.index,
                                          columns=not_exist_in_gene_list)
            self.exp = pd.concat([self.exp, _not_exist_exp], axis=1)
            self.exp = self.exp.loc[:, gene_list]
        if not pathway_list:
            if self.file_type == 'log_space':  # scaling to TPM after alignment
                self.to_tpm()
                self.to_log2cpm1p()
            else:
                self.file_type = 'non_log'
                self.to_tpm()


class TrainingDatasetLoader(object):
    def __init__(self, pos_data_path, neg_data_path):

        print("Opening {} and {}".format(pos_data_path, neg_data_path))
        sys.stdout.flush()

        self.cache_pos = ReadH5AD(pos_data_path)
        self.cache_neg = ReadH5AD(neg_data_path)

        print("Loading data into memory...")
        sys.stdout.flush()
        self.gep_pos = self.cache_pos.get_df()  # log space, samples by genes
        self.cell_prop_pos = self.cache_pos.get_cell_fraction()  # samples by cell types
        self.gep_neg = self.cache_neg.get_df()
        self.cell_prop_neg = self.cache_neg.get_cell_fraction()

        # self.images = self.cache['images'][:]
        # self.labels = self.cache['labels'][:].astype(np.float32)
        # self.image_dims = self.images.shape
        # n_train_samples = self.cell_prop_pos.shape[0] + self.cell_prop_neg.shape[0]

        # self.train_inds = np.random.permutation(np.arange(n_train_samples))

        self.pos_train_inds = np.random.permutation(np.arange(self.cell_prop_pos.shape[0]))
        self.neg_train_inds = np.random.permutation(np.arange(self.cell_prop_neg.shape[0]))

    def get_train_size(self):
        return self.cell_prop_pos.shape[0] + self.cell_prop_neg.shape[0]

    def get_n_genes(self):
        # the number of genes in each GEP
        return self.gep_pos.shape[1]

    def get_n_cell_types(self):
        return self.cell_prop_pos.shape[1]

    # def get_train_steps_per_epoch(self, batch_size, factor=10):
    #     return self.get_train_size()//factor//batch_size

    def get_batch(self, n, only_faces=False, p_pos=None, p_neg=None):
        if only_faces:
            selected_inds = np.random.choice(self.pos_train_inds, size=n, replace=False, p=p_pos)
            train_gep = self.gep_pos.iloc[selected_inds, :].copy()
            train_cell_prop = self.cell_prop_pos.iloc[selected_inds, :].copy()
        else:
            # selected_pos_inds = tfp.distributions.Multinomial(total_count=n//2, logits=self.pos_train_inds, )
            selected_pos_inds = np.random.choice(self.pos_train_inds, size=n // 2, replace=False, p=p_pos)
            selected_neg_inds = np.random.choice(self.neg_train_inds, size=n // 2, replace=False, p=p_neg)
            t_gep1 = self.gep_pos.iloc[selected_pos_inds, :].copy()
            t_cell_prop1 = self.cell_prop_pos.iloc[selected_pos_inds, :].copy()
            t_gep2 = self.gep_neg.iloc[selected_neg_inds, :].copy()
            t_cell_prop2 = self.cell_prop_neg.iloc[selected_neg_inds, :].copy()
            train_gep = pd.concat([t_gep1, t_gep2])
            train_cell_prop = pd.concat([t_cell_prop1, t_cell_prop2])

        return train_gep.values, train_cell_prop.values

    # def get_n_most_prob_faces(self, prob, n):
    #     idx = np.argsort(prob)[::-1]
    #     most_prob_inds = self.pos_train_inds[idx[:10*n:10]]
    #     return (self.images[most_prob_inds,...]/255.).astype(np.float32)

    def get_all_pos(self) -> np.ndarray:
        return self.gep_pos.values


def read_single_cell_type_dataset(sct_dataset_file_path: str, latent_z_nn_info_file: Union[str, pd.DataFrame] = None):
    """
    positive samples of SCT (single cell type), generated by SingleCellTypeGEPGenerator
    :param sct_dataset_file_path: the file path of GEPs for single cell type (SCT), positive samples
    :param latent_z_nn_info_file: neighbor information of latent z for all samples, used for QC
    """
    sct_dataset_obj = ReadH5AD(sct_dataset_file_path)
    sct_dataset_df = sct_dataset_obj.get_df(convert_to_tpm=True)
    cell_type_list = sct_dataset_obj.get_h5ad().obs.columns.tolist()
    inx2cell_type = get_inx2cell_type(cell_type_list=cell_type_list)
    if (latent_z_nn_info_file is not None) and os.path.exists(latent_z_nn_info_file):
        latent_z_nn_info = read_df(latent_z_nn_info_file)
        latent_z_nn_info = \
            latent_z_nn_info.loc[
                (latent_z_nn_info['class'] != -1) &  # only positive samples
                (latent_z_nn_info['n_neighbor_class'] == 1) &  # all neighbors belong to the same cell type
                (latent_z_nn_info['class'] == latent_z_nn_info['pred_class']),  # predicted class is same as true class
                ['class']].copy()
        latent_z_nn_info['cell_prop'] = latent_z_nn_info['class'].map(lambda x: inx2cell_type[x])
        sct_dataset_obs = latent_z_nn_info.copy()
    else:
        sct_dataset_obs = sct_dataset_obj.get_cell_fraction()
        sct_dataset_obs['class'] = sct_dataset_obs.values.argmax(axis=1)
        sct_dataset_obs['cell_prop'] = sct_dataset_obs['class'].map(lambda x: inx2cell_type[x])
    return sct_dataset_obs, sct_dataset_df


def read_gene_set(gene_set_file_path: list, max_n_genes: int = 300) -> pd.DataFrame:
    """
    read gene set from .gmt files and convert to DataFrame with genes as index and gene sets as columns,
     1 for a gene in a gene set and 0 for not
    :param gene_set_file_path: the file path of gene set
    :param max_n_genes: the maximum number of genes in a gene set
    :return: DataFrame of gene set with genes as index and gene sets as columns
    """
    gs2genes = {}
    all_genes = set()
    for gs_file in gene_set_file_path:
        if not os.path.exists(gs_file):
            raise FileNotFoundError(f'gene set file {gs_file} not found')
        with open(gs_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    gs, _, *genes = line.split('\t')
                    if len(genes) > max_n_genes:
                        genes = genes[:max_n_genes]
                    gs2genes[gs] = genes
                    all_genes.update(genes)
    gene_set_df = pd.DataFrame(index=list(all_genes), columns=list(gs2genes.keys()))
    for gs, genes in gs2genes.items():
        gene_set_df.loc[genes, gs] = 1
    gene_set_df.fillna(0, inplace=True)
    return gene_set_df


def get_gene_mean_std_across_cell_types(sct_dataset_fp: str, result_fp, gene_list_fp,
                                        cell_type_fp, scaling_by_constant: bool=True, log2p1: bool=True) -> None:
    """Get the mean and std of gene expression values across cell types in the SCT dataset."""
    # sct_dataset_obj = ReadH5AD(sct_dataset_fp)
    # sct_dataset_df = sct_dataset_obj.get_df(convert_to_tpm=True)
    if not os.path.exists(result_fp):
        gene_list = pd.read_csv(gene_list_fp, index_col=0, header=None).index.tolist()
        cell_type_list = pd.read_csv(cell_type_fp, index_col=0, header=None).index.tolist()
        sct_obj = ReadH5AD(sct_dataset_fp)

        h5ad = sct_obj.get_h5ad()
        h5ad_obs = h5ad.obs.copy()
        ct2ave = {}
        for col in cell_type_list:
            x = h5ad[h5ad_obs[col] == 1, :]
            x_df = pd.DataFrame(x.X, index=x.obs.index, columns=x.var.index)
            exp_obj = ReadExp(x_df, exp_type='log_space')
            exp_obj.align_with_gene_list(gene_list=gene_list, fill_not_exist=True)
            exp_obj.to_tpm()
            exp = exp_obj.get_exp()
            exp_avg = exp.mean(axis=0)
            exp_std = exp.std(axis=0)
            ct2ave[col + '_avg'] = exp_avg
            ct2ave[col + '_std'] = exp_std
        ct2ave = pd.DataFrame(ct2ave)
        if log2p1 is True:
            ct2ave = np.log2(ct2ave + 1)
        if scaling_by_constant is True:
            ct2ave = ct2ave / 20
        ct2ave.to_csv(result_fp, float_format='%.3f')


def load_or_compute_gene_mean_std(
    sct_gep_fp: str,
    gene_list: list[str],
    cell_type_fp: str,
    input_gene_list_fp: str,
    scaling_by_constant: bool,
    log_fn=print,
    out_fp=None,
) -> pd.DataFrame:
    """
    1) Determine the gene‐mean/std filename based on `scaling_by_constant`
    2) If it exists and perfectly matches `gene_list`, load & return it
    3) Otherwise, (re)compute it via `get_gene_mean_std_across_cell_types`
       and then load & return it
    """
    # base_dir = Path(sct_gep_fp).parent
    # fname = (
    #     "gene_mean_std_log2p1_scaled.csv"
    #     if scaling_by_constant
    #     else "gene_mean_std_log2p1.csv"
    # )
    # out_fp = base_dir / fname
    out_fp = out_fp

    def file_matches(df: pd.DataFrame) -> bool:
        # exact same genes, same order
        return (
            list(df.index) == gene_list
            and df.shape[0] == len(gene_list)
        )

    if out_fp.exists():
        df = pd.read_csv(out_fp, index_col=0)
        if file_matches(df):
            log_fn(f"> Using existing gene‐mean/std file: {out_fp}")
            return df
        else:
            log_fn(
                "> Existing gene‐mean/std file does not match current gene list, "
                "recomputing..."
            )

    else:
        log_fn(f"> No precomputed file found at {out_fp}, computing now...")

    # (re)compute
    log_fn("> Computing means & stds of each gene across cell types …")
    get_gene_mean_std_across_cell_types(
        result_fp=str(out_fp),
        gene_list_fp=input_gene_list_fp,
        cell_type_fp=cell_type_fp,
        sct_dataset_fp=sct_gep_fp,
        scaling_by_constant=scaling_by_constant,
    )

    # load and return
    df = pd.read_csv(out_fp, index_col=0)
    if not file_matches(df):
        raise RuntimeError(
            f"After computation, {out_fp} still does not match the expected gene list!"
        )
    return df
