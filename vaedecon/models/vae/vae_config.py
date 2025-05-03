from vaedecon.models.base.base_config import BaseModelConfig


class VAEConfig(BaseModelConfig):
    """VAE config class.

    Args:
        input_dim (tuple): The input_data dimension.

        latent_dim (int): The latent space dimension. Default: None.

        reconstruction_loss (str): The reconstruction loss to use ['bce', 'mse']. Default: 'mse'

        using_positional_encoding (bool): Whether to use positional encoding in the latent space to distinguish cell types
    """

    # reconstruction_loss: Literal["bce", "mse"] = "mse"
    using_positional_encoding: bool = True
    # the file path of gene list in the training set (after preprocessing,
    # such as interaction from multiple datasets) and for GEP-level reconstruction
    input_gene_list_fp: str = None
    # gene_list: list = None  # list of gene names in the training set
    cell_type_fp: str = None  # the file path of cell types in the training set and for reconstruction
    scaling_by_constant: bool = True  # whether to scale the input GEP data by a constant factor (20 by default)
    loss_coefficient: dict = {
        "recon_decoder": 0.2,  # coefficient for the reconstruction loss of the decoder
        "recon_convolution": 0.3,  # coefficient for the reconstruction loss by convolution after decoding
        "kld": 0.2,  # coefficient for the KL divergence loss
        "cell_prop": 0.3,  # coefficient for the prediction loss of cell type proportions
    }  # coefficient for each term in the total loss function
    encoder_hidden_dims: list[int] = [1024, 512, 512]
    decoder_hidden_dims: list[int] = [512, 512, 1024]
    encoder_dropout_rate: list[float] = [0.1, 0.1, 0.1]
    decoder_dropout_rate: list[float] = [0.1, 0.1, 0.1]
    predict_cell_prop: bool = False  # whether to predict cell type proportions
    gene_mean_std_fp: str = None  # the file path of the mean and std of gene expression values across cell types in the SCT dataset
