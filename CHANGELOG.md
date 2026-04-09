# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] - 2026-04-09

### Added
- **Per-celltype z-score KL regularization**: Add new regularization term that constrains empirical z-score distribution of each cell type to N(0, 1) after predicting GEPs in TPM/CPM space
- **Auxiliary loss for low mean/std genes**: Add configurable `low_mean_std_gene_loss` (MSE) to handle genes where z-scores are unreliable
- **Caching for single-cell GEP query**: Add caching to `find_sct_gep_of_bulk_sample` to avoid re-reading large h5ad files repeatedly, speeds up visualization
- **Stochastic Depth**: Add optional stochastic depth to residual blocks in `ResidualBlock` (improves generalization for deep networks)
- **torch.compile support**: Add config option `model.torch_compile` with proper error handling for Python 3.12+ compatibility
- **Gradient clipping**: Add configurable `gradient_clip_val` (default 1.0) to prevent gradient explosion with higher learning rates
- **Kaiming (He) weight initialization** to all neural network components:
  - `EncoderMLP`, `DecoderMLP` in `models/nn/mlp.py`
  - `EncoderResMLP`, `DecoderResMLP`, `ResidualBlock` in `models/nn/res_mlp.py`
  - `GeneTransformerEncoder` in `models/nn/transformer.py`
  - `EncoderHybrid` in `models/nn/fused_mlp_gnn.py`
  - `EncoderSGNN`, `PPIEncoder` in `models/gnn/ppi_only_embedding.py`
  - `Decoder_AE_MLP` in `models/base/base_model.py`

### Changed
- **Default optimizer**: Changed default from `Adam` → `AdamW` (modern best practice with proper weight decay)
- **Activation**: Changed ReLU → GELU in `DecoderResMLP` projector for consistency with residual blocks
- **LayerNorm epsilon**: Changed hardcoded `1e-6` → `EPS` constant for consistency across codebase
- **Import style**: Fixed mixed import in `res_mlp.py` to use relative imports consistently

### Fixed
- **AnnData backed mode error**: Fixed `.copy()` error by using `.to_memory()` instead when slicing in backed mode
- **Dimension mismatch**: Fixed IndexError from applying 3D mask on 2D tensor in `low_mean_std_gene_loss` calculation
- **Progress bar logging**: Added `low_mean_std_gene_loss` and `z_score_kl_loss` to progress bar metrics
- **PyTorch kaiming_normal_ issue**: Fixed `ValueError: Unsupported nonlinearity gelu` by using `nonlinearity='relu'` which works for both ReLU and GELU
- **torch.compile error handling**: Added try-catch around `torch.compile()` to handle Dynamo unsupported Python versions gracefully

### Configuration
- Added `z_score_kl_weight: float = 0.0` to `LossCoefficient` (weight for z-score KL regularization)
- Added `low_mean_std_weight: float = 1.0` to `LossCoefficient` (weight for auxiliary low mean/std gene loss)
- Added `torch_compile: bool = False` to `ModelConfig`
- Added `gradient_clip_val: Optional[float] = 1.0` to `TrainingConfig`
