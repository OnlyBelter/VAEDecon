"""
Simple inference example for VAEDecon
"""
from vaedecon.workflow import predict_vaedecon
from vaedecon.configs import VAEDeconConfig


# ============ Method 1 ============
def example_1_simple_prediction():
    """Simple prediction without visualization"""
    results = predict_vaedecon(
        model_dir='./output/vae/VAE_training_2025-11-03_22-28-38/final_model',
        data_file_path='./datasets/test_data.h5ad'
    )

    print("Prediction completed!")
    print(f"Predicted cell proportions shape: {results['pred_cell_prop'].shape}")


# ============ Method 2 ============
def example_2_predict_with_visualization():
    """Predict with visualization and additional inputs"""
    results = predict_vaedecon(
        model_dir='./output/vae/VAE_training_2025-11-03_22-28-38/final_model',
        data_file_path='./datasets/test_data.h5ad',
        output_dir='./results/test_predictions',
        visualize=True,
        pred_cell_prop_file_path='./results/deside_predictions.csv',
        sample2cell_id_file_path='./datasets/sample2cell_id.csv',
        sct_gep_file_path='./datasets/sct_gep.h5ad'
    )

    print("Prediction and visualization completed!")


# ============ Method 3 ============
def example_3_batch_prediction():
    """Predict multiple datasets in a batch"""
    model_dir = './output/vae/VAE_training_2025-11-03_22-28-38/final_model'

    datasets = [
        './datasets/test_set1.h5ad',
        './datasets/test_set2.h5ad',
        './datasets/tcga_data.h5ad',
    ]

    for data_file in datasets:
        print(f"\nProcessing: {data_file}")
        results = predict_vaedecon(
            model_dir=model_dir,
            data_file_path=data_file,
            visualize=False
        )
        print(f"  ✓ Completed: {data_file}")


# ============ Method 4 ============
def example_4_custom_config_prediction():
    """Prediction with custom configuration"""
    # Customize evaluation settings
    config = VAEDeconConfig()
    config.evaluation.n_samples = 5
    config.evaluation.plot_latent_space = True
    config.evaluation.figsize = (5, 5)

    results = predict_vaedecon(
        model_dir='./output/vae/VAE_training_2025-11-03_22-28-38/final_model',
        data_file_path='./datasets/test_data.h5ad',
        config=config,
        visualize=True
    )

    print("Prediction with custom config completed!")


# ============ 方法5: TCGA数据预测 ============
def example_5_tcga_prediction():
    """预测TCGA数据"""
    results = predict_vaedecon(
        model_dir='./output/vae/VAE_training_2025-11-03_22-28-38/final_model',
        data_file_path='./datasets/TCGA/merged_tpm.csv',
        output_dir='./results/tcga_predictions',
        device='cpu',  # TCGA数据较大，可能需要使用CPU
        visualize=False,
        save_reconstructed_geps=True,
        dataset_type='tcga'
    )

    print("TCGA prediction completed!")
    print(f"Reconstructed GEPs saved to: {results['gep_result_dir']}")


if __name__ == '__main__':
    # 选择一个示例运行
    # example_1_simple_prediction()
    example_2_predict_with_visualization()
    # example_3_batch_prediction()
    # example_4_custom_config_prediction()
    # example_5_tcga_prediction()