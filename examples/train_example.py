"""
Simple training example for VAEDecon
"""
from vaedecon.workflow import train_vaedecon
from vaedecon.configs import VAEDeconConfig


# ============ 方法1: 使用默认配置 ============
def example_1_default_config():
    """使用默认配置训练"""
    model_dir = train_vaedecon()
    print(f"Model saved to: {model_dir}")


# ============ 方法2: 使用配置文件 ============
def example_2_config_file():
    """使用配置文件训练"""
    model_dir = train_vaedecon(config_file='configs/example_config.yaml')
    print(f"Model saved to: {model_dir}")


# ============ 方法3: 自定义配置 ============
def example_3_custom_config():
    """使用自定义配置训练"""
    # 创建配置对象
    config = VAEDeconConfig()

    # 修改数据配置
    config.data.data_dir = './datasets/'
    config.data.scaling_by_constant = True

    # 修改训练配置
    config.training.output_dir = './output/vae'
    config.training.naming_postfix = 'my_experiment'
    config.training.num_epochs = 500
    config.training.batch_size = 512
    config.training.learning_rate = 1e-4

    # 修改模型配置
    config.model.latent_dim = 10
    config.model.n_cell_types = 16
    config.model.encoders = ['EncoderHybrid']

    # 执行训练
    model_dir = train_vaedecon(config=config)
    print(f"Model saved to: {model_dir}")


# ============ 方法4: 快速实验 ============
def example_4_quick_experiment():
    """快速实验（少量epoch）"""
    config = VAEDeconConfig()
    config.training.num_epochs = 10  # 快速测试
    config.training.naming_postfix = 'quick_test'

    model_dir = train_vaedecon(config=config)
    print(f"Model saved to: {model_dir}")


if __name__ == '__main__':
    # 选择一个示例运行
    # example_1_default_config()
    # example_2_config_file()
    example_3_custom_config()
    # example_4_quick_experiment()