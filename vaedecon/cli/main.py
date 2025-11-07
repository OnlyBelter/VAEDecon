"""
Command-line interface for VAEDecon
"""
import argparse
import sys
from pathlib import Path

from ..configs.default_config import VAEDeconConfig
from ..workflow.train import train_vaedecon
from ..workflow.inference import predict_vaedecon


def train_command(args):
    """训练命令"""
    if args.config:
        config = VAEDeconConfig.from_yaml(args.config)
    else:
        config = VAEDeconConfig()

    # 覆盖配置参数
    if args.data_dir:
        config.data.data_dir = args.data_dir
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.epochs:
        config.training.num_epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.naming_postfix:
        config.training.naming_postfix = args.naming_postfix

    # 执行训练
    model_dir = train_vaedecon(config=config)
    print(f"\n✅ Training completed! Model saved to: {model_dir}")


def predict_command(args):
    """预测命令"""
    if args.config:
        config = VAEDeconConfig.from_yaml(args.config)
    else:
        config = None

    # 执行预测
    results = predict_vaedecon(
        model_dir=args.model_dir,
        data_file_path=args.data_file,
        output_dir=args.output_dir,
        config=config,
        device=args.device,
        visualize=args.visualize,
        pred_cell_prop_file_path=args.pred_cell_prop,
        batch_size=args.batch_size,
    )

    print(f"\n✅ Prediction completed! Results saved to: {args.output_dir or 'default location'}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='VAEDecon: Variational Autoencoder for Deconvolution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default configuration
  vaedecon train --data-dir ./datasets --output-dir ./output

  # Train with custom configuration
  vaedecon train --config config.yaml

  # Predict with trained model
  vaedecon predict --model-dir ./output/vae/VAE_training_xxx/final_model --data-file ./data/test.h5ad

  # Predict with visualization
  vaedecon predict --model-dir ./model --data-file ./data/test.h5ad --visualize
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--config', type=str, help='Path to configuration file (YAML)')
    train_parser.add_argument('--data-dir', type=str, help='Data directory')
    train_parser.add_argument('--output-dir', type=str, help='Output directory')
    train_parser.add_argument('--epochs', type=int, help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, help='Batch size')
    train_parser.add_argument('--learning-rate', type=float, help='Learning rate')
    train_parser.add_argument('--naming-postfix', type=str, help='Naming postfix for output directory')
    train_parser.set_defaults(func=train_command)

    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict with trained model')
    predict_parser.add_argument('--model-dir', type=str, required=True, help='Path to trained model directory')
    predict_parser.add_argument('--data-file', type=str, required=True, help='Path to input data file')
    predict_parser.add_argument('--output-dir', type=str, help='Output directory')
    predict_parser.add_argument('--config', type=str, help='Path to configuration file (YAML)')
    predict_parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                                help='Device to use')
    predict_parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    predict_parser.add_argument('--pred-cell-prop', type=str, help='Path to predicted cell proportions file')
    predict_parser.add_argument('--batch-size', type=int, default=1024, help='Batch size')
    predict_parser.set_defaults(func=predict_command)

    # Parse arguments
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Execute command
    args.func(args)


if __name__ == '__main__':
    main()