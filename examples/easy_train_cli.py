"""Train VAEDecon and predict configured test sets."""

# Run from VAEDecon_example so config-relative data paths resolve:
# python ../VAEDecon/examples/easy_train_cli.py -c config.yaml
#
# Reuse a trained model and override its saved test sets:
# python ../VAEDecon/examples/easy_train_cli.py \
#     -c config_new_test_sets.yaml \
#     --skip-training \
#     --model-dir output/vae/my_run/final_model

import os
import subprocess
import logging
import argparse
from pathlib import Path
from typing import Dict, Any

from vaedecon import train_vaedecon, predict_vaedecon
from vaedecon.configs.default_config import VAEDeconConfig, TestSetConfig
from vaedecon.workflow.inference import _validate_input_file_path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        # logging.FileHandler('app.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


if __name__ == "__main__":
    # Setup Argument Parser
    parser = argparse.ArgumentParser(
        description=(
            "Train VAEDecon and then predict on configured test sets, OR "
            "run prediction-only on new test sets using an already-trained model "
            "(--skip-training --model-dir ...)."
        )
    )
    parser.add_argument('-c', '--config', type=str, default='vaedecon_config_debug.yaml',
        help='Path to the configuration YAML file.')
    parser.add_argument(
        '--skip-training',
        action='store_true',
        default=False,
        help=(
            "Skip the training step and run only prediction. Requires --model-dir "
            "to point at a trained VAEDecon final_model/ folder. Use this when you "
            "already have a well-trained model and want to score new test sets."
        ),
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default=None,
        help=(
            "Path to a trained VAEDecon model directory (the folder that contains "
            "checkpoint.pth / config.yaml / data_config.json). Required when using "
            "--skip-training; otherwise defaults to config.model.model_dir after "
            "training."
        ),
    )

    args = parser.parse_args()

    if args.skip_training and (not args.model_dir or str(args.model_dir).strip() == ''):
        parser.error("--skip-training requires --model-dir <path/to/final_model>")

    # Training (optional)
    config_file = args.config
    logger.info(f"Using configuration file: {config_file}")

    if args.skip_training:
        model_dir_path = Path(str(args.model_dir)).expanduser().resolve()
        logger.info("Skipping training; will run prediction only using model dir: %s", model_dir_path)

        saved_config_candidates = [
            model_dir_path / "config.yaml",
            model_dir_path / "used_config.yaml",
        ]
        loaded_from_save: VAEDeconConfig | None = None
        for candidate in saved_config_candidates:
            if candidate.exists() and candidate.is_file():
                try:
                    loaded_from_save = VAEDeconConfig.from_yaml(str(candidate))
                    logger.info("Loaded trained-model config from: %s", candidate)
                    break
                except Exception as exc:
                    logger.warning(
                        "Could not load saved training config %s; will fall back to "
                        "loading the input YAML passed via -c/--config instead. "
                        "Details: %s", candidate, exc
                    )

        if loaded_from_save is not None:
            base_config = loaded_from_save
        else:
            logger.info("Falling back to user-provided YAML: %s", config_file)
            base_config = VAEDeconConfig.from_yaml(config_file)

        # Pin model_dir on the loaded config so downstream code behaves exactly as
        # it would after a fresh training run in the same dir.
        try:
            base_config.model.model_dir = str(model_dir_path)
        except Exception:
            pass

        # The user's input YAML may contain *newer* test-set definitions than the
        # one saved under final_model/ (this is the common "reuse trained model on
        # new test sets" workflow). To make that work without hand-editing the
        # baked-in final_model/config.yaml, we merge test_sets from the user YAML
        # on top of the saved config: load the user YAML and, if it defines
        # data.test_sets (or legacy data.test_set_file_path), use those instead.
        try:
            user_yaml_cfg = VAEDeconConfig.from_yaml(config_file)
            user_test_sets: Dict[str, TestSetConfig] = dict(
                getattr(user_yaml_cfg.data, "test_sets", {}) or {}
            )
            if user_test_sets:
                base_config.data.test_sets = user_test_sets
                logger.info(
                    "Using test-set definitions from user-provided YAML %s "
                    "(test sets: %s). These override any test_sets baked into the "
                    "saved trained-model config.",
                    config_file,
                    ", ".join(sorted(user_test_sets.keys())),
                )
            else:
                user_legacy_test = getattr(user_yaml_cfg.data, "test_set_file_path", None)
                if user_legacy_test and str(user_legacy_test).strip() != "":
                    base_config.data.test_set_file_path = str(user_legacy_test)
                    logger.info(
                        "Using legacy test_set_file_path from user-provided YAML %s: %s",
                        config_file, user_legacy_test,
                    )
        except Exception as exc:
            logger.warning(
                "Could not merge test-set definitions from %s; using test sets "
                "from the saved trained-model config. Details: %s", config_file, exc
            )

        config = base_config
    else:
        config = train_vaedecon(config_file=config_file)

    # After training (or loading a saved model), reload the config that the
    # predictor will actually see. This matters because train_vaedecon may
    # persist an updated config.yaml / data_config.json under final_model, and
    # we want easy_train_cli's predict() to match what running
    # `vaedecon predict --model-dir ...` would use. It also makes any stale
    # paths (e.g. SimuTME 12ds vs 11ds naming) visible immediately instead of
    # failing deep inside GEPDataset -> ReadH5AD.
    if args.skip_training:
        model_dir_path = Path(str(args.model_dir)).expanduser().resolve()
        # For --skip-training we already produced the right effective_config
        # above; just make sure model_dir_path is correct.
    else:
        model_dir_path = Path(str(config.model.model_dir))
    saved_config_candidates = [
        model_dir_path / "config.yaml",
        model_dir_path / "used_config.yaml",
    ]
    if args.skip_training:
        # When skipping training, effective_config already reflects the saved
        # config overridden by the user YAML's test sets; don't clobber that.
        effective_config = config
    else:
        effective_config = config
        for candidate in saved_config_candidates:
            if candidate.exists() and candidate.is_file():
                try:
                    effective_config = VAEDeconConfig.from_yaml(str(candidate))
                    logger.info("Reloaded saved training config from: %s", candidate)
                    break
                except Exception as exc:
                    logger.warning(
                        "Could not reload saved training config %s; falling back to "
                        "in-memory training config. Details: %s", candidate, exc
                    )
        in_memory_test_sets: Dict[str, TestSetConfig] = dict(
            getattr(config.data, "test_sets", {}) or {}
        )
        reloaded_test_sets: Dict[str, TestSetConfig] = dict(
            getattr(effective_config.data, "test_sets", {}) or {}
        )
        if in_memory_test_sets and not reloaded_test_sets:
            logger.warning(
                "Reloaded saved config does not contain any test_sets, but the "
                "in-memory training config does. Reusing the in-memory test-set "
                "definitions for post-training prediction. This commonly happens "
                "when training generates debug-overfit test sets after the initial "
                "config save."
            )
            effective_config.data.test_sets = in_memory_test_sets
            first_test = next(iter(in_memory_test_sets.values()))
            effective_config.data.test_set_file_path = first_test.test_set_file_path
            effective_config.data.test_set_sample2cell_id_file_path = (
                first_test.test_set_sample2cell_id_file_path
            )
            effective_config.data.sct_gep_file_path = first_test.sct_gep_file_path

    # Preflight: validate every configured test-set data file before
    # launching prediction. This turns the deep, unhelpful
    # `h5py.File ... FileNotFoundError: unable to open file` stacktrace
    # into a single clear message listing the missing path and nearby
    # SimuTME-style candidates.
    configured_sets: Dict[str, TestSetConfig] = dict(
        getattr(effective_config.data, "test_sets", {}) or {}
    )
    missing_messages: list[str] = []
    if configured_sets:
        for test_name, test_cfg in configured_sets.items():
            try:
                _validate_input_file_path(
                    test_cfg.test_set_file_path,
                    context=f"configured test set '{test_name}'",
                )
            except FileNotFoundError as exc:
                missing_messages.append(str(exc))
    else:
        fallback_test = getattr(effective_config.data, "test_set_file_path", None)
        if fallback_test and str(fallback_test).strip() != "":
            try:
                _validate_input_file_path(
                    fallback_test,
                    context="legacy top-level test set",
                )
            except FileNotFoundError as exc:
                missing_messages.append(str(exc))

    if missing_messages:
        joined = "\n\n".join(missing_messages)
        raise FileNotFoundError(
            "VAEDecon inference preflight failed because one or more test-set "
            "paths are missing. Re-run SimuTME dataset generation with the "
            "same configuration that was used for training, and then update "
            "your VAEDecon YAML to point at the generated files.\n\n"
            + joined
        )

    # Prediction
    final_test_results_dir = model_dir_path / "test_results"
    results = predict_vaedecon(
        model_dir=model_dir_path,
        # data_file_path=config.data.test_set_file_path,
        output_dir=str(final_test_results_dir),
        config=effective_config,
        device='auto'
    )

    # Save configuration file used for training and prediction
    config_save_path = model_dir_path / 'used_config.yaml'
    effective_config.to_yaml(config_save_path)
    logger.info(f"Configuration file saved to: {config_save_path}")

    # Self-define some parameters
    # from vaedecon import VAEDeconConfig, train_vaedecon
    # config = VAEDeconConfig()
    # config.training.num_epochs = 500
    # model_dir = train_vaedecon(config=config)
