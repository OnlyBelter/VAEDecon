"""Combined training and prediction CLI with optional email notification."""

import argparse
import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from ..configs.default_config import TestSetConfig, VAEDeconConfig
from ..workflow.inference import _validate_input_file_path, predict_vaedecon
from ..workflow.train import (
    _preserve_existing_outputs,
    _should_reuse_staged_test_results,
    train_vaedecon,
)

logger = logging.getLogger(__name__)


def send_email_notification(
    subject: str,
    body: str,
    sender: str,
    password: str,
    receiver: str,
    smtp_server: str,
    smtp_port: int = 465,
) -> None:
    """Send a completion email using SMTP over SSL."""
    message = MIMEMultipart()
    message["From"] = sender
    message["To"] = receiver
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain", "utf-8"))

    with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
        server.login(sender, password)
        server.sendmail(sender, receiver, message.as_string())
    print("✅ Email sent successfully!")


def load_email_settings(file_path: str | Path) -> Dict[str, Any]:
    """Load and validate an explicit email configuration file."""
    import yaml

    path = Path(file_path).expanduser().resolve()
    try:
        with path.open(encoding="utf-8") as handle:
            settings = yaml.safe_load(handle)
    except OSError:
        raise ValueError(f"Cannot read email configuration file: {path}") from None
    except (yaml.YAMLError, UnicodeError):
        raise ValueError("Email configuration must be valid UTF-8 YAML.") from None
    if not isinstance(settings, dict):
        raise ValueError("Email configuration must contain a YAML mapping.")

    required = ("sender", "receiver", "password", "smtp_server")
    unknown = set(settings) - set(required) - {"smtp_port"}
    if unknown:
        raise ValueError("Email configuration contains unsupported fields.")
    for name in required:
        if not isinstance(settings.get(name), str) or not settings[name].strip():
            raise ValueError(f"Email configuration requires a nonempty '{name}' string.")

    port = settings.get("smtp_port", 465)
    if isinstance(port, bool) or not isinstance(port, (str, int)):
        raise ValueError("Email configuration 'smtp_port' must be an integer from 1 to 65535.")
    try:
        port = int(port)
    except ValueError:
        raise ValueError("Email configuration 'smtp_port' must be an integer from 1 to 65535.") from None
    if not 1 <= port <= 65535:
        raise ValueError("Email configuration 'smtp_port' must be an integer from 1 to 65535.")
    return {**settings, "smtp_port": port}


def notify_completion(
    config_path: str | Path,
    email_settings: Optional[Dict[str, Any]] = None,
) -> None:
    """Send a completion email when configured; notification failures are nonfatal."""
    if email_settings is None:
        environment_names = {
            "sender": "VAEDECON_EMAIL_SENDER",
            "password": "VAEDECON_EMAIL_PASSWORD",
            "receiver": "VAEDECON_EMAIL_RECEIVER",
            "smtp_server": "VAEDECON_SMTP_SERVER",
        }
        email_settings = {
            key: os.environ.get(name) for key, name in environment_names.items()
        }
        missing = [
            name for key, name in environment_names.items()
            if not email_settings[key]
        ]
        if missing:
            logger.warning(
                "Completion email skipped; missing environment variables: %s",
                ", ".join(missing),
            )
            return
        email_settings["smtp_port"] = os.environ.get("VAEDECON_SMTP_PORT", "465")

    try:
        smtp_port = int(email_settings["smtp_port"])
        if not 1 <= smtp_port <= 65535:
            raise ValueError("SMTP port is out of range")
        send_email_notification(
            subject="✅ Program Finished",
            body=(
                "Your program (VAEDecon) has finished running with the config "
                f"file {config_path}. Please check the results."
            ),
            sender=email_settings["sender"],
            password=email_settings["password"],
            receiver=email_settings["receiver"],
            smtp_server=email_settings["smtp_server"],
            smtp_port=smtp_port,
        )
    except Exception as exc:
        logger.warning(
            "VAEDecon completed, but email notification failed (%s).",
            type(exc).__name__,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train VAEDecon and then predict on configured test sets, OR "
            "run prediction-only on new test sets using an already-trained model "
            "(--skip-training --model-dir ...)."
        )
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="vaedecon_config_debug.yaml",
        help="Path to the configuration YAML file.",
    )
    parser.add_argument(
        "--overwrite-outputs",
        action="store_true",
        help="Preserve existing training outputs in a backup and start training.",
    )
    parser.add_argument(
        "--email-config",
        type=Path,
        help="Optional local email YAML file; overrides environment email settings.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help=(
            "Skip training and run prediction only. Requires --model-dir to point "
            "to a trained VAEDecon final_model/ folder."
        ),
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help=(
            "Path to a trained model directory. Required when using "
            "--skip-training; otherwise taken from the config after training."
        ),
    )
    return parser


def _load_saved_model_config(
    config_file: str,
    model_dir_path: Path,
) -> VAEDeconConfig:
    saved_config_candidates = [
        model_dir_path / "config.yaml",
        model_dir_path / "used_config.yaml",
    ]
    base_config: Optional[VAEDeconConfig] = None
    for candidate in saved_config_candidates:
        if candidate.exists() and candidate.is_file():
            try:
                base_config = VAEDeconConfig.from_yaml(str(candidate))
                logger.info("Loaded trained-model config from: %s", candidate)
                break
            except Exception as exc:
                logger.warning(
                    "Could not load saved training config %s; will fall back to "
                    "the input YAML. Details: %s",
                    candidate,
                    exc,
                )

    if base_config is None:
        logger.info("Falling back to user-provided YAML: %s", config_file)
        base_config = VAEDeconConfig.from_yaml(config_file)

    try:
        base_config.model.model_dir = str(model_dir_path)
    except Exception:
        pass
    try:
        user_yaml_config = VAEDeconConfig.from_yaml(config_file)
        user_test_sets = dict(getattr(user_yaml_config.data, "test_sets", {}) or {})
        if user_test_sets:
            base_config.data.test_sets = user_test_sets
            logger.info(
                "Using test-set definitions from user-provided YAML %s "
                "(test sets: %s). These override any test_sets baked into the "
                "saved trained-model config.",
                config_file,
                ", ".join(sorted(user_test_sets)),
            )
        else:
            user_legacy_test = getattr(
                user_yaml_config.data,
                "test_set_file_path",
                None,
            )
            if user_legacy_test and str(user_legacy_test).strip():
                base_config.data.test_set_file_path = str(user_legacy_test)
                logger.info(
                    "Using legacy test_set_file_path from user-provided YAML %s: %s",
                    config_file,
                    user_legacy_test,
                )
    except Exception as exc:
        logger.warning(
            "Could not merge test-set definitions from %s; using test sets "
            "from the saved trained-model config. Details: %s",
            config_file,
            exc,
        )
    return base_config


def _resolve_effective_config(
    config: VAEDeconConfig,
    *,
    model_dir_path: Path,
    skip_training: bool,
) -> VAEDeconConfig:
    if skip_training:
        return config

    effective_config = config
    for candidate in (
        model_dir_path / "config.yaml",
        model_dir_path / "used_config.yaml",
    ):
        if candidate.exists() and candidate.is_file():
            try:
                effective_config = VAEDeconConfig.from_yaml(str(candidate))
                logger.info("Reloaded saved training config from: %s", candidate)
                break
            except Exception as exc:
                logger.warning(
                    "Could not reload saved training config %s; falling back to "
                    "in-memory training config. Details: %s",
                    candidate,
                    exc,
                )

    in_memory_test_sets = dict(getattr(config.data, "test_sets", {}) or {})
    reloaded_test_sets = dict(
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
    return effective_config


def _validate_test_sets(
    config: VAEDeconConfig,
) -> Dict[str, TestSetConfig]:
    configured_sets = dict(getattr(config.data, "test_sets", {}) or {})
    missing_messages = []
    if configured_sets:
        for test_name, test_config in configured_sets.items():
            try:
                _validate_input_file_path(
                    test_config.test_set_file_path,
                    context=f"configured test set '{test_name}'",
                )
            except FileNotFoundError as exc:
                missing_messages.append(str(exc))
    else:
        fallback_test = getattr(config.data, "test_set_file_path", None)
        if fallback_test and str(fallback_test).strip():
            try:
                _validate_input_file_path(
                    fallback_test,
                    context="legacy top-level test set",
                )
            except FileNotFoundError as exc:
                missing_messages.append(str(exc))

    if missing_messages:
        raise FileNotFoundError(
            "VAEDecon inference preflight failed because one or more test-set "
            "paths are missing. Re-run SimuTME dataset generation with the "
            "same configuration used for training, and update the VAEDecon "
            "YAML to point at the generated files.\n\n"
            + "\n\n".join(missing_messages)
        )
    return configured_sets


def _run_prediction(
    config: VAEDeconConfig,
    *,
    config_file: str,
    model_dir_path: Path,
    configured_sets: Dict[str, TestSetConfig],
    skip_training: bool,
    overwrite_outputs: bool,
) -> None:
    final_test_results_dir = model_dir_path / "test_results"
    config_save_path = model_dir_path / "used_config.yaml"
    staged = config.training.staged_training
    protected = [Path(config_file)]
    if staged and staged.enabled:
        protected.extend(Path(path) for path in staged.stage_init_checkpoints.values())
    for test_config in configured_sets.values():
        for field in (
            "test_set_file_path",
            "test_set_sample2cell_id_file_path",
            "sct_gep_file_path",
        ):
            path = getattr(test_config, field, None)
            if path and str(path).strip():
                protected.append(Path(path))
    if not configured_sets:
        for field in (
            "test_set_file_path",
            "test_set_sample2cell_id_file_path",
            "sct_gep_file_path",
        ):
            path = getattr(config.data, field, None)
            if path and str(path).strip():
                protected.append(Path(path))

    if _should_reuse_staged_test_results(
        config,
        final_test_results_dir,
        skip_training=skip_training,
    ):
        logger.info(
            "Reusing final test predictions produced by staged training at %s",
            final_test_results_dir,
        )
    else:
        _preserve_existing_outputs(
            [final_test_results_dir, config_save_path],
            overwrite=overwrite_outputs,
            protected=protected,
        )
        final_test_results_dir.mkdir(parents=True, exist_ok=False)
        predict_vaedecon(
            model_dir=model_dir_path,
            output_dir=str(final_test_results_dir),
            config=config,
            device="auto",
        )

    if not config_save_path.exists() and not config_save_path.is_symlink():
        config.to_yaml(config_save_path, exclusive=True)
        logger.info("Configuration file saved to: %s", config_save_path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the combined train-and-predict workflow."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.skip_training and (
        not args.model_dir or not str(args.model_dir).strip()
    ):
        parser.error("--skip-training requires --model-dir <path/to/final_model>")

    email_settings = None
    if args.email_config is not None:
        try:
            email_settings = load_email_settings(args.email_config)
        except ValueError as exc:
            parser.error(str(exc))

    config_file = args.config
    logger.info("Using configuration file: %s", config_file)
    if args.skip_training:
        model_dir_path = Path(args.model_dir).expanduser().resolve()
        logger.info(
            "Skipping training; will run prediction only using model dir: %s",
            model_dir_path,
        )
        config = _load_saved_model_config(config_file, model_dir_path)
    else:
        config = train_vaedecon(
            config_file=config_file,
            overwrite_outputs=args.overwrite_outputs,
        )
        model_dir_path = Path(str(config.model.model_dir))

    effective_config = _resolve_effective_config(
        config,
        model_dir_path=model_dir_path,
        skip_training=args.skip_training,
    )
    configured_sets = _validate_test_sets(effective_config)
    _run_prediction(
        effective_config,
        config_file=config_file,
        model_dir_path=model_dir_path,
        configured_sets=configured_sets,
        skip_training=args.skip_training,
        overwrite_outputs=args.overwrite_outputs,
    )
    notify_completion(config_file, email_settings=email_settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
