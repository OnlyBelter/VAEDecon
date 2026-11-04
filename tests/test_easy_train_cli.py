import runpy
import tomllib
from pathlib import Path

import pytest

import vaedecon.cli.easy_train as easy_train_cli


def test_email_settings_load_and_normalize_port(tmp_path):
    config_path = tmp_path / "email.yaml"
    config_path.write_text(
        "sender: sender@example.org\n"
        "receiver: receiver@example.org\n"
        "password: test-secret\n"
        "smtp_server: smtp.example.org\n"
        "smtp_port: '587'\n",
        encoding="utf-8",
    )

    settings = easy_train_cli.load_email_settings(config_path)

    assert settings["smtp_port"] == 587
    assert settings["sender"] == "sender@example.org"


def test_package_email_config_example_is_valid():
    example_path = (
        Path(__file__).resolve().parents[1]
        / "vaedecon"
        / "cli"
        / "email.local.example.yaml"
    )

    settings = easy_train_cli.load_email_settings(example_path)

    assert settings["smtp_port"] == 465
    assert settings["sender"] == "your-email@example.com"


def test_package_includes_only_the_email_example_yaml():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    package_data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))[
        "tool"
    ]["setuptools"]["package-data"]["vaedecon.cli"]

    assert package_data == ["email.local.example.yaml"]


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("sender: sender@example.org\n", "requires a nonempty"),
        (
            "sender: sender@example.org\n"
            "receiver: receiver@example.org\n"
            "password: test-secret\n"
            "smtp_server: smtp.example.org\n"
            "smtp_port: true\n",
            "smtp_port",
        ),
        (
            "sender: sender@example.org\n"
            "receiver: receiver@example.org\n"
            "password: test-secret\n"
            "smtp_server: smtp.example.org\n"
            "unexpected: value\n",
            "unsupported fields",
        ),
    ],
)
def test_email_settings_reject_invalid_values(tmp_path, contents, message):
    config_path = tmp_path / "email.yaml"
    config_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        easy_train_cli.load_email_settings(config_path)


def test_notify_completion_uses_environment_settings(monkeypatch):
    environment = {
        "VAEDECON_EMAIL_SENDER": "sender@example.org",
        "VAEDECON_EMAIL_PASSWORD": "test-secret",
        "VAEDECON_EMAIL_RECEIVER": "receiver@example.org",
        "VAEDECON_SMTP_SERVER": "smtp.example.org",
        "VAEDECON_SMTP_PORT": "587",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    sent = {}
    monkeypatch.setattr(
        easy_train_cli,
        "send_email_notification",
        lambda **kwargs: sent.update(kwargs),
    )

    easy_train_cli.notify_completion("config.yaml")

    assert sent["sender"] == "sender@example.org"
    assert sent["receiver"] == "receiver@example.org"
    assert sent["smtp_port"] == 587


def test_notify_completion_keeps_email_failures_nonfatal(monkeypatch, caplog):
    def fail_to_send(**kwargs):
        raise OSError("SMTP unavailable")

    monkeypatch.setattr(easy_train_cli, "send_email_notification", fail_to_send)

    easy_train_cli.notify_completion(
        "config.yaml",
        {
            "sender": "sender@example.org",
            "password": "test-secret",
            "receiver": "receiver@example.org",
            "smtp_server": "smtp.example.org",
            "smtp_port": 465,
        },
    )

    assert "email notification failed (OSError)" in caplog.text


def test_easy_train_help_lists_combined_cli_options(capsys):
    with pytest.raises(SystemExit) as exc_info:
        easy_train_cli.main(["--help"])

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--email-config" in help_text
    assert "Optional local email YAML file" in help_text
    assert "--skip-training" in help_text
    assert "--overwrite-outputs" in help_text


def test_email_config_argument_is_optional():
    args = easy_train_cli._build_parser().parse_args(["-c", "config.yaml"])

    assert args.email_config is None


@pytest.mark.parametrize(
    "relative_path",
    [
        Path("examples/easy_train_cli.py"),
        Path("../VAEDecon_example/easy_train_cli.py"),
    ],
)
def test_example_scripts_delegate_to_package_main(relative_path):
    script_path = Path(__file__).resolve().parents[1] / relative_path
    if not script_path.is_file():
        pytest.skip(f"Optional example script is unavailable: {script_path}")

    namespace = runpy.run_path(str(script_path))

    assert namespace["main"] is easy_train_cli.main
