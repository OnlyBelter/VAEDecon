"""Train VAEDecon and predict configured test sets.

Run from VAEDecon_example so config-relative paths resolve:
python ../VAEDecon/examples/easy_train_cli.py -c config.yaml \\
    --email-config email.local.yaml

Reuse a trained model with new test sets:
python ../VAEDecon/examples/easy_train_cli.py \\
    -c config_new_test_sets.yaml \\
    --skip-training \\
    --model-dir output/vae/my_run/final_model \\
    --email-config email.local.yaml

Prefer the installed command for new runs:
vaedecon-easy-train -c my_config.yaml --email-config email.local.yaml

Run on GPU 0 in the background:
cfg="my_config.yaml" && CUDA_VISIBLE_DEVICES=0 nohup vaedecon-easy-train -c "$cfg" > "${cfg%.yaml}.log" 2>&1 &

Email notification is optional. Use --email-config PATH or set
VAEDECON_EMAIL_SENDER, VAEDECON_EMAIL_PASSWORD, VAEDECON_EMAIL_RECEIVER,
VAEDECON_SMTP_SERVER, and optional VAEDECON_SMTP_PORT.
"""

from vaedecon.cli.easy_train import (
    main,
    send_email_notification,
    load_email_settings,
    notify_completion,
)

if __name__ == "__main__":
    raise SystemExit(main())
