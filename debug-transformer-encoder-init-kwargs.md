# Debug session: transformer-encoder-init-kwargs

Session ID: transformer-encoder-init-kwargs

Status: [OPEN]

## Symptom

When training a config that uses `GeneTransformerEncoder` inside a multi-encoder
VAE setup, model construction fails at:

```text
TypeError: GeneTransformerEncoder.__init__() got an unexpected keyword argument 'data_config'
```

The stack trace points to:
- `train.py -> _create_model()`
- `workflow.py -> create_model()` calling
  `encoder_registry[encoder_name](**kwargs)` with `data_config` in kwargs.

This was reported while launching the dual-encoder YAML derived from ablation 30.

## Hypotheses

1. `GeneTransformerEncoder.__init__` signature was refactored recently and no
   longer accepts `data_config`, while `create_model()` still passes it
   unconditionally to every encoder.
2. A subset of encoders (notably pathway-aware ones) require `data_config`, so
   the workflow correctly includes it, but the Transformer encoder forgot to
   keep a matching optional parameter.
3. The encoder registry / kwargs construction path changed behavior between the
   local commit and the deployed remote checkout, introducing a version skew
   where one side passes `data_config` and the other does not accept it.
4. `EncoderPathNet` was not the only data-config-dependent encoder; additional
   encoder types now depend on `data_config`, so the workaround should be a
   signature-level compatibility layer rather than a per-encoder special case.
5. The recent Transformer internal refactor removed or reshuffled parent class
   delegation so `data_config` is no longer absorbed by `BaseEncoder` or another
   shared constructor.

## Current evidence

- Static traceback shows failure at `workflow.py:108` inside the per-encoder
  loop.
- We already know pathway encoders depend on `data_config`, so removing it
  globally is unsafe.
- The immediate failure is on `GeneTransformerEncoder`, suggesting a signature
  incompatibility rather than a logic error in the Transformer forward pass.

## Planned instruments

- Add focused logging inside `workflow.create_model` to report:
  - encoder name
  - keys in `kwargs`
  - explicit acceptance of `data_config` via introspection
- Add the same report inside each encoder entry used in hybrid runs to confirm
  parameter consumption.

## Fix direction under evaluation

- Make `GeneTransformerEncoder.__init__` accept an optional `data_config`
  keyword explicitly and ignore it when unused, mirroring the convention used
  by encoders that do not need dataset metadata.
- Optionally align other non-pathway encoders similarly if they also lack the
  parameter, to prevent recurrence in future 3/4-encoder hybrids.

## Status notes

- [x] Instrumentation deployed
- [x] Evidence collected
- [x] Candidate fix applied
- [x] Post-fix verification logs captured
- [ ] User confirmed resolution
- [ ] Debug artifacts cleaned up

## Evidence log

### Pre-fix evidence

From `workflow.create_model -> before_encoder_init`:

1. EncoderMLP reported:
   - `declares_data_config: true`
   - `init_params: ["self", "args", "data_config", "position_encoding"]`
   - constructor called successfully
2. GeneTransformerEncoder reported:
   - `declares_data_config: false`
   - `init_params: ["self", "args", "position_encoding"]`
   - kwargs included `"data_config"`
   - immediate failure:
     `TypeError: GeneTransformerEncoder.__init__() got an unexpected keyword argument 'data_config'`

Interpretation:
- Hypothesis 1 confirmed: the recent transformer refactor dropped `data_config`
  from the constructor while the workflow still passes it universally.
- Hypothesis 2 confirmed: pathway/data-aware encoders such as EncoderPathNet
  require `data_config`, so removing it from `kwargs` is not safe.
- Hypothesis 3 not supported locally: the same code revision reproduces the
  failure without needing deployment skew.
- Hypothesis 4 partially validated in follow-up static check:
  EncoderResMLP also lacked `data_config`, so a general compatibility
  signature is needed across encoders, not just a Transformer one-off.
- Hypothesis 5 not required: the issue is confined to constructor signature
  mismatch, not BaseEncoder delegation behavior.

### Candidate fix

Made the following minimal changes so structure-only encoders remain compatible
with the universal `create_model` kwargs:

1. `vaedecon/models/nn/transformer.py`
   - `GeneTransformerEncoder.__init__(..., data_config=None, ...)`
   - store `self.data_config = data_config`

2. `vaedecon/models/nn/res_mlp.py`
   - `EncoderResMLP.__init__(..., data_config=None, ...)`
   - store `self.data_config = data_config`

Pathway/SGNN encoders keep their existing signatures unchanged because they
actually use `data_config`.

### Post-fix evidence

Same reproducer and instrumentation after patch:

1. EncoderMLP reported:
   - `declares_data_config: true`
2. GeneTransformerEncoder reported:
   - `declares_data_config: true`
   - `init_params: ["self", "args", "data_config", "position_encoding"]`
3. Reproducer completed model creation successfully:
   - `MODEL_CREATED_OK VAE`

### Regression runs

- `python3 -m py_compile vaedecon/models/nn/transformer.py vaedecon/models/nn/res_mlp.py vaedecon/workflow/workflow.py ...`
  - result: `PY_COMPILE_OK`
- `pytest -q tests/test_config_yaml.py tests/test_transformer_encoder.py tests/test_cell_prop_prediction.py`
  - result: `82 passed`
