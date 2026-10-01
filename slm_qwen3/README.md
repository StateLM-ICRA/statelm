# SLM on Qwen3-4B

Qwen3-4B fine-tuned with 4-bit QLoRA (rank 8, 3 epochs, learning rate 5e-5).
For every user turn the model returns one JSON decision with eight fields:
action, item family, resolved item, attributes, missing attributes, whether a
failure was detected, failure type and recovery strategy. The drawer is read
from the inventory, never generated, and the reply is rendered from a template.

## Files

| File | Role |
|---|---|
| `slm_runtime.py` | decision schema, validator, prompt and the `RobotSession` dialogue loop |
| `training_data.py` | builds training records from the reviewed cases and the behavior supplement |
| `slm_training.py` | tokenization, with the prompt masked out of the loss |
| `notebooks/train_qwen3_slm.ipynb` | Colab notebook: training, evaluation, export and a chat playground |
| `runs/` | configuration and evaluation of two trained runs |
| `RESULTS.md` | results of run `qwen3_robot_slm_20260916_001415` |

## Run

Open `notebooks/train_qwen3_slm.ipynb` in Colab with a GPU (an A100 was used)
and run the cells in order. `MODE = "inference_only"` with `RELOAD_BUNDLE`
set to a saved bundle skips training. The notebook expects the behavior
supplement at the path in `BEHAVIOR_SUPPLEMENT_PATH`; the files are in
`../datasets/generated/`.

A saved bundle can also be run from a terminal:

```bash
python slm_runtime.py --bundle path/to/robot_slm_bundle
```

## Runs

| Run | Behavior supplement | Notes |
|---|---|---|
| `qwen3_robot_slm_20260915_163730` | v1 | bundle configuration and evaluation |
| `qwen3_robot_slm_20260916_001415` | v2, adds unavailable items | results, predictions and figures in `runs/` |

The dialogue input holds the last 8 turns of the current interaction and the
dialogue memory (pending item family, known and missing attributes). An
interaction ends after one recovery; `/reset` starts the next one.
