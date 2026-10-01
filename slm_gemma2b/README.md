# SLM on Gemma-2-2B

Gemma-2-2B-it with 4-bit QLoRA (rank 8, alpha 16, dropout 0.05, learning rate
2e-5, up to 12 epochs with early stopping). The model does not write free text.
It scores a fixed set of choices: an inventory item, clarify, unknown or
acknowledge for item resolution, and one of eight recovery strategies after a
failure. The prompt contains the whole conversation of the interaction. Drawers
come from the inventory and replies from templates.

Recovery training puts probability on any strategy the participant preferred,
with a KL anchor to the base model. On validation, the adapter is used for
recovery only if it beats the base model.

## Files

| Path | Role |
|---|---|
| `statelm/core.py` | inventory, FSM, hybrid agent and reply templates |
| `statelm/slm.py` | prompts, choice scoring, QLoRA training |
| `statelm/data.py`, `statelm/evaluation.py` | case preparation, folds, evaluation |
| `statelm/playground.py`, `statelm/notebook_ui.py` | interactive comparison of FSM, base SLM, trained SLM and hybrid |
| `statelm/simulations.py`, `simulation_ui.py` | the two saved-model simulations |
| `data/` | reviewed cases, inventory, participant folds |
| `notebooks/retrain_and_playground.ipynb` | Colab notebook for training and the playground |

## Run

Colab: open the notebook with a GPU runtime and add a Hugging Face token with
access to Gemma as the Colab secret `HF_TOKEN`.

Locally, with Python 3.12 and a CUDA GPU:

```bash
pip install -r requirements-colab.txt
python -m statelm demo "Where is blue tape?" --cart lab
python -m statelm train --output runs/recovery --folds 0 1 2 3 4 --seeds 42
python -m unittest tests.test_core
```

The `demo` command uses only the FSM and needs no GPU. `tests.test_core` runs
on a CPU; the other test files need PyTorch.
