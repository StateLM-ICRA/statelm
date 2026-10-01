# Qwen3 Robot SLM - final run 2026-09-16

This directory preserves the model, best training checkpoint, and evaluation outputs produced by `notebooks/SLM_only_final.ipynb`.

## Contents

- `model/`: deployable LoRA adapter and runtime bundle.
- `checkpoint/checkpoint-204/`: best resumable Trainer checkpoint, including optimizer, scheduler, RNG, and adapter state.
- `results/`: raw metric summaries, detailed predictions, paper-oriented CSV, and figures.
- `RESULTS.md`: concise interpretation of the measured results and their evidence boundaries.
- `MANIFEST.sha256`: SHA-256 checksum for every run artifact.
- `generate_results.py`: regenerates the summary CSV, figures, report, and checksum manifest.

## Model identity

- Base model: `Qwen/Qwen3-4B`
- Base revision: `1cfa9a7208912126459214e8b04321603b3df60c`
- Fine-tuning: QLoRA, rank 8, 3 epochs, learning rate 5e-5
- Best checkpoint: global step 204, evaluation loss 0.0219620019
- Base model weights: not included

The exported adapter and checkpoint-204 adapter have the same SHA-256 digest:

```text
4b6e4f76c761db9678af7e5f6fe7e7ccd788481fa88c6139211bfa6aacf15464
```

## Reproduce the results report

```bash
python3 runs/qwen3_robot_slm_20260916_001415/generate_results.py
```

Matplotlib is required only for regenerating the figures. Runtime dependencies are listed in `model/requirements.txt`.

## Git LFS

The adapter, optimizer state, tokenizer, and other large checkpoint files are stored with Git LFS. Clone with Git LFS enabled before attempting to load or resume the model.

## Evidence boundary

The metrics in this run evaluate the SLM decision component. They do not evaluate the complete StateLM router or the live participant study. The 22-record real held-out set is development evidence, and the 145-record supplemental behavior test is reported separately.
