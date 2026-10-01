# FSM-only baseline

A deterministic dialogue FSM for item requests. Query patterns and transitions
are compiled from the labeled training split. When no pattern matches exactly,
the request is compared with training prototypes by TF-IDF cosine similarity.
It locates items, asks which one when several match, reports items it does not
have, and declines unrelated requests. No language model is used.

Python 3.10 or newer, no extra packages. Run from this folder.

Chat:

```bash
python3 -m fsm_only chat models/fsm_v1/model.json \
  datasets/fsm_v1/inventory.json --setting lab --json
```

Compile the model from the training split:

```bash
python3 -m fsm_only compile datasets/fsm_v1/train.jsonl \
  -o models/fsm_v1/model.json
```

Evaluate on the held-out similarity cases:

```bash
python3 scripts/evaluate_fsm.py \
  --model models/fsm_v1/model.json \
  --inventory datasets/fsm_v1/inventory.json \
  --heldout datasets/fsm_v1/similarity/test.jsonl
```

Tests: `python3 -m unittest discover -s tests`. 19 of 20 pass. The failing
test rebuilds the dataset from the notebook in `notebooks/` and compares file
hashes; the notebook no longer carries the external source links, so the
hashes differ. Compilation from the included training split is unaffected.
