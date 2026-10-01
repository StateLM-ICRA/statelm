# FSM v1 compiled model

`model.json` is a deterministic, standard-library-only similarity-augmented
FSM/FST artifact compiled exclusively from `datasets/fsm_v1/train.jsonl`. It
contains a learned query transducer, a learned controller transition table, and
sparse TF-IDF query embeddings compared by cosine similarity. It contains no
LLM, pretrained encoder, API call, or trained location fact; object locations
remain in the separate runtime inventory.

The committed evaluation reports cover:

- `heldout_results.json`: 4/4 conversations and 5/5 turns passed.
- `generated_validation_results.json`: 18/18 conversations and 21/21 turns
  passed.
- `generated_test_results.json`: 18/18 conversations and 21/21 turns passed.
- `original_validation_results.json`: 4/4 conversations and 4/4 turns passed.
- `original_test_results.json`: 5/5 conversations and 5/5 turns passed.
- `similarity_validation_results.json`: 5/5 conversations and 5/5 turns passed.
- `similarity_test_results.json`: 5/5 conversations and 5/5 turns passed.

Reproduce the model and all checks from the repository root:

```bash
python3 scripts/build_fsm_dataset.py
python3 -m fsm_only compile datasets/fsm_v1/train.jsonl \
  -o models/fsm_v1/model.json
python3 -m unittest discover -s tests -v
```
