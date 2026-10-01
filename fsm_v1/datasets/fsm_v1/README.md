# FSM v1 dataset

This directory is built deterministically from the reviewed snapshot embedded in
`notebooks/Robot_SLM_Adaptive_FSM_Inventory_Query(AAA).ipynb`. It supports a
finite-state inventory-location model without using an LLM, pretrained encoder,
or learned location facts at runtime. The similarity fallback uses sparse
TF-IDF embeddings learned from the training split.

## Build

From the repository root:

```bash
python3 scripts/build_fsm_dataset.py
```

The builder uses only the Python standard library. It finds the notebook cell by
the `EMBEDDED_DATA` and `EXPECTED_SHA256` assignments rather than by cell index,
decodes the three zlib/base64 payloads, and verifies their SHA-256 hashes against
both the notebook declarations and hashes pinned in the builder. A mismatch
stops the build.

The builder never reads or writes `datasets/original/`.

## Outputs

- `train.jsonl`: the only file used to compile the FSM. It contains labeled
  query patterns and labeled controller transitions, with every row carrying
  its source kind and provenance.
- `validation.jsonl` and `test.jsonl`: compiler-schema query rows retained for
  auditing; neither file is read during compilation.
- `inventory.json`: the runtime inventory. Its globally unique IDs use
  `setting:item_id`, while `source_item_id` preserves the reviewed source key.
- `heldout.jsonl`: compiler-independent acceptance cases consumed only after
  the model has been compiled.
- `similarity/{validation,test}.jsonl`: generated evaluation-only cases for
  omitted articles, an `item` prefix, misspellings, ambiguity, missing objects,
  and unrelated requests. They are excluded from compilation.
- `original_derived/{train,validation,test}.jsonl`: the 25 reviewed retrieval
  cases only, split by fold 0 participant IDs (16 train, 4 validation, 5 test).
  Source utterances are unchanged. The conversion only maps a reviewed
  `locate` target and verified drawer to the FSM labels.
- `generated/{train,validation,test}.jsonl`: deterministic scenarios kept
  separate from participant data (36 train, 18 validation, 18 test).
- `inventory/original_inventory.json`: all 45 reviewed lab/hospital inventory
  rows, including their evidence metadata.
- `heldout/examples.jsonl`: four untouched acceptance dialogues. These include
  `I need tape` → `ASK_WHICH_ONE`, `Blue` → `RETURN_LOCATION`, `I need a pen`
  → `OBJECT_MISSING`, one unique lookup, and one missing-location lookup.
- `manifest.json`: source hashes, counts, scenario distributions, artifact
  hashes, and the held-out exclusion result.

The source `item_id` is unique only within a setting. Audit rows therefore use
`(setting, item_id)`, and runtime rows expose a globally unique
`setting:item_id` ID.

## Compile and evaluate

```bash
python3 -m fsm_only compile datasets/fsm_v1/train.jsonl \
  -o models/fsm_v1/model.json
python3 scripts/evaluate_fsm.py \
  --model models/fsm_v1/model.json \
  --inventory datasets/fsm_v1/inventory.json \
  --heldout datasets/fsm_v1/heldout.jsonl
python3 -m unittest discover -s tests -v
```

The unit suite rebuilds every generated artifact byte-for-byte, verifies source
hashes and split isolation, recompiles the model from `train.jsonl`, and runs
all held-out, validation, exact-match, and cosine-similarity conversations.

## Record schema

Each JSONL row is one labeled dialogue:

- `source_kind`: `original_derived`, `generated`, or `heldout`.
- `split`: `train`, `validation`, `test`, or `heldout`.
- `scenario`: `unique`, `ambiguous`, `missing_object`, `missing_location`,
  `clarification`, or `unsupported`.
- `inventory_overrides`: generated counterfactual changes. Missing-location
  rows set only the selected row's `drawer` to `null`; the reviewed inventory
  file is never changed.
- `turns`: ordered FSM transitions. Each transition contains its state before,
  utterance, finite input symbols, deterministic lookup observation, candidate
  items, target action, next state, and structured output fields.
- `source_context`: the complete reviewed source context for original-derived
  rows and `null` otherwise.
- `provenance`: whether and how the row was derived or generated.

Audit states are `WAITING_FOR_QUERY`, `WAITING_FOR_CLARIFICATION`, and `DONE`.
Target actions are `RETURN_LOCATION`, `ASK_WHICH_ONE`, `OBJECT_MISSING`,
`LOCATION_MISSING`, and `UNSUPPORTED_REQUEST`. Lookup observations are
`ONE_WITH_LOCATION`, `ONE_WITHOUT_LOCATION`, `MULTIPLE_MATCHES`,
`ZERO_MATCHES`, and `NOT_APPLICABLE`.

## Data boundaries

The original-derived portion contains only the 25 retrieval cases from the
118-case reviewed package. The 93 recovery cases are intentionally excluded.
The source annotations are marked `provisional_text_review`; this dataset does
not upgrade that review status.

Generated examples are labeled as generated and never represented as original
participant observations. They cover all six required behaviors independently
in every generated split. Held-out utterances and complete held-out rows are
checked against every train, validation, and test row before files are written.
The held-out file must remain test-only.

Similarity thresholds are declared in the training configuration. They are
tuned against `similarity/validation.jsonl`; `similarity/test.jsonl` remains
evaluation-only. The sparse embedding vocabulary and IDF values in the compiled
artifact are fitted only from query patterns in `train.jsonl`.
