# Similarity-augmented FSM inventory locator

This package compiles labeled JSONL into two deterministic learned artifacts:

1. a minimized query-pattern finite-state transducer; and
2. a controller transition table learned from labeled `(state, event, action, next_state)` examples.

It uses only the Python standard library. It contains no LLM, pretrained
encoder, network call, randomness, or hidden object-specific rule. The compiler
learns sparse TF-IDF word/character n-gram embeddings from training patterns,
and the runtime uses cosine similarity only when exact FSM traversal fails.
Inventory names, aliases, attributes, settings, and locations remain separate
runtime data.

The controller is still a finite-state machine. Similarity maps previously
unseen wording to a learned discrete intent and object span; it does not choose
the dialogue action or next state.

## Labeled JSONL schema

Each line is one of these record types:

```json
{"type":"config","start_state":"READY","entity_slot":"object"}
{"type":"query","text":"where is the blue tape","slots":{"object":"blue tape"},"intent":"FIND_LOCATION"}
{"type":"query","pattern":["find","{object}"],"intent":"FIND_LOCATION"}
{"type":"transition","state":"READY","event":"FIND_LOCATION","action":"LOOKUP","next_state":"LOOKUP"}
{"type":"transition","state":"LOOKUP","event":"ONE_WITH_LOCATION","action":"RETURN_LOCATION","next_state":"READY"}
```

Query rows may instead supply equal-length `tokens` and BIO-style `token_labels`. Compatible duplicates increase learned support. Conflicting labels fail compilation.

The required lookup events are `ZERO_MATCHES`, `ONE_WITH_LOCATION`, `ONE_WITHOUT_LOCATION`, and `MULTIPLE_MATCHES`. Their learned actions must pass the requested behavior invariants: `OBJECT_MISSING`, `RETURN_LOCATION`, `LOCATION_MISSING`, and `ASK_WHICH_ONE`. The clarification state reached by `MULTIPLE_MATCHES` must contain the same four labeled outcome transitions.

The compiler performs exact suffix-equivalent state minimization and stores both pre- and post-minimization state counts.

## Similarity fallback

Exact traversal remains the highest-confidence path. If no exact query path is
accepted, the runtime replaces each possible object span with a slot marker,
embeds that candidate template using IDF weights learned from `train.jsonl`, and
compares it with learned query-pattern embeddings using cosine similarity.

The same exact-first/cosine-second policy is used for inventory names and
aliases. Thresholds are stored in the training configuration and compiled model.
A request below threshold is not forced to the nearest inventory item.

Run chat with `--json` to inspect `query_match_method`, `query_similarity`,
`inventory_match_method`, and `inventory_similarity` for every response.

## Compile

```sh
python -m fsm_only compile labeled.jsonl -o learned_fsm.json
```

Module API:

```python
from fsm_only import compile_jsonl

model = compile_jsonl("labeled.jsonl", "learned_fsm.json")
```

## Inventory and runtime

Inventory input is a JSON array, an object with an `items` array, or JSONL. Required fields are `id` and `name`; optional fields are `aliases`, `family`, `attributes`, `setting`, and nullable `location`. Extra provenance fields are preserved by the source file but safely ignored. Legacy `drawer` is accepted when `location` is absent.

```python
from fsm_only import CompiledFSM, Inventory, InventorySession

model = CompiledFSM.load("learned_fsm.json")
inventory = Inventory.load("inventory.json", setting="lab")
session = InventorySession(model, inventory)

result = session.handle("Where is the tape?")
print(result.text)
if session.awaiting_clarification:
    print(session.reply("the blue one"))
```

Interactive use defaults to the `lab` setting. Pass an empty setting to load all settings:

```sh
python -m fsm_only chat learned_fsm.json inventory.json --setting lab
```

Lookup returns a location for one match, asks which object when multiple items
match, reports an unknown object when no candidate clears the threshold, and
reports missing location data for a known item whose location is null.
