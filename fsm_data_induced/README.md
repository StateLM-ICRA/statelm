# FSM induced from successful trials

Builds the initial FSM (FSM_0) from the 41 successful trials. Request phrases
and drawer mentions are extracted with spaCy, embedded with a frozen
`BAAI/bge-small-en-v1.5` encoder, and merged into states. The result is an
item and drawer table, request prototypes and the most frequent successful
reply. The clarification and unavailable-item controller in
`steps/16_manual_dialogue_controller.py` is written by hand.

The `steps/` files are the notebook cells in order; `run.py` runs them.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py --output runs/bge --backend bge
```

`--backend tfidf` uses sparse similarity instead of BGE. Participants are split
into train and held-out groups; `09_heldout_evaluation.py` reports request
recognition on the held-out group.
