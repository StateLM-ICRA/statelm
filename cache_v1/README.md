# Cache v1: candidate cache with promotion

First version of growing the FSM from SLM answers. A request that no trusted
pattern matches goes to the SLM. Correct SLM decisions are stored as candidate
patterns (state, request pattern, decision template), tested in shadow on later
requests, and promoted into the FSM after repeated, reliable confirmation.
Drawers are never cached; they are read from the inventory.

`adaptive_router.py` is the router cell of
`notebooks/adaptive_cache_eval.ipynb`. It uses the thresholds and the loaded
SLM runtime defined in the earlier cells, so run it through the notebook.
The evaluation stream was a generated set of 240 requests and is not included.
