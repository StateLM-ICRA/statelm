# Cache v3: state-aware semantic cache on the real stream

The version used for Fig. 2. The FSM starts as FSM_0, seeded from the 41
successful trials (34 request patterns, 34 review patterns). The deployment
stream is the 93 recovery trials in participant order. Each trial gives two
turns as the robot saw them: the request, and the review of the robot line
that failed in the recording (wrong drawer, "Open the drawer", "I did not
understand that request", or a delay). 36 LLM-written requests for
clarification, unavailable items and off-topic questions are interleaved and
reported separately, for 222 turns in total.

A cached transition holds a dialogue state, a typed request pattern, a
decision template and the next state. It is reused only in the state where it
was learned, and only if the item comes from the user's own words or the
dialogue memory and the drawer matches the inventory. A pattern is admitted
after 2 distinct observations. Embeddings: all-MiniLM-L6-v2. Thresholds:
merge similarity 0.78, routing similarity 0.82, admission reliability 0.85.

## Files

| Path | Role |
|---|---|
| `statelm_realstream/contract.py` | decision schema, validator, dialogue memory |
| `statelm_realstream/features.py` | typed slots and embeddings |
| `statelm_realstream/router.py` | routing, cache admission and reuse |
| `statelm_realstream/session.py` | chat session with injected robot lines |
| `scripts/build_stream.py` | FSM_0 seed data and the stream |
| `scripts/teacher_pass.py` | runs the SLM once per stream turn (GPU) |
| `scripts/replay_stream.py` | replays routing and learning from the SLM cache |
| `scripts/make_figure.py` | Fig. 2 |
| `notebooks/StateLM_RealStream_Fig2.ipynb` | all steps in Colab, plus a chat with the learned model |
| `results/run_v1/` | figure and summary of the main run |

## Run

In Colab, open the notebook with a GPU, set `BUNDLE_DIR` to a trained Qwen3
bundle from `../slm_qwen3` and `WORK_DIR` to an output folder, then run the
cells in order. Without a GPU:

```bash
python scripts/build_stream.py --out build
python -m unittest discover -s tests
```
