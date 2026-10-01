# Cache v2: progressive FSM

`ProgressiveFSMRouter` learns FSM transitions from SLM answers on 972 generated
requests in three batches and draws the three-panel convergence figure.

| File | Role |
|---|---|
| `adaptive_fsm.py` | router and promotion rules |
| `fsm_small_data.py` | request generator |
| `fsm_experiment.py` | stream runner and figure |
| `notebooks/fsm_convergence_small.ipynb` | Colab notebook |

Open the notebook in Colab with a GPU and a trained Qwen3 bundle from
`../slm_qwen3`, then run all cells. SLM answers are cached, so a restart
continues where it stopped.

Replaced by `../cache_v3`, which uses recorded trials instead of generated
requests.
