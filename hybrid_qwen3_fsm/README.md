# Qwen3 SLM with a hand-written FSM

The Qwen3 runtime from `../slm_qwen3` wrapped by a small hand-written
`DialogueFSM`. The FSM keeps a pending request across an unrelated turn and
resolves a short attribute answer, such as "blue" after "Which tape?",
directly from the inventory. Everything else goes to the SLM. There is no
cache.

| File | Role |
|---|---|
| `slm_runtime_fsm.py` | runtime with the FSM |
| `notebooks/chat_with_dialogue_fsm.ipynb` | loads a saved Qwen3 bundle and opens a text chat |
| `notebooks/train_with_dialogue_fsm.ipynb` | the training notebook with the FSM added |

Open a notebook in Colab with a GPU. In the chat notebook, set the path of a saved
Qwen3 bundle in step 2.
