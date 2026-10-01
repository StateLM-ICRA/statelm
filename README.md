# StateLM

Working repository for the crash-cart dialogue system. A fine-tuned small
language model (SLM) handles unfamiliar requests and failure recovery, a
finite-state machine (FSM) answers familiar requests, and the caching policy
turns repeated SLM decisions into new FSM transitions.

This is the private working copy with every version found so far. No code was
rewritten while organizing it. Code that lived inside notebooks
(`%%writefile` cells and base64-embedded archives) was copied into `.py` files
byte for byte, and each copy was checked against its source (see
`VERSIONS.md`). The original notebooks and files are kept unchanged in `archive/`.

## Folders

| Folder | Content | SLM | FSM | Cache |
|---|---|---|---|---|
| `gemma2b/` | StateLM v4.1: hybrid agent, recovery retraining, playground, two simulations | Gemma-2-2B-it, QLoRA, scores a fixed set of choices | authored transitions, exact inventory match | none |
| `qwen3_slm/` | SLM training, runtime and evaluation | Qwen3-4B, QLoRA, writes a JSON decision | none | none |
| `qwen3_dialogue_fsm/` | Qwen3 SLM inside a small hand-written FSM | Qwen3-4B | hand-written `DialogueFSM` | none |
| `fsm_baseline/` | similarity-augmented FSM from the original repository (review copy) | none | compiled from data and rules | none |
| `fsm_success_derived/` | FSM induced from the 41 success trials with BGE embeddings (review copy) | none | data-induced | none |
| `cache_v1_candidate/` | first FSM cache | Qwen3-4B | seed rules | candidate cache with promotion |
| `cache_v2_progressive/` | progressive FSM on 972 generated requests | Qwen3-4B | learned | progressive promotion |
| `cache_v3_realstream/` | Fig. 2 on the real deployment stream | Qwen3-4B | FSM_0 from the success trials | state-aware semantic cache |
| `datasets/` | labels, recovery preferences, behavior supplements v1 and v2 | | | |
| `archive/` | originals as received | | | |

The ICRA submission describes the Gemma-2-2B model. Fig. 2 is being redone with
`cache_v3_realstream`, whose teacher is Qwen3-4B.

## One interaction

An interaction is one item request. It may contain one clarification or one
failure, then one recovery, and then it ends. The model input for every turn has
to contain all earlier turns of the same interaction, so "I need a pencil",
"Which color?", "blue" is read as a request for the blue pencil.

| Version | What the SLM receives | Note |
|---|---|---|
| `gemma2b` | the whole conversation, not truncated | its FSM does not store the pending item family, so a bare "blue" is left to the SLM |
| `qwen3_slm`, `qwen3_dialogue_fsm` | last 8 turns plus dialogue memory (pending family, known and missing attributes) | the session closes after one recovery; `/reset` starts the next interaction |
| `cache_v3_realstream` stream | the recorded turns of the trial up to the decision | the stream has no answers to clarification questions |
| `cache_v3_realstream` chat | last 8 turns plus dialogue memory | history is not cleared between interactions until `/reset` |

