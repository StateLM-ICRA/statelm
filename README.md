# StateLM

Code and data for the paper **Low Inference Conversational Robots: Convergent
Caching of Generative Behavior into Deterministic Dialogue Control**
(under review). Project page: https://statelm-icra.github.io

StateLM is a dialogue controller for a voice-guided crash-cart robot. A
finite-state machine (FSM) answers familiar requests in about a millisecond, a
fine-tuned small language model (SLM) handles unfamiliar requests and failure
recovery, and a state-aware semantic cache turns verified, recurring SLM
decisions into new FSM transitions during deployment. Over time the FSM
absorbs the recurring interaction patterns and the SLM is called less often.

## Repository structure

```
statelm/
  slm_qwen3/          SLM on Qwen3-4B: QLoRA training, runtime, evaluation
  slm_gemma2b/        SLM on Gemma-2-2B: QLoRA recovery training, hybrid agent, playground
  fsm_handcrafted/    FSM-only baseline compiled from labeled requests
  fsm_data_induced/   FSM_0 induced from successful trials with sentence embeddings
  hybrid_qwen3_fsm/   Qwen3 SLM with a hand-written dialogue FSM, no cache
  cache_v1/           caching policy v1: candidate patterns with promotion
  cache_v2/           caching policy v2: progressive FSM on generated requests
  cache_v3/           caching policy v3: state-aware semantic cache on the real stream (Fig. 2)
  datasets/           trial labels, recovery preferences, generated behavior supplements
```

Each folder is self-contained and has its own README with the commands to
run it. The three caching versions are kept so that the development can be
followed; `cache_v3` is the one reported in the paper.

## Installation

The FSM baselines and the cache replay need only Python 3.10 or newer, with
no third-party packages.

The SLM training and inference notebooks run in Google Colab on a GPU
runtime. Open a notebook from the `notebooks/` folder of a version, choose a
GPU under Runtime > Change runtime type, and run the cells in order. The
pinned package versions are installed by the first cell. Gemma-2-2B requires a
Hugging Face token with access to the model, stored as the Colab secret
`HF_TOKEN`.

For local use, each SLM folder has a `requirements` file:

```bash
cd slm_gemma2b
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-colab.txt
```

## Quick start

Talk to the FSM-only baseline (no GPU, no packages):

```bash
cd fsm_handcrafted
python3 -m fsm_only chat models/fsm_v1/model.json \
  datasets/fsm_v1/inventory.json --setting lab --json
```

Run the FSM path of the Gemma hybrid agent on one request:

```bash
cd slm_gemma2b
python -m statelm demo "Where is blue tape?" --cart lab
```

Build the deployment stream for Fig. 2 and run the controller tests:

```bash
cd cache_v3
python scripts/build_stream.py --out build
python -m unittest discover -s tests
```

The full Fig. 2 experiment (SLM pass over the stream, replay, figure and
sensitivity table) is in `cache_v3/notebooks/StateLM_RealStream_Fig2.ipynb`.

## Tests

```bash
cd cache_v3        && python -m unittest discover -s tests   # 12 tests
cd fsm_handcrafted && python -m unittest discover -s tests   # 20 tests
cd slm_gemma2b     && python -m unittest tests.test_core     # 23 tests
```

The other test files in `slm_gemma2b/tests` need PyTorch and a GPU.

## Data

The dialogues come from recorded crash-cart retrieval trials with 41
participants (RFM-HRI). `datasets/original/` holds the per-trial labels with
transcripts (`authoritative_labels.jsonl`) and the participants' recovery
preferences from the post-trial survey (`recovery_preferences.jsonl`). Each
version keeps the reviewed cases it was trained or evaluated on in its own
`data/` folder (`original_cases.json`: 25 retrieval and 93 recovery cases).
Participants are identified by number only.

The recordings contain successful requests and robot failures but no
clarification questions, unavailable items or off-topic requests. Those three
behaviors are covered by LLM-written requests that the authors checked by
hand. They are kept in separate files (`datasets/generated/` and
`cache_v3/data/augmented_requests_v1.jsonl`) and are reported separately
from the recorded data.

## Models

| Model | Base | Adaptation | Used for |
|---|---|---|---|
| `slm_qwen3` | Qwen/Qwen3-4B | 4-bit QLoRA, rank 8 | Fig. 2, caching experiments |
| `slm_gemma2b` | google/gemma-2-2b-it | 4-bit QLoRA, rank 8 | user study on the Raspberry Pi 5, videos |

The base models are public. The trained LoRA adapters will be released after
the review.

## License and citation

The license and the citation will be added after the review.
