# CASPER: Learning Structure-Preserving Embedding Alignment

**C**ross-encoder **A**lignment with **S**tructure-**P**reserving **E**mbedding p**R**ojection

CASPER projects embeddings produced by a *source* encoder into the native space of a *target* encoder while explicitly preserving the relational geometry of the source. It supports encoders with different dimensionalities, trains in ~1/8 of the wall-clock time of adversarial baselines (vec2vec), and reaches Structure Preservation (SP) of 0.93–0.99 across 20 encoder pairs versus 0.48–0.73 for vec2vec.

The same recipe transfers, unchanged, to the residual streams of autoregressive decoders (cross-family bridging, e.g. Qwen3-0.6B → Llama-3.2-1B).

> Paper: *CASPER: Learning Structure-Preserving Embedding Alignment* (under review, ARR / EACL 2027). Trained mappers will be released on Hugging Face upon acceptance.

---

## Two inference variants, one backbone

| Variant | Inputs at inference | Script | Mapper class |
|---|---|---|---|
| **CASPER** (default, source-only) | `z_s` | `ALIGNMENT_OFFICIAL_diff_dim_linear_unsupervised.py` | `StandaloneMapper` |
| **CASPER-TC** (target-conditioned) | `z_s`, `z_t` | `ALIGNMENT_OFFICIAL_diff_dim_linear.py` | `LinearInjectionMapper` |
| **CASPER-TC on decoders** | `h_s`, `h_t` (last-layer residuals) | `ALIGNMENT_OFFICIAL_diff_dim_linear_decoder.py` | `LinearInjectionMapper` |

Both variants share the same residual-MLP backbone and the same multi-objective loss; they differ only at the injection step:

```
h_0 = W_s z_s            (CASPER)
h_0 = W_s z_s + W_t z_t  (CASPER-TC)
h_{l+1} = h_l + FFN_l(LN(h_l))        l = 1..L   (L = 4 encoders, 5 decoders)
ẑ_t = LN(W_o LN(h_L))
```

Each FFN block is `LayerNorm → Linear(d, 4d) → GELU → Dropout → Linear(4d, d) → Dropout`. The reverse mapper `g_φ : Z_t → Z_s` is defined symmetrically; both directions are trained jointly.

CASPER-TC is **not** a passthrough: on every pair its SP (0.93–0.99) clears the target-copy ceiling `f(z_s, z_t) = z_t` (0.47–0.69, paper Appendix H), and its residual `Δ = 1 − cos(ẑ_t, z_t)` is a usable cross-model disagreement signal.

## Training objective

Five losses survive ablation (Random-Forest permutation importance + Shapley attribution over 8 setups). Final weights, selected by hierarchical grid search over 25 configurations:

```
L = 0.375 · L_struct + 0.1875 · L_Macro + 0.1875 · L_Micro + 0.1875 · L_tlm + 0.0625 · L_con
```

| Term | Code key | What it constrains |
|---|---|---|
| `struct` | `structure` | Pearson correlation between source and mapped pairwise cosine-distance vectors (200-point subsample per step) |
| `Macro` | `global` | L1 between full batch distance matrices, source vs mapped |
| `Micro` | `local` | L1 restricted to each anchor's *k* = 200 nearest source neighbours |
| `tlm` | `tlm` | Symmetric masked reconstruction (Bernoulli retention p = 0.15, loss on retained coordinates) |
| `con` | `con` | Bidirectional NT-Xent (τ = 0.05) between mapped and target |

`mse` and `cycle` are implemented but weighted 0 in the official configuration. All terms are computed in both directions and summed.

## Repository structure

*(adapt to the actual tree)*

```
CASPER/
├── ALIGNMENT_OFFICIAL_diff_dim_linear.py               # CASPER-TC, encoders
├── ALIGNMENT_OFFICIAL_diff_dim_linear_unsupervised.py  # CASPER (source-only), encoders
├── ALIGNMENT_OFFICIAL_diff_dim_linear_decoder_v2.py    # CASPER-TC, decoder residual streams
├── nq_reservoir.db                                     # built on first run (encoders)
├── nq_reservoir_resid.db                               # built on first run (decoders)
└── saved_mappers_<run_tag>/                            # outputs, one folder per run
```

## Installation

```bash
python -m venv casper_env && source casper_env/bin/activate
pip install torch transformers sentence-transformers datasets \
            numpy scipy pandas matplotlib seaborn tqdm
```

Tested with Python 3.12, PyTorch ≥ 2.1, a single 48 GB GPU (RTX A6000). Some Hugging Face models (Qwen3-Embedding, Granite) require `trust_remote_code=True`, already set in the loaders. Gated models (Llama-3.2) need a logged-in `huggingface-cli`.

## Quick start

### 1. Encoder alignment

Edit the model pair at the top of the script, then run:

```python
SRC_NAME = "sentence-transformers/gtr-t5-base"
TGT_NAME = "intfloat/e5-base-v2"
run_tag  = "official_gtr_e5_1M_MacroStr_SimTlm"
```

```bash
# CASPER-TC (target-conditioned)
PYTHONUNBUFFERED=1 python ALIGNMENT_OFFICIAL_diff_dim_linear.py

# CASPER (source-only)
PYTHONUNBUFFERED=1 python ALIGNMENT_OFFICIAL_diff_dim_linear_unsupervised.py
```

Pipeline on first run:
1. Streams `jxm/nq_corpus_dpr` (train split), chunks into 64-token windows, stores 1 M chunks in a SQLite reservoir (`nq_reservoir.db`, reused across runs).
2. Pre-computes source and target embeddings (mean-pooled for HF models, native pooling for SentenceTransformer models).
3. Trains both mappers with AdamW (η = 1e-4, batch 256), early stopping (δ = 1e-5, patience 3) on the epoch-averaged objective, restoring the best in-memory state.
4. Evaluates on 65,536 held-out NQ dev passages: mean cosine, Top-1, mean rank, SP (Spearman over pairwise cosine distances).

The source-only script accepts `SKIP_TRAINING_IF_EXISTS = True`: if both checkpoints are already in `OUT_DIR`, it probes the embedding dimensions, reloads the weights and runs evaluation only.

### 2. Using a trained mapper

```python
import torch
from ALIGNMENT_OFFICIAL_diff_dim_linear_unsupervised import StandaloneMapper

m = StandaloneMapper(input_dim=768, output_dim=768, hidden_dim=768, n_layers=4)
m.load_state_dict(torch.load("saved_mappers_<run_tag>/official_mapper_src2tgt.pt"))
m.eval()

z_hat_t, _ = m(z_s)                       # CASPER
# z_hat_t, _ = m_tc(z_s, memory=z_t)      # CASPER-TC (LinearInjectionMapper)
```

`hidden_dim` must match the one used at training time (see *Configuration*).

### 3. Decoder bridging (CASPER-TC on residual streams)

```bash
PYTHONUNBUFFERED=1 python ALIGNMENT_OFFICIAL_diff_dim_linear_decoder_v2.py
```

Extracts last-layer, last-token **pre-norm** residuals of `Qwen/Qwen3-0.6B` and `meta-llama/Llama-3.2-1B` on 2 M NQ chunks into on-disk memmaps, standardises each side by its empirical std, trains a 5-block bridge, and saves a self-describing checkpoint:

```python
{'s2t': ..., 't2s': ..., 'ss': σ_s, 'ts': σ_t,
 'arch': 'LinearInjectionMapper',
 'config': {'hidden_dim', 'n_layers', 'n_heads', 'd_src', 'd_tgt'},
 'loss_config': {...}}
```

At inference the projected residual must pass through the target's final `RMSNorm` before `lm_head` (Eq. 15 of the paper); the script ends with a 4-way generation demo showing exactly this. Merging recipes (Hybrid-α, SLERP) and the hijacking / LLM-as-a-Judge evaluations of Appendix N build on this checkpoint.

## Configuration

All hyperparameters are module-level constants at the top of each script.

| Constant | Encoder scripts | Decoder script |
|---|---|---|
| `SRC_NAME` / `TGT_NAME` (`SRC_ID` / `TGT_ID`) | HF model ids | HF model ids |
| `SUBSAMPLE` | 1,000,000 | 2,000,000 |
| `CHUNK_TOKENS` | 64 | 64 |
| `TRAIN_BATCH` / `LR` | 256 / 1e-4 | 256 / 1e-4 |
| `MAPPER_LAYERS` | 4 | 5 |
| `hidden_dim` | `max(D_src, D_tgt)` | `max(D_src, D_tgt)` |
| `PATIENCE` / `MIN_DELTA` | 3 / 1e-5 | 3 / 1e-5 |
| `AB_EVAL_SIZE` | 65,536 | — |

The loss weights live in `official_weights` / `OFFICIAL_WEIGHTS`; set `mse` or `cycle` > 0 to re-enable the ablated terms.

## Outputs

Each run writes to `saved_mappers_<run_tag>/`:

| File | Content |
|---|---|
| `official_mapper_src2tgt.pt`, `official_mapper_tgt2src.pt` | mapper state dicts |
| `official_metrics.csv` | cosine / Top-1 / mean rank / SP per direction |
| `run_meta.json`, `param_counts.json`, `wall_clock.json` | reproducibility record |
| `mapper_timing_train.csv`, `mapper_timing_train_summary.csv` | per-batch and per-epoch fwd+bwd timing |
| `mapper_timing_eval.csv` | inference throughput per direction |

## Models used in the paper

| Role | Model | Dim | HF id |
|---|---|---|---|
| Encoder | GTR | 768 | `sentence-transformers/gtr-t5-base` |
| Encoder | e5 | 768 | `intfloat/e5-base-v2` |
| Encoder | Granite | 768 | `ibm-granite/granite-embedding-278m-multilingual` |
| Encoder | GTE | 768 | `thenlper/gte-base-v2` |
| Encoder | Stella | 768 | `infgrad/stella-base-en-v2` |
| Encoder | Qwen3-Embedding-0.6B | 1024 | `Qwen/Qwen3-Embedding-0.6B` |
| Encoder | GTE-Large | 1024 | `thenlper/gte-large` |
| Encoder (domain) | FinBERT | 768 | `ProsusAI/finbert` |
| Decoder | Qwen3-0.6B | — | `Qwen/Qwen3-0.6B` |
| Decoder | Llama-3.2-1B | — | `meta-llama/Llama-3.2-1B` |
| Decoder | Gemma-4-E4B-it | — | `google/gemma-4-E4B-it` |
| Judge | Qwen2.5-1.5B-Instruct | — | `Qwen/Qwen2.5-1.5B-Instruct` |

## Headline results

NQ dev, 65,536 passages, 20 encoder pairs (paper Table 2):

| | Cos | Top-1 | Mean rank | SP |
|---|---|---|---|---|
| vec2vec | 0.74–0.92 | 0.82–1.00 | 1.0–5.2 | 0.48–0.73 |
| **CASPER** | 0.86–0.94 | 0.99–1.00 | 1.0 | 0.94–0.99 |
| **CASPER-TC** | 0.90–0.98 | 1.00 | 1.0 | 0.93–0.99 |

Median training time: 5h50m (CASPER) and 5h29m (CASPER-TC) vs 48h19m (vec2vec) on one RTX A6000. SP is validated as a predictor of neighbourhood retention (Spearman(SP, Jaccard@10) = 0.97, Appendix J).

Downstream applications (Section 6 / Appendices K–N): adversarial detection on AdvGLUE, temporal-drift detection, domain-specialisation detection, cross-family decoder bridging with strict Pareto gains over both parents on TruthfulQA and NQ-Open.

## Citation

```bibtex
@inproceedings{casper2027,
  title     = {CASPER: Learning Structure-Preserving Embedding Alignment},
  author    = {Anonymous},
  booktitle = {Under review},
  year      = {2027}
}
```

## License

TBD.
