#!/usr/bin/env python3
"""
OOD Evaluation of CASPER / CASPER-TC mappers — vec2vec protocol, ALL PAIRS
==========================================================================
Replicates the out-of-distribution evaluation of vec2vec (Jha et al., 2025,
"Harnessing the Universal Geometry of Embeddings", Table 3) on the CASPER
mappers, iterating over EVERY model pair for which checkpoints exist.

Datasets (vec2vec OOD protocol)
-------------------------------
  * TweetTopic : cardiffnlp/tweet_topic_multilingual, config 'en', split
                 'test' — the FULL test set (800 tweets), exactly as in the
                 vec2vec codebase (eval_attribute_extraction.py).

  * MIMIC      : vec2vec uses the Pseudo Re-identified MIMIC-III of Lehman
                 et al. (2021), which is CREDENTIALED (PhysioNet) and NOT on
                 the HF hub. vec2vec builds it locally with create_mimic.py
                 and evaluates on an 8192-record subset of the 'unsupervised'
                 split, with notes trimmed to 64 whitespace tokens.

                 This script tries, in order:
                   1) MIMIC_PATH (local load_from_disk), if present — the
                      faithful reproduction, 8192 records;
                   2) an HF fallback of public clinical notes, subsampled to
                      the SAME 8192 records and trimmed to 64 whitespace
                      tokens exactly as create_mimic.py does. This is a
                      SURROGATE (not the credentialed set); it is recorded in
                      the CSV under dataset key 'clinical_notes_hf' so results
                      are never silently conflated with the paper's MIMIC.

Sizes come straight from the vec2vec paper: TweetTopic rank spans 1..800,
MIMIC rank spans 1..8192 (so 8192 eval records). See Table 3.

Metrics (identical to evaluate_mapper_bidirectional + vec2vec Sec. 4.2):
  mean_cosine, top1_acc, mean_rank (over the whole eval set), plus a
  CASPER-specific structure_preservation (Spearman of pairwise-distance
  matrices).

Architecture is AUTO-DETECTED from each checkpoint's state_dict:
  memory_proj.weight present -> LinearInjectionMapper (CASPER-TC),
  absent -> StandaloneMapper (CASPER). hidden_dim / n_layers / dims are read
  from tensor shapes, so both hidden_dim=1024 and hidden_dim=max(D_src,D_tgt)
  checkpoints load with no manual flags.

Missing checkpoints are non-fatal: the pair/variant is skipped with a
warning and the run continues. Embeddings are computed once per (dataset,
model) and cached, so many pairs sharing an encoder pay the encoding cost
only once.
"""

import os
import json
import time
import itertools
from collections import OrderedDict
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from datasets import load_dataset, load_from_disk

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# -------------------------
# Config
# -------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# -------------------------------------------------------------------------
# Model pairs = EXACTLY the 5 encoders vec2vec uses in its OOD table
# (Table 3): e5, granite (gra), gte, gtr, stella. All 10 unordered pairs.
# finbert / qwen3 / gte-large (gtel) / biobert are intentionally EXCLUDED —
# they are not in vec2vec's table.
#
# The directory naming convention is:
#   saved_mappers_official_{src}_{tgt}_1M_MacroStr_SimTlm_{variant}
# so 'src'/'tgt' below MUST follow the directory order (the src2tgt mapper
# maps src->tgt); feeding the wrong side would silently produce garbage
# because every encoder here is 768-dim (a dim check can't catch swaps).
#
# variant -> ckpt suffix:
#   casper    (source-only, StandaloneMapper)      -> _unsupervised
#   casper_tc (target-conditioned, LinearInjection)-> _linear
# The actual architecture is auto-detected from the checkpoint; the tag is
# informational and used only for the CSV.
# -------------------------------------------------------------------------
VEC2VEC_MODEL_IDS = {
    "e5":     "intfloat/e5-base-v2",
    "gra":    "ibm-granite/granite-embedding-278m-multilingual",
    "gte":    "thenlper/gte-base",
    "gtr":    "sentence-transformers/gtr-t5-base",
    "stella": "infgrad/stella-base-en-v2",
}

VARIANT_SUFFIX = {"casper": "unsupervised", "casper_tc": "linear"}

# (src_tag, tgt_tag, variant) — order matches the on-disk directory names.
# CASPER (unsupervised): all 10 unordered pairs of the 5 vec2vec encoders.
# CASPER-TC (linear): the 5 pairs for which a _linear checkpoint exists.
_PAIR_SPECS = [
    # --- CASPER (source-only) : all 10 pairs ---
    ("e5",  "stella", "casper"),
    ("gra", "e5",     "casper"),
    ("gra", "gte",    "casper"),
    ("gra", "stella", "casper"),
    ("gte", "e5",     "casper"),
    ("gte", "stella", "casper"),
    ("gtr", "e5",     "casper"),
    ("gtr", "gra",    "casper"),
    ("gtr", "gte",    "casper"),
    ("gtr", "stella", "casper"),
    # --- CASPER-TC (target-conditioned) : linear checkpoints available ---
    ("gra", "gte", "casper_tc"),
    ("gte", "e5",  "casper_tc"),
    ("gtr", "e5",  "casper_tc"),
    ("gtr", "gra", "casper_tc"),
    ("gtr", "gte", "casper_tc"),
]

PAIRS = [
    {
        "src": VEC2VEC_MODEL_IDS[s],
        "tgt": VEC2VEC_MODEL_IDS[t],
        "variant": v,
        "ckpt_dir": f"saved_mappers_official_{s}_{t}_1M_MacroStr_SimTlm_{VARIANT_SUFFIX[v]}",
    }
    for (s, t, v) in _PAIR_SPECS
]

# --- Datasets ---
TWEETTOPIC_DATASET = ("cardiffnlp/tweet_topic_multilingual", "en", "test")  # 800

# MIMIC (vec2vec, faithful): local HF dataset built by create_mimic.py
MIMIC_PATH          = "data/mimic"
MIMIC_SUBSET_SIZE   = 8192          # from vec2vec paper (rank spans 1..8192)
MIMIC_SHUFFLE_SEED  = 42
MIMIC_TRIM_TOKENS   = 64            # create_mimic.py trims notes to 64 ws-tokens

# MIMIC (HF surrogate): used ONLY if MIMIC_PATH is absent. Public clinical
# notes; NOT the credentialed Lehman set. Recorded as 'clinical_notes_hf'.
HF_CLINICAL_FALLBACK = {
    "name": "AGBonnet/augmented-clinical-notes",
    "split": "train",
    "text_field": "note",          # falls back to first string column if absent
}

INCLUDE_NAIVE_BASELINE = True       # F(x)=x, only when D_src == D_tgt

EMB_BATCH   = 64
OUT_DIR     = "eval_ood_vec2vec_protocol"
CACHE_DIR   = os.path.join(OUT_DIR, "emb_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# -------------------------
# Mapper architectures (verbatim from the training scripts)
# -------------------------
class LinearInjectionMapper(nn.Module):
    """CASPER-TC: h0 = input_proj(z_s) + memory_proj(z_t)."""
    def __init__(self, input_dim, output_dim, memory_dim,
                 hidden_dim=1024, n_layers=4, n_heads=16, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.input_proj  = nn.Linear(input_dim,  hidden_dim)
        self.memory_proj = nn.Linear(memory_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout),
            )
            for _ in range(n_layers)
        ])
        self.final_ln    = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.out_ln      = nn.LayerNorm(output_dim)

    def forward(self, x, memory):
        if x.dim() == 3:
            x = x.squeeze(1)
        if memory.dim() == 3:
            memory = memory.squeeze(1)
        h = self.input_proj(x) + self.memory_proj(memory)
        for block in self.blocks:
            h = h + block(h)
        h = self.final_ln(h)
        out = self.output_proj(h)
        out = self.out_ln(out)
        return out, h


class StandaloneMapper(nn.Module):
    """CASPER: source-only, pure f(z_s) -> z_hat_t. 'memory' is ignored."""
    def __init__(self, input_dim, output_dim, memory_dim=None,
                 hidden_dim=1024, n_layers=4, n_heads=16, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout),
            )
            for _ in range(n_layers)
        ])
        self.final_ln    = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.out_ln      = nn.LayerNorm(output_dim)

    def forward(self, x, memory=None):
        if x.dim() == 3:
            x = x.squeeze(1)
        del memory
        h = self.input_proj(x)
        for block in self.blocks:
            h = h + block(h)
        h = self.final_ln(h)
        out = self.output_proj(h)
        out = self.out_ln(out)
        return out, h

# -------------------------
# Checkpoint auto-detection & loading
# -------------------------
def build_mapper_from_state_dict(state: "OrderedDict[str, torch.Tensor]") -> nn.Module:
    hidden_dim = state["input_proj.weight"].shape[0]
    input_dim  = state["input_proj.weight"].shape[1]
    output_dim = state["output_proj.weight"].shape[0]
    n_layers   = len({k.split(".")[1] for k in state if k.startswith("blocks.")})

    if "memory_proj.weight" in state:
        memory_dim = state["memory_proj.weight"].shape[1]
        mapper = LinearInjectionMapper(input_dim, output_dim, memory_dim,
                                       hidden_dim=hidden_dim, n_layers=n_layers)
        arch = "LinearInjectionMapper"
    else:
        mapper = StandaloneMapper(input_dim, output_dim,
                                  hidden_dim=hidden_dim, n_layers=n_layers)
        arch = "StandaloneMapper"

    mapper.load_state_dict(state)
    print(f"    [+] {arch}: in={input_dim} out={output_dim} "
          f"hidden={hidden_dim} layers={n_layers}")
    return mapper, arch


def load_pair_mappers(ckpt_dir: str):
    src_ckpt = os.path.join(ckpt_dir, "official_mapper_src2tgt.pt")
    tgt_ckpt = os.path.join(ckpt_dir, "official_mapper_tgt2src.pt")
    if not (os.path.exists(src_ckpt) and os.path.exists(tgt_ckpt)):
        raise FileNotFoundError(f"Missing checkpoints in '{ckpt_dir}'")
    print(f"  [*] Loading mappers from '{ckpt_dir}'")
    m_s2t, arch_s2t = build_mapper_from_state_dict(torch.load(src_ckpt, map_location="cpu"))
    m_t2s, arch_t2s = build_mapper_from_state_dict(torch.load(tgt_ckpt, map_location="cpu"))
    return m_s2t.to(DEVICE).eval(), m_t2s.to(DEVICE).eval(), arch_s2t

# -------------------------
# Embedding backends (same as the training scripts)
# -------------------------
_ENCODER_CACHE = {}   # model_name -> (tokenizer, model)

def load_embedding_backend(name):
    if name.startswith("sentence-transformers/") and SentenceTransformer is not None:
        model = SentenceTransformer(name)
        tok = model.tokenizer
        return tok, model, "sentencetransformer"
    tok = AutoTokenizer.from_pretrained(name)
    try:
        model = AutoModel.from_pretrained(name, trust_remote_code=True)
    except Exception:
        model = AutoModelForSequenceClassification.from_pretrained(name, trust_remote_code=True)
    return tok, model, "hf"


def get_encoder(name):
    if name not in _ENCODER_CACHE:
        tok, mod, _ = load_embedding_backend(name)
        _ENCODER_CACHE[name] = (tok, mod)
    return _ENCODER_CACHE[name]


@torch.no_grad()
def embed_sentences(texts, tokenizer, model, device, batch_size=EMB_BATCH, max_length=512):
    embeddings = []
    is_st = SentenceTransformer is not None and isinstance(model, SentenceTransformer)
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i + batch_size]
        if is_st:
            emb = model.encode(batch, convert_to_tensor=True, device=device, show_progress_bar=False)
            embeddings.append(emb.cpu())
        else:
            model = model.to(device).eval()
            inputs = tokenizer(batch, return_tensors="pt", padding=True,
                               truncation=True, max_length=max_length).to(device)
            out = model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1)
            hidden = out.last_hidden_state * mask
            pooled = hidden.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            embeddings.append(pooled.cpu())
    return torch.cat(embeddings, dim=0).float()


def embed_with_cache(texts, dataset_key, model_name):
    safe = model_name.replace("/", "__")
    cache_path = os.path.join(CACHE_DIR, f"{dataset_key}__{safe}.pt")
    if os.path.exists(cache_path):
        emb = torch.load(cache_path, map_location="cpu").float()
        if emb.shape[0] == len(texts):
            print(f"    [+] Cache hit: {cache_path} {tuple(emb.shape)}")
            return emb
        print(f"    [!] Cache size mismatch for {cache_path}; re-encoding.")
    tok, mod = get_encoder(model_name)
    emb = embed_sentences(texts, tok, mod, DEVICE)
    torch.save(emb, cache_path)
    # free VRAM held by this encoder before the next one loads
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return emb

# -------------------------
# Datasets (vec2vec OOD protocol)
# -------------------------
def load_tweettopic_texts():
    name, config, split = TWEETTOPIC_DATASET
    ds = load_dataset(name, config)[split]
    texts = [t for t in ds["text"] if t and t.strip()]
    print(f"[+] TweetTopic: {len(texts)} texts (expected 800)")
    return texts


def _trim_ws_tokens(text, max_tokens=MIMIC_TRIM_TOKENS):
    # identical to create_mimic.py: split on whitespace, keep first N
    return " ".join(text.split()[:max_tokens])


def load_mimic_texts():
    """
    Returns (dataset_key, texts).
      * ('mimic', ...) when the credentialed local build is present;
      * ('clinical_notes_hf', ...) for the public HF surrogate;
      * (None, None) if neither is available.
    Both paths yield MIMIC_SUBSET_SIZE (8192) records, trimmed to 64 ws-tokens.
    """
    # 1) Faithful local reproduction
    if os.path.exists(MIMIC_PATH):
        dset = load_from_disk(MIMIC_PATH)
        if "unsupervised" in dset:
            dset = dset["unsupervised"]
        dset = dset.shuffle(seed=MIMIC_SHUFFLE_SEED)
        n = min(MIMIC_SUBSET_SIZE, len(dset))
        texts = [t for t in dset.select(range(n))["text"] if t and t.strip()]
        print(f"[+] MIMIC (local, faithful): {len(texts)} texts")
        return "mimic", texts

    # 2) HF public surrogate
    print(f"[!] MIMIC not found at '{MIMIC_PATH}'. "
          f"Falling back to HF surrogate '{HF_CLINICAL_FALLBACK['name']}' "
          "(public clinical notes, NOT the credentialed Lehman set).")
    try:
        ds = load_dataset(HF_CLINICAL_FALLBACK["name"], split=HF_CLINICAL_FALLBACK["split"])
    except Exception as e:
        print(f"[!] Could not load HF clinical fallback ({e}). Skipping MIMIC.")
        return None, None

    field = HF_CLINICAL_FALLBACK["text_field"]
    if field not in ds.column_names:
        str_cols = [c for c in ds.column_names
                    if isinstance(ds[0][c], str)]
        if not str_cols:
            print("[!] No string column in HF fallback. Skipping MIMIC.")
            return None, None
        field = str_cols[0]
        print(f"    [i] text_field not found; using '{field}'.")

    ds = ds.shuffle(seed=MIMIC_SHUFFLE_SEED)
    n = min(MIMIC_SUBSET_SIZE, len(ds))
    raw = ds.select(range(n))[field]
    texts = [_trim_ws_tokens(t) for t in raw if t and t.strip()]
    print(f"[+] clinical_notes_hf (surrogate): {len(texts)} texts "
          f"(target {MIMIC_SUBSET_SIZE}, trimmed to {MIMIC_TRIM_TOKENS} ws-tokens)")
    return "clinical_notes_hf", texts

# -------------------------
# Metrics (identical to evaluate_mapper_bidirectional / vec2vec Sec. 4.2)
# -------------------------
@torch.no_grad()
def compute_structure_preservation_spearman(original_embs, mapped_embs, sample_size=1000):
    original_embs = original_embs.float()
    mapped_embs = mapped_embs.float()
    N = original_embs.size(0)
    if N > sample_size:
        indices = torch.randperm(N)[:sample_size]
        original_embs = original_embs[indices]
        mapped_embs = mapped_embs[indices]
    original_norm = F.normalize(original_embs, dim=1)
    mapped_norm = F.normalize(mapped_embs, dim=1)
    original_sim = original_norm @ original_norm.T
    mapped_sim = mapped_norm @ mapped_norm.T
    triu = torch.triu_indices(original_sim.size(0), original_sim.size(1), offset=1)
    original_dists = (1 - original_sim[triu[0], triu[1]]).cpu().numpy()
    mapped_dists = (1 - mapped_sim[triu[0], triu[1]]).cpu().numpy()
    correlation, _ = spearmanr(original_dists, mapped_dists)
    return float(correlation)


@torch.no_grad()
def rank_metrics(mapped: torch.Tensor, target_all: torch.Tensor):
    mapped = F.normalize(mapped.float(), dim=-1).cpu()
    target_norm = F.normalize(target_all.float(), dim=-1).cpu()
    ranks, sims = [], []
    block_size = 1000
    for i in range(0, mapped.size(0), block_size):
        end = min(i + block_size, mapped.size(0))
        sim_matrix = mapped[i:end] @ target_norm.T
        for j in range(sim_matrix.size(0)):
            global_idx = i + j
            row = sim_matrix[j]
            rank = (torch.sort(row, descending=True).indices == global_idx)\
                       .nonzero(as_tuple=True)[0].item() + 1
            ranks.append(rank)
            sims.append(row[global_idx].item())
    return {
        "mean_cosine": float(np.mean(sims)),
        "top1_acc": sum(1 for r in ranks if r == 1) / len(ranks),
        "mean_rank": float(np.mean(ranks)),
        "n": len(ranks),
    }


@torch.no_grad()
def eval_mapper_on_pair(mapper, query, memory, target_all):
    is_cuda = (DEVICE.type == "cuda")
    if is_cuda:
        torch.cuda.synchronize(DEVICE)
    t0 = time.perf_counter()
    mapped, _ = mapper(query.to(DEVICE).float(), memory=memory.to(DEVICE).float())
    if is_cuda:
        torch.cuda.synchronize(DEVICE)
    forward_time = time.perf_counter() - t0
    res = rank_metrics(mapped, target_all)
    res["structure_preservation"] = compute_structure_preservation_spearman(query, mapped.cpu())
    res["forward_time_sec"] = forward_time
    return res

# -------------------------
# Main
# -------------------------
def main():
    # 1) Datasets (loaded once, shared across all pairs)
    datasets = {}
    datasets["tweettopic"] = load_tweettopic_texts()
    mimic_key, mimic_texts = load_mimic_texts()
    if mimic_texts is not None:
        datasets[mimic_key] = mimic_texts

    run_meta = {
        "protocol": "vec2vec OOD (Table 3), all pairs",
        "pairs": [{k: p[k] for k in ("src", "tgt", "variant", "ckpt_dir")} for p in PAIRS],
        "datasets": list(datasets.keys()),
        "tweettopic": TWEETTOPIC_DATASET,
        "mimic_path": MIMIC_PATH,
        "mimic_used_key": mimic_key,
        "mimic_subset_size": MIMIC_SUBSET_SIZE,
        "mimic_trim_tokens": MIMIC_TRIM_TOKENS,
        "hf_clinical_fallback": HF_CLINICAL_FALLBACK,
        "seed": SEED,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(OUT_DIR, "run_meta.json"), "w") as f:
        json.dump(run_meta, f, indent=2)

    rows = []
    naive_done = set()   # (dataset_key, frozenset({src,tgt})) -> avoid dup baselines

    # 2) Iterate over every model pair
    for pi, pair in enumerate(PAIRS):
        src, tgt, variant, ckpt_dir = pair["src"], pair["tgt"], pair["variant"], pair["ckpt_dir"]
        pair_tag = f"{src.split('/')[-1]}__{tgt.split('/')[-1]}"
        print(f"\n[*] ===== Pair {pi+1}/{len(PAIRS)}: {pair_tag} "
              f"(variant tag: {variant}) =====")

        # Load mappers first: if missing, skip the pair BEFORE any encoding
        try:
            m_s2t, m_t2s, arch = load_pair_mappers(ckpt_dir)
        except FileNotFoundError as e:
            print(f"  [!] {e} — skipping pair '{pair_tag}'.")
            continue

        # Encode both sides for every dataset (cached, so shared encoders are free)
        pair_embs = {}
        for key, texts in datasets.items():
            print(f"  [*] Encoding '{key}' | src={src}")
            pair_embs[(key, "src")] = embed_with_cache(texts, key, src)
            print(f"  [*] Encoding '{key}' | tgt={tgt}")
            pair_embs[(key, "tgt")] = embed_with_cache(texts, key, tgt)

        # Evaluate both directions on every dataset
        for key in datasets:
            s_emb, t_emb = pair_embs[(key, "src")], pair_embs[(key, "tgt")]
            print(f"  [*] {pair_tag} on {key} (N={s_emb.shape[0]}, "
                  f"D_src={s_emb.shape[1]}, D_tgt={t_emb.shape[1]})")

            for direction, mapper, query, memory, target in [
                ("src2tgt", m_s2t, s_emb, t_emb, t_emb),
                ("tgt2src", m_t2s, t_emb, s_emb, s_emb),
            ]:
                res = eval_mapper_on_pair(mapper, query, memory, target)
                rows.append({
                    "pair": pair_tag, "src": src, "tgt": tgt,
                    "variant": variant, "arch": arch,
                    "dataset": key, "direction": direction, **res,
                })
                print(f"      {direction}: cos={res['mean_cosine']:.3f} "
                      f"T-1={res['top1_acc']:.2f} rank={res['mean_rank']:.2f} "
                      f"SP={res['structure_preservation']:.3f}")

            # Naive baseline F(x)=x, once per (dataset, unordered pair), if dims match
            naive_key = (key, frozenset({src, tgt}))
            if (INCLUDE_NAIVE_BASELINE and naive_key not in naive_done
                    and s_emb.shape[1] == t_emb.shape[1]):
                for direction, query, target in [("src2tgt", s_emb, t_emb),
                                                 ("tgt2src", t_emb, s_emb)]:
                    res = rank_metrics(query, target)
                    res["structure_preservation"] = \
                        compute_structure_preservation_spearman(query, query)
                    res["forward_time_sec"] = 0.0
                    rows.append({
                        "pair": pair_tag, "src": src, "tgt": tgt,
                        "variant": "naive_baseline", "arch": "identity",
                        "dataset": key, "direction": direction, **res,
                    })
                naive_done.add(naive_key)

        # free mapper VRAM before the next pair
        del m_s2t, m_t2s, pair_embs
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    # 3) Persist
    if not rows:
        print("\n[!] No pairs evaluated (no checkpoints found). Nothing saved.")
        return

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "ood_metrics_vec2vec_protocol.csv")
    df.to_csv(csv_path, index=False)

    print("\n=== OOD RESULTS (vec2vec protocol, all pairs) ===")
    cols = ["pair", "variant", "dataset", "direction",
            "mean_cosine", "top1_acc", "mean_rank", "structure_preservation", "n"]
    print(df[cols].to_string(index=False))
    print(f"\n[+] Saved: {csv_path}")


if __name__ == "__main__":
    main()