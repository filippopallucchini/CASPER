#!/usr/bin/env python3
"""
DIAGNOSTIC: Run an already-trained CASPER (linear-injection) mapper
in unsupervised-at-inference mode and quantify the gap vs supervised.

You DO NOT retrain. You load the existing checkpoints
(official_mapper_src2tgt.pt, official_mapper_tgt2src.pt) and run the
same bidirectional evaluation under four different `memory` strategies:

    real_z_t   - actual paired target embedding (original supervised setup;
                 this is your published number)
    zeros      - torch.zeros — the honest 'no memory at inference' substitute
    mean_z_t   - mean of z_t over the eval set (a plausible 'we don't have
                 the per-sample target but we know the average target' fallback)
    noise      - Gaussian noise matched to z_t's per-dim mean/std (sanity floor:
                 if zeros and noise score similarly, memory is just being
                 used as a constant offset and the model didn't really need it)

For each (strategy, direction) it reports:
    mean_cosine, top1_acc, mean_rank, structure_preservation

How to read the output
----------------------
Look at the GAP between `real_z_t` and `zeros`. That gap is the price you
pay for going unsupervised-at-inference with the existing checkpoint:

  - SMALL GAP  (e.g. SP drops by <0.02, Top-1 drops by <2 pts):
      Your trained mapper barely used z_t. You can deploy the existing
      checkpoint as-is in the unsupervised setting. Just pass zeros (or
      mean_z_t, whichever scores better here) as memory at inference time.

  - LARGE GAP  (e.g. SP drops by >0.10, Top-1 drops by >10 pts):
      Your trained mapper relied heavily on z_t. The existing checkpoint
      is not safe for unsupervised deployment. You need to retrain with
      the standalone (no memory) variant.

  - zeros vs noise comparable:
      Confirms the model is treating memory as a constant — useful sanity check.

  - mean_z_t much better than zeros:
      The model is using z_t mostly as a "where in target space" hint;
      providing the mean recovers most of that. Cheap deployment fix.

CONFIG
------
Point CKPT_DIR at the run directory produced by
ALIGNMENT_OFFICIAL_diff_dim_linear.py and make sure SRC_NAME / TGT_NAME
match the encoders used in that run.
"""

import os
import json
import time
from typing import Dict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from datasets import load_dataset

import numpy as np
from scipy.stats import spearmanr
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# =============================================================================
# CONFIG  — point these at your existing CASPER run
# =============================================================================
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# Directory where the trained mapper was saved by the official script
CKPT_DIR = "saved_mappers_official_gtr_gte_1M_MacroStr_SimTlm_linear"
SRC_CKPT = os.path.join(CKPT_DIR, "official_mapper_src2tgt.pt")
TGT_CKPT = os.path.join(CKPT_DIR, "official_mapper_tgt2src.pt")

# Must match the encoders used in the training run
SRC_NAME = "sentence-transformers/gtr-t5-base"
TGT_NAME = "thenlper/gte-base"

# Mapper hyperparameters — must match the training run EXACTLY
MAPPER_HIDDEN = 1024
MAPPER_LAYERS = 4
MAPPER_HEADS  = 16   # unused, kept for ctor parity

# Evaluation set size — same default as the official script
AB_EVAL_SIZE = 65_536

# Output for this diagnostic
OUT_DIR = os.path.join(CKPT_DIR, "diagnostic_unsupervised_inference")
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# MAPPER CLASS — must be EXACTLY the same as the trained one for state_dict
# to load cleanly. Copied verbatim from ALIGNMENT_OFFICIAL_diff_dim_linear.py.
# =============================================================================
class LinearInjectionMapper(nn.Module):
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

# =============================================================================
# EMBEDDING BACKEND (copied from the official script)
# =============================================================================
def load_embedding_backend(name):
    if name.startswith("sentence-transformers/") and SentenceTransformer is not None:
        model = SentenceTransformer(name)
        tok = model.tokenizer
        return tok, model, "sentencetransformer"
    else:
        tok = AutoTokenizer.from_pretrained(name)
        try:
            model = AutoModel.from_pretrained(name, trust_remote_code=True)
        except Exception:
            model = AutoModelForSequenceClassification.from_pretrained(name, trust_remote_code=True)
        return tok, model, "hf"

@torch.no_grad()
def embed_sentences(texts, tokenizer, model, device, batch_size=64, max_length=512):
    embeddings = []
    is_st = isinstance(model, SentenceTransformer) if SentenceTransformer is not None else False
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]
        if is_st:
            emb = model.encode(batch, convert_to_tensor=True, device=device, show_progress_bar=False)
            embeddings.append(emb.cpu())
        else:
            model = model.to(device).eval()
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
            out = model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1)
            hidden = out.last_hidden_state * mask
            pooled = hidden.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            embeddings.append(pooled.cpu())
    return torch.cat(embeddings, dim=0)

# =============================================================================
# METRICS (copied from the official script)
# =============================================================================
@torch.no_grad()
def compute_structure_preservation_spearman(original_embs, mapped_embs, sample_size=1000):
    N = original_embs.size(0)
    if N > sample_size:
        indices = torch.randperm(N)[:sample_size]
        original_embs = original_embs[indices]
        mapped_embs = mapped_embs[indices]
    original_norm = F.normalize(original_embs, dim=1)
    mapped_norm   = F.normalize(mapped_embs, dim=1)
    original_sim = original_norm @ original_norm.T
    mapped_sim   = mapped_norm @ mapped_norm.T
    triu_indices = torch.triu_indices(original_sim.size(0), original_sim.size(1), offset=1)
    original_dists = (1 - original_sim[triu_indices[0], triu_indices[1]]).cpu().numpy()
    mapped_dists   = (1 - mapped_sim[triu_indices[0], triu_indices[1]]).cpu().numpy()
    correlation, _ = spearmanr(original_dists, mapped_dists)
    return float(correlation)

def compute_retrieval_metrics(mapped: torch.Tensor, target_all: torch.Tensor):
    """mean_cosine, top1_acc, mean_rank against the gallery of all targets."""
    mapped = F.normalize(mapped, dim=-1).cpu()
    target_norm = F.normalize(target_all, dim=-1).cpu()
    ranks, sims = [], []
    block_size = 1000
    for i in range(0, mapped.size(0), block_size):
        end = min(i + block_size, mapped.size(0))
        batch_mapped = mapped[i:end]
        sim_matrix = batch_mapped @ target_norm.T
        for j in range(sim_matrix.size(0)):
            global_idx = i + j
            row = sim_matrix[j]
            rank = (torch.sort(row, descending=True).indices == global_idx).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(rank)
            sims.append(row[global_idx].item())
    return {
        "mean_cosine": float(np.mean(sims)),
        "top1_acc":    sum(1 for r in ranks if r == 1) / len(ranks),
        "mean_rank":   float(np.mean(ranks)),
    }

# =============================================================================
# MEMORY SUBSTITUTION STRATEGIES
# =============================================================================
def build_memory(strategy: str, real_memory: torch.Tensor) -> torch.Tensor:
    """
    Returns a tensor of the same shape as `real_memory` to be passed as the
    `memory` argument of the trained mapper.

    real_memory: [N, D_memory] — the actual z_t (or z_s for the reverse direction)
                                 over the entire eval set.
    """
    if strategy == "real_z_t":
        return real_memory.clone()

    if strategy == "zeros":
        return torch.zeros_like(real_memory)

    if strategy == "mean_z_t":
        # Single mean vector broadcast to every row
        mean_vec = real_memory.mean(dim=0, keepdim=True)
        return mean_vec.expand_as(real_memory).clone()

    if strategy == "noise":
        # Gaussian matched to per-dimension mean/std of real memory
        mu = real_memory.mean(dim=0, keepdim=True)
        sd = real_memory.std(dim=0, keepdim=True).clamp(min=1e-6)
        return torch.randn_like(real_memory) * sd + mu

    raise ValueError(f"Unknown strategy: {strategy}")

# =============================================================================
# EVALUATION LOOP
# =============================================================================
@torch.no_grad()
def evaluate_under_strategy(
    mapper: nn.Module,
    query_emb: torch.Tensor,
    real_memory_emb: torch.Tensor,
    target_all: torch.Tensor,
    strategy: str,
    device,
    eval_batch: int = 2048,
) -> Dict:
    mapper.eval()
    memory_eval = build_memory(strategy, real_memory_emb)

    # Run mapper in batches to keep memory sane on large eval sets
    is_cuda = (device.type == 'cuda')
    if is_cuda:
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    mapped_chunks = []
    for i in range(0, query_emb.size(0), eval_batch):
        q = query_emb[i:i+eval_batch].to(device).float()
        m = memory_eval[i:i+eval_batch].to(device).float()
        out, _ = mapper(q, m)
        mapped_chunks.append(out.cpu())
    if is_cuda:
        torch.cuda.synchronize(device)
    forward_time = time.perf_counter() - t0

    mapped = torch.cat(mapped_chunks, dim=0)
    retrieval = compute_retrieval_metrics(mapped, target_all)
    sp = compute_structure_preservation_spearman(query_emb, mapped)

    return {
        **retrieval,
        "structure_preservation": sp,
        "forward_time_sec": forward_time,
        "n_samples": int(query_emb.size(0)),
    }

# =============================================================================
# MAIN
# =============================================================================
def main():
    # Sanity-check checkpoint paths up front
    for p in (SRC_CKPT, TGT_CKPT):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Checkpoint not found: {p}\n"
                f"Update CKPT_DIR at the top of this script to point at the "
                f"directory produced by your official CASPER training run."
            )

    # Encoders
    print(f"[*] Loading encoders ({SRC_NAME} | {TGT_NAME})")
    s_tok, s_mod, _ = load_embedding_backend(SRC_NAME)
    t_tok, t_mod, _ = load_embedding_backend(TGT_NAME)

    # Eval texts (same as the official script)
    print(f"[*] Loading eval set (NQ dev, up to {AB_EVAL_SIZE} samples)")
    eval_ds = load_dataset("jxm/nq_corpus_dpr", split="dev")
    eval_texts = eval_ds.select(range(min(AB_EVAL_SIZE, len(eval_ds))))["text"]

    # Pre-compute embeddings once — they are shared across all strategies
    print("[*] Embedding eval set with source encoder...")
    s_emb = embed_sentences(eval_texts, s_tok, s_mod, DEVICE).float()
    print("[*] Embedding eval set with target encoder...")
    t_emb = embed_sentences(eval_texts, t_tok, t_mod, DEVICE).float()

    D_src, D_tgt = s_emb.shape[1], t_emb.shape[1]
    print(f"[*] Embedding dims: D_src={D_src} | D_tgt={D_tgt}")

    # Build mappers and load checkpoints
    # src2tgt:  takes z_s, conditioned on z_t (memory_dim = D_tgt), outputs in target space
    # tgt2src:  takes z_t, conditioned on z_s (memory_dim = D_src), outputs in source space
    print("[*] Building mappers and loading checkpoints...")
    m_s2t = LinearInjectionMapper(
        input_dim=D_src, output_dim=D_tgt, memory_dim=D_tgt,
        hidden_dim=MAPPER_HIDDEN, n_layers=MAPPER_LAYERS, n_heads=MAPPER_HEADS,
    ).to(DEVICE)
    m_t2s = LinearInjectionMapper(
        input_dim=D_tgt, output_dim=D_src, memory_dim=D_src,
        hidden_dim=MAPPER_HIDDEN, n_layers=MAPPER_LAYERS, n_heads=MAPPER_HEADS,
    ).to(DEVICE)
    m_s2t.load_state_dict(torch.load(SRC_CKPT, map_location=DEVICE))
    m_t2s.load_state_dict(torch.load(TGT_CKPT, map_location=DEVICE))
    print("[+] Checkpoints loaded successfully.")

    # Run the diagnostic grid:
    #   2 directions x 4 strategies = 8 evaluations
    STRATEGIES = ["real_z_t", "zeros", "mean_z_t", "noise"]
    records = []
    for strategy in STRATEGIES:
        print(f"\n==== Strategy: {strategy} ====")
        # src2tgt: query=z_s, real memory=z_t, gallery=z_t
        print("  -> direction src2tgt")
        r_s2t = evaluate_under_strategy(m_s2t, s_emb, t_emb, t_emb, strategy, DEVICE)
        records.append({"direction": "src2tgt", "memory_strategy": strategy, **r_s2t})
        # tgt2src: query=z_t, real memory=z_s, gallery=z_s
        print("  -> direction tgt2src")
        r_t2s = evaluate_under_strategy(m_t2s, t_emb, s_emb, s_emb, strategy, DEVICE)
        records.append({"direction": "tgt2src", "memory_strategy": strategy, **r_t2s})

    # Save the raw grid
    df = pd.DataFrame(records)
    raw_csv = os.path.join(OUT_DIR, "diagnostic_metrics.csv")
    df.to_csv(raw_csv, index=False)

    # Save a pretty pivot table per metric — easier to read at a glance
    pretty_rows = []
    for metric in ["mean_cosine", "top1_acc", "mean_rank", "structure_preservation"]:
        for direction in ["src2tgt", "tgt2src"]:
            row = {"metric": metric, "direction": direction}
            for strategy in STRATEGIES:
                val = df[(df.direction == direction) & (df.memory_strategy == strategy)][metric].iloc[0]
                row[strategy] = val
            # Compute the gap real_z_t vs zeros — the headline "cost of going unsupervised"
            row["gap_real_minus_zeros"] = row["real_z_t"] - row["zeros"]
            pretty_rows.append(row)
    pretty_df = pd.DataFrame(pretty_rows)
    pretty_csv = os.path.join(OUT_DIR, "diagnostic_summary.csv")
    pretty_df.to_csv(pretty_csv, index=False)

    print("\n========== DIAGNOSTIC SUMMARY ==========")
    print(pretty_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nFiles written:")
    print(f"  - {raw_csv}")
    print(f"  - {pretty_csv}")
    print(
        "\nHOW TO READ:\n"
        "  Look at the gap_real_minus_zeros column for each metric.\n"
        "  Small gap  -> your checkpoint is safe to deploy unsupervised; pick\n"
        "                whichever of {zeros, mean_z_t} scored higher.\n"
        "  Large gap  -> the trained mapper relied on z_t; you need to retrain\n"
        "                with the standalone (no-memory) variant.\n"
        "  Mean_z_t vs zeros tells you whether the model is using z_t mostly\n"
        "  as a 'where in target space' anchor (mean recovers it) or as\n"
        "  per-sample information (mean does not recover it).\n"
    )

    # Persist run metadata
    with open(os.path.join(OUT_DIR, "diagnostic_meta.json"), "w") as f:
        json.dump({
            "ckpt_dir": CKPT_DIR,
            "src_model": SRC_NAME,
            "tgt_model": TGT_NAME,
            "ab_eval_size": AB_EVAL_SIZE,
            "strategies": STRATEGIES,
        }, f, indent=2)


if __name__ == "__main__":
    main()