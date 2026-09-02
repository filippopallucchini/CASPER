#!/usr/bin/env python3
"""
Official Alignment Training: GTR-Base <-> E5-Base (Asymmetric Dimensions Supported)
Configuration: MACRO_STR | SimTlm + StrMain (Weighted)
Dataset: 1 Million Samples (NQ) with Early Stopping

UNSUPERVISED-AT-INFERENCE VARIANT (no z_t memory at inference)
==============================================================
This variant is identical to the linear-injection CASPER setup EXCEPT that
the mapper no longer takes z_t (the target embedding) as a side-channel
input. The mapper is a pure function f(z_s) -> z_hat_t, so at inference
time only the source embedding is required.

----------------------------------------------------------------------------
FIXES IN THIS REVISION
----------------------------------------------------------------------------
1) DTYPE FIX (eval): some embedding backends (SentenceTransformer on GPU,
   certain HF models) return float16 embeddings. The mapper weights are
   float32, so the eval-time matmul raised:
       "mat1 and mat2 must have the same dtype, but got Half and Float".
   evaluate_mapper_bidirectional now casts all embeddings and mapper
   inputs to float32. (Training was already safe because EmbeddingPairsDataset
   stores .float() tensors.)

2) SKIP-IF-EXISTS: if both mapper checkpoints already exist in OUT_DIR and
   SKIP_TRAINING_IF_EXISTS is True, training is skipped entirely; the script
   only probes the embedding dimensions, rebuilds the architecture, loads the
   saved weights, and runs evaluation. This lets you recover an interrupted
   run (e.g. one that trained successfully but crashed at eval) without
   re-training, and re-run evaluation cheaply.
"""

import os
import random
import sqlite3 
import json
import copy
import time                # NEW: for mapper timing
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from datasets import load_dataset

import numpy as np
from scipy.stats import spearmanr

# plotting & analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# optional SentenceTransformer backend
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# -------------------------
# Config
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# Selected Models (Asymmetric Pair)
SRC_NAME = "intfloat/e5-base-v2" #"Qwen/Qwen3-Embedding-0.6B"#"thenlper/gte-base"#"ibm-granite/granite-embedding-278m-multilingual"#"sentence-transformers/gtr-t5-base"#"dmis-lab/biobert-v1.1"#"nlpaueb/legal-bert-base-uncased"#"dmis-lab/biobert-base-cased-v1.1"#"infgrad/stella-base-en-v2" #"thenlper/gte-large"##"ProsusAI/finbert"## ## 
TGT_NAME = "infgrad/stella-base-en-v2"#"intfloat/e5-base-v2"#"thenlper/gte-base"#"ibm-granite/granite-embedding-278m-multilingual"#"sentence-transformers/gtr-t5-base"#"ibm-granite/granite-embedding-278m-multilingual"# # # # ## # 

# Data Params
SUBSAMPLE = 1_000_000  # Official Run Size
CHUNK_TOKENS = 64

# Training Params
TRAIN_BATCH = 256
LR = 1e-4
MAX_EPOCHS = 100       # High cap, controlled by Early Stopping

# Evaluation Params
AB_EVAL_SIZE = 65_536
MAPPER_LAYERS = 4
MAPPER_HEADS = 16      # kept for parity with the original config (unused in standalone variant)

# Early Stopping Config
PATIENCE = 3           
MIN_DELTA = 1e-5

# Output Setup
run_tag = "official_e5_stella_1M_MacroStr_SimTlm_unsupervised"
OUT_DIR = f"saved_mappers_{run_tag}"
os.makedirs(OUT_DIR, exist_ok=True)
RESERVOIR_DB = "nq_reservoir.db"

# If both mapper checkpoints already exist in OUT_DIR, skip training entirely
# and go straight to evaluation. Set to False to force a fresh training run
# (existing checkpoints would then be overwritten after training).
SKIP_TRAINING_IF_EXISTS = True

# -------------------------
# 1. Early Stopping Logic
# -------------------------
class EarlyStopping:
    def __init__(self, patience=PATIENCE, min_delta=MIN_DELTA, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_state_src2tgt = None
        self.best_state_tgt2src = None

    def __call__(self, current_loss, m_src2tgt, m_tgt2src):
        if self.best_loss is None:
            self.best_loss = current_loss
            self.save_checkpoint(m_src2tgt, m_tgt2src)
        elif current_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"      [EarlyStopping] No improvement. Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = current_loss
            self.save_checkpoint(m_src2tgt, m_tgt2src)
            self.counter = 0

    def save_checkpoint(self, m_src2tgt, m_tgt2src):
        if self.verbose:
            print(f"      [EarlyStopping] Loss improved to {self.best_loss:.6f}. Saving best state in memory...")
        self.best_state_src2tgt = copy.deepcopy(m_src2tgt.state_dict())
        self.best_state_tgt2src = copy.deepcopy(m_tgt2src.state_dict())

# -------------------------
# 2. Models & Loader
# -------------------------
def load_embedding_backend(name):
    if name.startswith("sentence-transformers/") and SentenceTransformer is not None:
        model = SentenceTransformer(name)
        tok = model.tokenizer 
        return tok, model, "sentencetransformer"
    else:
        tok = AutoTokenizer.from_pretrained(name)
        try:
            model = AutoModel.from_pretrained(name, trust_remote_code=True)
        except:
            model = AutoModelForSequenceClassification.from_pretrained(name, trust_remote_code=True)
        return tok, model, "hf"

@torch.no_grad()
def embed_sentences(texts, tokenizer, model, device, batch_size=64, max_length=512):
    embeddings = []
    is_st = isinstance(model, SentenceTransformer)
    
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

class EmbeddingPairsDataset(Dataset):
    def __init__(self, src_emb, tgt_emb):
        self.src = src_emb.float()
        self.tgt = tgt_emb.float()
    def __len__(self): return self.src.shape[0]
    def __getitem__(self, idx): return self.src[idx], self.tgt[idx]

# -------------------------------------------------------------------
# STANDALONE MAPPER  (unsupervised-at-inference)
# -------------------------------------------------------------------
# Pure function f(z_s) -> z_hat_t. The 'memory' branch is REMOVED, so the
# mapper has no access to z_t at any point. forward(x, memory=None) keeps
# the same signature as the linear-injection version: any 'memory' passed
# in is silently ignored.
# -------------------------------------------------------------------
class StandaloneMapper(nn.Module):
    def __init__(self, input_dim, output_dim, memory_dim=None,
                 hidden_dim=1024, n_layers=4, n_heads=16, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Stage 1: project source into hidden space. NOTE: no memory_proj.
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Stage 2: stack of residual FFN blocks
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

        # Stage 3: final projection and normalisation
        self.final_ln    = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.out_ln      = nn.LayerNorm(output_dim)

    def forward(self, x, memory=None):
        """
        Args:
            x:      [B, input_dim]  - source query embeddings
            memory: ignored (signature parity with the linear-injection variant).
        Returns:
            out: [B, output_dim]
            h:   internal hidden state, kept for return-tuple parity.
        """
        if x.dim() == 3:
            x = x.squeeze(1)
        del memory  # no-op; documents that the argument is unused

        h = self.input_proj(x)   # [B, hidden_dim]
        for block in self.blocks:
            h = h + block(h)

        h = self.final_ln(h)
        out = self.output_proj(h)
        out = self.out_ln(out)
        return out, h

# Keep the original class name as an alias so the rest of the code below
# (which references TransformerEmbeddingMapper) still works unchanged.
TransformerEmbeddingMapper = StandaloneMapper

# -------------------------
# 3. Weighted Loss Components
# -------------------------
def pairwise_dist(x):
    x = F.normalize(x, dim=1)
    return 1 - x @ x.T

def nt_xent_loss(z_i, z_j, temperature=0.05):
    z_i, z_j = F.normalize(z_i, dim=1), F.normalize(z_j, dim=1)
    reps = torch.cat([z_i, z_j], dim=0)
    sim = reps @ reps.T / temperature
    B = z_i.size(0)
    labels = torch.arange(B, device=z_i.device)
    labels = torch.cat([labels + B, labels], dim=0)
    mask = torch.eye(2 * B, device=z_i.device).bool()
    sim = sim.masked_fill(mask, -9e15)
    return F.cross_entropy(sim, labels)

def structure_preservation_loss_pearson(orig, mapped, sample_size=200, eps=1e-8):
    N = orig.size(0)
    if N > sample_size:
        idx = torch.randperm(N, device=orig.device)[:sample_size]
        orig, mapped = orig[idx], mapped[idx]
    orig_n, mapped_n = F.normalize(orig.float(), dim=1), F.normalize(mapped.float(), dim=1)
    orig_sim, map_sim = orig_n @ orig_n.T, mapped_n @ mapped_n.T
    triu = torch.triu_indices(orig_sim.size(0), orig_sim.size(1), 1)
    o_d = (1 - orig_sim[triu[0], triu[1]]).view(-1)
    m_d = (1 - map_sim[triu[0], triu[1]]).view(-1)
    ox, mx = o_d - o_d.mean(), m_d - m_d.mean()
    cov = (ox * mx).mean()
    corr = cov / ((o_d.std(unbiased=False)+eps)*(m_d.std(unbiased=False)+eps))
    return 1.0 - torch.clamp(corr, -1, 1)

def global_similarity_loss(mapped, target):
    return F.l1_loss(pairwise_dist(mapped), pairwise_dist(target))

def local_similarity_loss(mapped, target, k=200):
    dist_target, dist_mapped = pairwise_dist(target), pairwise_dist(mapped)
    k = min(k, dist_target.size(1)-1)
    knn_idx = dist_target.topk(k+1, largest=False).indices[:,1:]
    losses = [F.l1_loss(dist_mapped[i, knn_idx[i]], dist_target[i, knn_idx[i]]) for i in range(target.size(0))]
    return torch.stack(losses).mean()

def flexible_loss_weighted(model_src_emb, model_tgt_emb,
                           mapper_src2tgt, mapper_tgt2src,
                           weights: Dict[str, float],
                           k_local=10, temperature=0.05, tlm_mask_prob=0.15,
                           device='cpu'):
    
    model_src_emb = model_src_emb.to(device)
    model_tgt_emb = model_tgt_emb.to(device)
    
    total_loss = 0.0
    logs = {}

    # --- 1) Standard Forward Pass ---
    mapped_src2tgt, _ = mapper_src2tgt(model_src_emb, memory=model_tgt_emb)
    mapped_tgt2src, _ = mapper_tgt2src(model_tgt_emb, memory=model_src_emb)
    
    # --- 2) MSE (Weighted) ---
    w_mse = weights.get('mse', 0.0)
    if w_mse > 0:
        loss_mse = F.mse_loss(mapped_src2tgt, model_tgt_emb) + F.mse_loss(mapped_tgt2src, model_src_emb)
        total_loss = total_loss + w_mse * loss_mse
        logs['mse'] = loss_mse.item()
    else:
        logs['mse'] = 0.0

    # --- 3) Cycle Consistency (Weighted) ---
    w_cycle = weights.get('cycle', 0.0)
    if w_cycle > 0:
        src_cycle, _ = mapper_tgt2src(mapped_src2tgt.detach(), memory=model_src_emb)
        tgt_cycle, _ = mapper_src2tgt(mapped_tgt2src.detach(), memory=model_tgt_emb)
        loss_cycle = F.mse_loss(src_cycle, model_src_emb) + F.mse_loss(tgt_cycle, model_tgt_emb)
        total_loss = total_loss + w_cycle * loss_cycle
        logs['cycle'] = loss_cycle.item()
    else:
        logs['cycle'] = 0.0

    # --- 4) Contrastive NT-Xent (Weighted) ---
    w_con = weights.get('con', 0.0)
    if w_con > 0:
        loss_con = nt_xent_loss(mapped_src2tgt, model_tgt_emb, temperature) + \
                   nt_xent_loss(mapped_tgt2src, model_src_emb, temperature)
        total_loss = total_loss + w_con * loss_con
        logs['contrastive'] = loss_con.item()
    else:
        logs['contrastive'] = 0.0

    # --- 5) Global similarity (Weighted) ---
    w_global = weights.get('global', 0.0)
    if w_global > 0:
        loss_global_s = global_similarity_loss(mapped_src2tgt, model_src_emb)
        loss_global_t = global_similarity_loss(mapped_tgt2src, model_tgt_emb)
        loss_global = loss_global_s + loss_global_t
        total_loss = total_loss + w_global * loss_global
        logs['global'] = loss_global.item()
    else:
        logs['global'] = 0.0

    # --- 6) Local similarity (Weighted) ---
    w_local = weights.get('local', 0.0)
    if w_local > 0:
        loss_local_s = local_similarity_loss(mapped_src2tgt, model_src_emb, k=k_local)
        loss_local_t = local_similarity_loss(mapped_tgt2src, model_tgt_emb, k=k_local)
        loss_local = loss_local_s + loss_local_t
        total_loss = total_loss + w_local * loss_local
        logs['local'] = loss_local.item()
    else:
        logs['local'] = 0.0

    # --- 7) TLM (Weighted) - asymmetric-dim aware ---
    w_tlm = weights.get('tlm', 0.0)
    if w_tlm > 0:
        mask_src = (torch.rand_like(model_src_emb) < tlm_mask_prob).float()
        mask_tgt = (torch.rand_like(model_tgt_emb) < tlm_mask_prob).float()
        
        noisy_src = model_src_emb * mask_src
        noisy_tgt = model_tgt_emb * mask_tgt
        
        recon_tgt, _ = mapper_src2tgt(noisy_src, memory=noisy_tgt)
        recon_src, _ = mapper_tgt2src(noisy_tgt, memory=noisy_src)
        
        loss_tlm_tgt = F.mse_loss(recon_tgt * mask_tgt, model_tgt_emb * mask_tgt)
        loss_tlm_src = F.mse_loss(recon_src * mask_src, model_src_emb * mask_src)
        loss_tlm = loss_tlm_tgt + loss_tlm_src
            
        total_loss = total_loss + w_tlm * loss_tlm
        logs['tlm'] = loss_tlm.item()
    else:
        logs['tlm'] = 0.0

    # --- 8) Structure Preservation Loss (Weighted) ---
    w_struct = weights.get('structure', 0.0)
    if w_struct > 0:
        loss_struct_s = structure_preservation_loss_pearson(model_src_emb, mapped_src2tgt)
        loss_struct_t = structure_preservation_loss_pearson(model_tgt_emb, mapped_tgt2src)
        loss_struct = loss_struct_s + loss_struct_t
        
        total_loss = total_loss + w_struct * loss_struct
        logs['structure'] = loss_struct.item()
    else:
        logs['structure'] = 0.0

    logs['total'] = total_loss.item()
    return total_loss, logs

# -------------------------
# 4. Training Loop  (with mapper timing)
# -------------------------
def train_mapper_official(m_s2t, m_t2s, loader, optimizer, device, weights, max_epochs=MAX_EPOCHS):
    m_s2t.to(device).train()
    m_t2s.to(device).train()
    stopper = EarlyStopping(patience=PATIENCE)

    print(f"[*] Starting Training (STANDALONE / unsupervised-at-inference) with Weights: {json.dumps(weights, indent=2)}")

    timing_records = []
    is_cuda = (device.type == 'cuda')

    for epoch in range(1, max_epochs + 1):
        epoch_loss, n_batches = 0.0, 0
        epoch_time_total = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{max_epochs}")
        
        for batch_idx, (src, tgt) in enumerate(pbar):
            if is_cuda:
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()

            optimizer.zero_grad()
            loss, logs = flexible_loss_weighted(src, tgt, m_s2t, m_t2s, weights, device=device)
            loss.backward()
            optimizer.step()

            if is_cuda:
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - t0

            timing_records.append({
                'epoch': epoch,
                'batch': batch_idx,
                'batch_size': src.size(0),
                'time_sec': elapsed,
                'loss': loss.item(),
            })

            epoch_loss += loss.item()
            epoch_time_total += elapsed
            n_batches += 1
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'ms/batch': f"{elapsed*1000:.1f}"})
        
        avg_loss = epoch_loss / n_batches if n_batches > 0 else 0
        avg_time_ms = (epoch_time_total / n_batches * 1000) if n_batches > 0 else 0
        print(f"  Epoch {epoch} Avg Loss: {avg_loss:.6f} | Avg mapper step: {avg_time_ms:.2f} ms/batch")
        
        stopper(avg_loss, m_s2t, m_t2s)
        if stopper.early_stop:
            print(f"  [!] Early stopping triggered at epoch {epoch}. Restoring best model.")
            m_s2t.load_state_dict(stopper.best_state_src2tgt)
            m_t2s.load_state_dict(stopper.best_state_tgt2src)
            break
    
    timing_df = pd.DataFrame(timing_records)
    timing_csv_path = os.path.join(OUT_DIR, "mapper_timing_train.csv")
    timing_df.to_csv(timing_csv_path, index=False)
    
    if not timing_df.empty:
        summary = (
            timing_df.groupby('epoch')
            .agg(n_batches=('batch', 'count'),
                 mean_time_sec=('time_sec', 'mean'),
                 std_time_sec=('time_sec', 'std'),
                 total_time_sec=('time_sec', 'sum'),
                 mean_loss=('loss', 'mean'))
            .reset_index()
        )
        summary['mean_time_ms_per_batch'] = summary['mean_time_sec'] * 1000.0
        summary_csv_path = os.path.join(OUT_DIR, "mapper_timing_train_summary.csv")
        summary.to_csv(summary_csv_path, index=False)
        print(f"[+] Mapper training timing saved: {timing_csv_path}")
        print(f"[+] Mapper training timing summary saved: {summary_csv_path}")

    return m_s2t, m_t2s

# -------------------------
# 5. Helpers (Evaluation & Data)
# -------------------------
@torch.no_grad()
def compute_structure_preservation_spearman(original_embs, mapped_embs, sample_size=1000):
    # Cast to float32 for numerical stability and to avoid half/float clashes
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
    
    triu_indices = torch.triu_indices(original_sim.size(0), original_sim.size(1), offset=1)
    original_dists = (1 - original_sim[triu_indices[0], triu_indices[1]]).cpu().numpy()
    mapped_dists = (1 - mapped_sim[triu_indices[0], triu_indices[1]]).cpu().numpy()
    correlation, _ = spearmanr(original_dists, mapped_dists)
    return correlation

def evaluate_mapper_bidirectional(m_s2t, m_t2s, s_mod, s_tok, t_mod, t_tok, texts, device):
    """
    Evaluates both directions and records mapper inference timing.

    DTYPE NOTE: embeddings are cast to float32 immediately after encoding.
    Some backends (e.g. SentenceTransformer on GPU) return float16, which
    would otherwise clash with the float32 mapper weights at matmul time
    ("mat1 and mat2 must have the same dtype, but got Half and Float").
    """
    m_s2t.eval(); m_t2s.eval()
    # FIX: force float32 on the encoded embeddings
    s_emb = embed_sentences(texts, s_tok, s_mod, device).float()
    t_emb = embed_sentences(texts, t_tok, t_mod, device).float()
    
    res = {}
    eval_timing_records = []
    is_cuda = (device.type == 'cuda')

    for d, mapper, query, memory, target_all in [('src2tgt', m_s2t, s_emb, t_emb, t_emb), 
                                                 ('tgt2src', m_t2s, t_emb, s_emb, s_emb)]:
        if is_cuda:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        with torch.no_grad():
            # FIX: cast to float32 at the mapper boundary as well (belt-and-braces)
            mapped, _ = mapper(query.to(device).float(), memory=memory.to(device).float())

        if is_cuda:
            torch.cuda.synchronize(device)
        forward_time = time.perf_counter() - t0

        eval_timing_records.append({
            'direction': d,
            'n_samples': query.size(0),
            'forward_time_sec': forward_time,
            'samples_per_sec': query.size(0) / forward_time if forward_time > 0 else float('nan'),
            'ms_per_sample': (forward_time / query.size(0) * 1000.0) if query.size(0) > 0 else float('nan'),
        })

        mapped = F.normalize(mapped, dim=-1).cpu().float()
        target_norm = F.normalize(target_all, dim=-1).float()
        
        ranks, sims = [], []
        block_size = 1000
        for i in range(0, mapped.size(0), block_size):
            end = min(i + block_size, mapped.size(0))
            batch_mapped = mapped[i:end]
            # FIX: ensure both operands are float32 for the similarity matmul
            sim_matrix = batch_mapped.float() @ target_norm.T.float()
            
            for j in range(sim_matrix.size(0)):
                global_idx = i + j
                row = sim_matrix[j]
                rank = (torch.sort(row, descending=True).indices == global_idx).nonzero(as_tuple=True)[0].item() + 1
                ranks.append(rank)
                sims.append(row[global_idx].item())
            
        res[d] = {
            "mean_cosine": float(np.mean(sims)),
            "top1_acc": sum(1 for r in ranks if r == 1) / len(ranks),
            "mean_rank": float(np.mean(ranks)),
            "structure_preservation": compute_structure_preservation_spearman(query, mapped)
        }

    eval_timing_df = pd.DataFrame(eval_timing_records)
    eval_timing_csv = os.path.join(OUT_DIR, "mapper_timing_eval.csv")
    eval_timing_df.to_csv(eval_timing_csv, index=False)
    print(f"[+] Mapper evaluation timing saved: {eval_timing_csv}")

    return res

def sentence_split_into_64_token_chunks(text, tokenizer, chunk_tokens=CHUNK_TOKENS):
    toks = tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]
    chunks = []
    for i in range(0, len(toks), chunk_tokens):
        window = toks[i:i + chunk_tokens]
        if len(window) == chunk_tokens:
            chunks.append(tokenizer.decode(window, skip_special_tokens=True))
    return chunks

def build_reservoir_sqlite(db_path, sample_size, tokenizer):
    print(f"[*] Building new reservoir DB with {sample_size} samples...")
    ds = load_dataset("jxm/nq_corpus_dpr", split="train", streaming=True)
    conn = sqlite3.connect(db_path); cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS sentences")
    cur.execute("CREATE TABLE sentences (id INTEGER PRIMARY KEY, text TEXT)")
    
    inserted = 0
    for ex in tqdm(ds, desc="Streaming & Chunking"):
        if inserted >= sample_size: break
        txt = ex.get("text", "") or ex.get("document_text", "")
        if not txt: continue
        
        chunks = sentence_split_into_64_token_chunks(txt.replace("\n", " "), tokenizer)
        for ch in chunks:
            if inserted < sample_size:
                cur.execute("INSERT INTO sentences (id, text) VALUES (?, ?)", (inserted+1, ch))
                inserted += 1
            else:
                break
        if inserted % 10000 == 0: conn.commit()
    conn.commit(); conn.close()

def load_texts_for_training(db_path, subsample_size, tokenizer):
    if not os.path.exists(db_path):
        build_reservoir_sqlite(db_path, subsample_size, tokenizer)
    
    print(f"[*] Loading {subsample_size} texts from DB...")
    conn = sqlite3.connect(db_path); cur = conn.cursor()
    cur.execute("SELECT text FROM sentences LIMIT ?", (subsample_size,))
    texts = [r[0] for r in cur.fetchall()]
    conn.close()
    
    if len(texts) < subsample_size:
        print(f"Warning: Requested {subsample_size} but DB only had {len(texts)}. Using all.")
    return texts

# -------------------------
# Main Execution
# -------------------------
def main():
    run_meta = {
        'run_tag': run_tag,
        'variant': 'standalone_unsupervised_at_inference',
        'mapper_reads_memory_at_inference': False,
        'src_model': SRC_NAME,
        'tgt_model': TGT_NAME,
        'subsample': SUBSAMPLE,
        'train_batch': TRAIN_BATCH,
        'lr': LR,
        'max_epochs': MAX_EPOCHS,
        'mapper_layers': MAPPER_LAYERS,
        'mapper_heads_unused_in_standalone': MAPPER_HEADS,
        'patience': PATIENCE,
        'min_delta': MIN_DELTA,
        'seed': SEED,
        'started_at': datetime.now().isoformat(timespec='seconds'),
    }
    with open(os.path.join(OUT_DIR, "run_meta.json"), 'w') as f:
        json.dump(run_meta, f, indent=2)

    s_tok, s_mod, _ = load_embedding_backend(SRC_NAME)
    t_tok, t_mod, _ = load_embedding_backend(TGT_NAME)
    chunk_tok = s_tok if s_tok else t_tok

    # --- Decide whether to train or just load existing checkpoints ---
    src_ckpt = os.path.join(OUT_DIR, "official_mapper_src2tgt.pt")
    tgt_ckpt = os.path.join(OUT_DIR, "official_mapper_tgt2src.pt")
    checkpoints_exist = os.path.exists(src_ckpt) and os.path.exists(tgt_ckpt)
    do_train = not (checkpoints_exist and SKIP_TRAINING_IF_EXISTS)

    if checkpoints_exist and SKIP_TRAINING_IF_EXISTS:
        print(f"[*] Found existing checkpoints in '{OUT_DIR}'.")
        print("[*] SKIP_TRAINING_IF_EXISTS=True -> skipping training, evaluation only.")
    elif checkpoints_exist and not SKIP_TRAINING_IF_EXISTS:
        print(f"[!] Checkpoints exist in '{OUT_DIR}' but SKIP_TRAINING_IF_EXISTS=False "
              "-> they will be RETRAINED and overwritten.")

    # --- Dimension detection ---
    # If training, dims come for free from the precomputed 1M embeddings.
    # If skipping, probe a single sentence to avoid re-embedding the corpus.
    ab_loader = None
    if do_train:
        texts_train = load_texts_for_training(RESERVOIR_DB, SUBSAMPLE, chunk_tok)
        print(f"[+] Loaded {len(texts_train)} training sequences.")

        print("[*] Pre-computing source embeddings...")
        s_emb = embed_sentences(texts_train, s_tok, s_mod, DEVICE).float()
        print("[*] Pre-computing target embeddings...")
        t_emb = embed_sentences(texts_train, t_tok, t_mod, DEVICE).float()

        ab_loader = DataLoader(EmbeddingPairsDataset(s_emb, t_emb), batch_size=TRAIN_BATCH, shuffle=True)

        D_src = s_emb.shape[1]
        D_tgt = t_emb.shape[1]
    else:
        print("[*] Probing embedding dimensions on a tiny sample (no 1M re-embed)...")
        probe_s = embed_sentences(["probe sentence for dim detection"], s_tok, s_mod, DEVICE).float()
        probe_t = embed_sentences(["probe sentence for dim detection"], t_tok, t_mod, DEVICE).float()
        D_src = probe_s.shape[1]
        D_tgt = probe_t.shape[1]

    print(f"[*] Detected Dimensions: Source={D_src}, Target={D_tgt}")

    # OFFICIAL Weights (used only if training)
    official_weights = {
        'structure': 0.75 * 0.50,  # 0.375
        'global':    0.75 * 0.25,  # 0.1875
        'local':     0.75 * 0.25,  # 0.1875
        'tlm':       0.25 * 0.75,  # 0.1875
        'con':       0.25 * 0.25,  # 0.0625
        'mse': 0.0,
        'cycle': 0.0
    }

    # Initialize Mappers (STANDALONE / unsupervised-at-inference).
    # IMPORTANT: hidden_dim MUST match the saved checkpoint exactly, or
    # load_state_dict will fail. The training run used HIDDEN_DIM = max(D_src, D_tgt),
    # so reproduce it here for BOTH the train and the load paths.
    print("[*] Initializing Standalone Mappers (no z_t memory at inference)...")
    HIDDEN_DIM = max(D_src, D_tgt)

    m_s2t = TransformerEmbeddingMapper(
        input_dim=D_src, output_dim=D_tgt, memory_dim=D_tgt, 
        hidden_dim=HIDDEN_DIM, n_layers=MAPPER_LAYERS, n_heads=MAPPER_HEADS
    ).to(DEVICE)
    
    m_t2s = TransformerEmbeddingMapper(
        input_dim=D_tgt, output_dim=D_src, memory_dim=D_src, 
        hidden_dim=HIDDEN_DIM, n_layers=MAPPER_LAYERS, n_heads=MAPPER_HEADS
    ).to(DEVICE)

    n_params_s2t = sum(p.numel() for p in m_s2t.parameters())
    n_params_t2s = sum(p.numel() for p in m_t2s.parameters())
    print(f"[*] Mapper params: src2tgt={n_params_s2t:,} | tgt2src={n_params_t2s:,}")
    with open(os.path.join(OUT_DIR, "param_counts.json"), 'w') as f:
        json.dump({'src2tgt': n_params_s2t, 'tgt2src': n_params_t2s,
                   'total': n_params_s2t + n_params_t2s}, f, indent=2)

    # Train with Early Stopping (and timing) — OR load existing checkpoints
    if do_train:
        opt = torch.optim.AdamW(list(m_s2t.parameters()) + list(m_t2s.parameters()), lr=LR)
        train_start = time.perf_counter()
        m_s2t, m_t2s = train_mapper_official(m_s2t, m_t2s, ab_loader, opt, DEVICE,
                                             official_weights, MAX_EPOCHS)
        train_total_sec = time.perf_counter() - train_start
        print(f"[+] Total training wall time: {train_total_sec:.1f} s "
              f"({train_total_sec/60.0:.1f} min)")

        print(f"[+] Saving final models to {OUT_DIR}...")
        torch.save(m_s2t.state_dict(), src_ckpt)
        torch.save(m_t2s.state_dict(), tgt_ckpt)
    else:
        print(f"[*] Loading existing mappers from '{OUT_DIR}'...")
        m_s2t.load_state_dict(torch.load(src_ckpt, map_location=DEVICE))
        m_t2s.load_state_dict(torch.load(tgt_ckpt, map_location=DEVICE))
        print("[+] Mappers loaded successfully — proceeding directly to evaluation.")
        train_total_sec = 0.0

    # Final Evaluation
    print("[*] Loading evaluation data...")
    eval_ds = load_dataset("jxm/nq_corpus_dpr", split="dev")
    eval_texts = eval_ds.select(range(min(AB_EVAL_SIZE, len(eval_ds))))["text"]
    
    print("[*] Running final evaluation...")
    metrics = evaluate_mapper_bidirectional(m_s2t, m_t2s, s_mod, s_tok, t_mod, t_tok, eval_texts, DEVICE)
    
    res_df = pd.DataFrame([
        {'direction': 'src2tgt', **metrics['src2tgt']},
        {'direction': 'tgt2src', **metrics['tgt2src']}
    ])
    res_df.to_csv(os.path.join(OUT_DIR, "official_metrics.csv"), index=False)
    
    with open(os.path.join(OUT_DIR, "wall_clock.json"), 'w') as f:
        json.dump({
            'train_total_seconds': train_total_sec,
            'train_total_minutes': train_total_sec / 60.0,
            'skipped_training': (not do_train),
            'finished_at': datetime.now().isoformat(timespec='seconds'),
        }, f, indent=2)

    print("\n=== OFFICIAL RESULTS (UNSUPERVISED-AT-INFERENCE) ===")
    print(res_df.to_string())
    print(f"[+] Done. Artifacts saved in {OUT_DIR}")

if __name__ == "__main__":
    main()