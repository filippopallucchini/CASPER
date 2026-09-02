#!/usr/bin/env python3
"""
Alignment training on NQ with Cross-Attention Transformer Mapper.
Updated for Hierarchical Weight Grid Search (25 Experiments).

Workflow:
 - Load/Build Reservoir DB.
 - Embed sampled sentences.
 - Run Grid Search: 24 Hierarchical configs + 1 Baseline.
 - Components: con, tlm (Similarity Group) vs structure, global, local (Structure Group).
"""

import os
import random
import sqlite3 
import json
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
from datetime import datetime
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
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
# Config (edit as needed)
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# embedding model names
SRC_NAME = "sentence-transformers/gtr-t5-base"#"Qwen/Qwen3-Embedding-0.6B"#"ibm-granite/granite-embedding-278m-multilingual"# #"intfloat/e5-base-v2" #"infgrad/stella-base-en-v2" #"thenlper/gte-base" ## 
TGT_NAME = "intfloat/e5-base-v2"#"thenlper/gte-large"#"thenlper/gte-base"# #"sentence-transformers/gtr-t5-base"#"infgrad/stella-base-en-v2"# # #"ibm-granite/granite-embedding-278m-multilingual"# #"ibm-granite/granite-embedding-278m-multilingual"# # 

# reservoir / sampling
SUBSAMPLE = 300_000
CHUNK_TOKENS = 64

# training
TRAIN_BATCH = 256
TRAIN_EPOCHS = 3
LR = 1e-4

# Grid Search parameters
AB_EVAL_SIZE = 2000      # evaluate on 2000 dev examples from NQ
AB_EPOCHS = 3            # epochs per configuration

# mapper architecture params
MAPPER_LAYERS = 4
MAPPER_HEADS = 16

# Updated Output Directory
OUT_DIR = "grid_search_hierarchical_weights_gtr_e5"
os.makedirs(OUT_DIR, exist_ok=True)

# sqlite reservoir DB path (on-disk)
RESERVOIR_DB = "nq_reservoir.db"


# -------------------------
# Helper: unified loader
# -------------------------
def load_embedding_backend(name):
    if name.startswith("sentence-transformers/") and SentenceTransformer is not None:
        model = SentenceTransformer(name)
        tok = model.tokenizer 
        return tok, model, "sentencetransformer"
    else:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModel.from_pretrained(name)
        return tok, model, "hf"

@torch.no_grad()
def embed_sentences(texts, tokenizer, model, device, batch_size=64, max_length=512):
    embeddings = []
    is_st = isinstance(model, SentenceTransformer)

    for i in range(0, len(texts), batch_size):
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


# -------------------------
# Mappers, Losses, Dataset
# -------------------------
class EmbeddingPairsDataset(Dataset):
    def __init__(self, src_emb, tgt_emb):
        assert src_emb.shape == tgt_emb.shape
        self.src = src_emb.float()
        self.tgt = tgt_emb.float()

    def __len__(self):
        return self.src.shape[0]

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]


class TransformerEmbeddingMapper(nn.Module):
    """
    Cross-Attention Mapper: x (Source/Query) is cross-attended against memory (Target/Key/Value).
    """
    def __init__(self, emb_dim, hidden_dim=1024, n_layers=4, n_heads=16, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(emb_dim, hidden_dim)
        self.memory_proj = nn.Linear(emb_dim, hidden_dim) 
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim*4, dropout=dropout,
            activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        self.output_proj = nn.Linear(hidden_dim, emb_dim)
        self.ln = nn.LayerNorm(emb_dim)

    def forward(self, x: torch.Tensor, memory: torch.Tensor):
        if x.dim() == 2:
            x = x.unsqueeze(1)
            single = True
        else:
            single = False

        # 1. Project Query (Source)
        h = self.input_proj(x)
        
        # 2. Project Memory (Target)
        if memory.dim() == 2:
            mem_proj = self.memory_proj(memory.unsqueeze(1))
        else:
            mem_proj = self.memory_proj(memory)

        # 3. Cross-Attention
        h = self.transformer(tgt=h, memory=mem_proj) 
        
        out = self.output_proj(h)
        out = self.ln(out)
        return (out[:, 0, :], h) if single else (out, h)


# Loss components
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
        orig = orig[idx]
        mapped = mapped[idx]
    orig_n, mapped_n = F.normalize(orig.float(), dim=1), F.normalize(mapped.float(), dim=1)
    orig_sim = orig_n @ orig_n.T
    map_sim = mapped_n @ mapped_n.T
    triu = torch.triu_indices(orig_sim.size(0), orig_sim.size(1), 1)
    o_d = (1 - orig_sim[triu[0], triu[1]]).view(-1)
    m_d = (1 - map_sim[triu[0], triu[1]]).view(-1)
    ox, mx = o_d - o_d.mean(), m_d - m_d.mean()
    cov = (ox * mx).mean()
    corr = cov / ((o_d.std(unbiased=False)+eps)*(m_d.std(unbiased=False)+eps))
    corr = torch.clamp(corr, -1, 1)
    return 1.0 - corr

def global_similarity_loss(mapped, target):
    dist_mapped = pairwise_dist(mapped)
    dist_target = pairwise_dist(target)
    return F.l1_loss(dist_mapped, dist_target)

def local_similarity_loss(mapped, target, k=200):
    dist_target = pairwise_dist(target)
    dist_mapped = pairwise_dist(mapped)
    k = min(k, dist_target.size(1)-1)
    knn_idx = dist_target.topk(k+1, largest=False).indices[:,1:]
    losses = []
    for i in range(target.size(0)):
        j_idx = knn_idx[i]
        losses.append(F.l1_loss(dist_mapped[i, j_idx], dist_target[i, j_idx]))
    return torch.stack(losses).mean()

def structure_preservation_loss_spearman(original_embs, mapped_embs, sample_size=200):
    """Spearman correlation loss (1 - correlation) for training check."""
    N = original_embs.size(0)
    if N > sample_size:
        idx = torch.randperm(N, device=original_embs.device)[:sample_size]
        original_embs = original_embs[idx]
        mapped_embs = mapped_embs[idx]
    original = F.normalize(original_embs, dim=1).detach()
    mapped = F.normalize(mapped_embs, dim=1).detach()
    orig_sim = original @ original.T
    map_sim = mapped @ mapped.T
    triu = torch.triu_indices(orig_sim.size(0), orig_sim.size(1), offset=1)
    orig_d = (1 - orig_sim[triu[0], triu[1]]).cpu().numpy()
    map_d = (1 - map_sim[triu[0], triu[1]]).cpu().numpy()
    corr, _ = spearmanr(orig_d, map_d)
    if np.isnan(corr): corr = 0.0
    return torch.tensor(1.0 - corr, dtype=torch.float32, device=original_embs.device)

# -------------------------
# NEW: Weighted Flexible Loss (Adapted for Grid Search)
# -------------------------
def flexible_loss_weighted(model_src_emb, model_tgt_emb,
                           mapper_src2tgt, mapper_tgt2src,
                           weights: Dict[str, float],
                           k_local=10, temperature=0.05, tlm_mask_prob=0.15,
                           device='cpu'):
    """
    Computes loss based on explicit weights dictionary.
    Keys: 'con', 'tlm', 'structure', 'global', 'local', 'mse', 'cycle'
    """
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

    # --- 7) TLM (Weighted) ---
    w_tlm = weights.get('tlm', 0.0)
    if w_tlm > 0:
        B, D = model_src_emb.shape
        mask = (torch.rand(B, D, device=device) < tlm_mask_prob).float()
        
        noisy_src = model_src_emb * mask
        noisy_tgt = model_tgt_emb * mask
        
        recon_tgt, _ = mapper_src2tgt(noisy_src, memory=noisy_tgt)
        recon_src, _ = mapper_tgt2src(noisy_tgt, memory=noisy_src)
        
        loss_tlm_tgt = F.mse_loss(recon_tgt*mask, model_tgt_emb*mask)
        loss_tlm_src = F.mse_loss(recon_src*mask, model_src_emb*mask)
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
# Evaluation helpers (Spearman for Evaluation)
# -------------------------
@torch.no_grad()
def compute_structure_preservation(original_embs, mapped_embs, sample_size=1000):
    """
    Compute Spearman correlation between pairwise distances in original vs mapped space 
    (used for final evaluation).
    """
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


# -------------------------
# Utility Functions
# -------------------------

def sentence_split_into_64_token_chunks(text: str, tokenizer, chunk_tokens=CHUNK_TOKENS):
    toks = tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]
    chunks = []
    for i in range(0, len(toks), chunk_tokens):
        window = toks[i:i + chunk_tokens]
        if len(window) != chunk_tokens:
            continue
        decoded = tokenizer.decode(window, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        if decoded.strip():
            chunks.append(decoded)
    return chunks

def build_reservoir_sqlite(db_path: str, sample_size: int, dataset_name="jxm/nq_corpus_dpr", split="train", tokenizer_for_chunking=None):
    ds = load_dataset(dataset_name, split=split, streaming=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS sentences")
    cur.execute("CREATE TABLE sentences (id INTEGER PRIMARY KEY, text TEXT)")
    conn.commit()
    inserted = 0
    print(f"[*] Streaming dataset {dataset_name}:{split} to collect {sample_size} chunks and build DB...")
    for ex in tqdm(ds, desc="collecting chunks"):
        if inserted >= sample_size: break
        text = ex.get("text") or ex.get("document_text") or (ex.get("question", "") + " " + ex.get("answer", ""))
        if not text: continue
        chunks = sentence_split_into_64_token_chunks(text.replace("\n", " "), tokenizer_for_chunking, CHUNK_TOKENS)
        for ch in chunks:
            if inserted < sample_size:
                cur.execute("INSERT INTO sentences (id, text) VALUES (?, ?)", (inserted + 1, ch))
                inserted += 1
            if inserted % 10000 == 0: conn.commit()
            if inserted >= sample_size: break
    conn.commit()
    cur.execute("SELECT COUNT(*) FROM sentences")
    cnt = cur.fetchone()[0]
    conn.close()
    return cnt

def stream_reservoir_texts(db_path: str, subsample_size: int, chunk_size: int = 50000):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM sentences")
    total = cur.fetchone()[0]
    if total == 0:
        conn.close()
        return []
    load_count = min(total, subsample_size)
    offset = 0
    all_sents = []
    print(f"[*] Loading {load_count} texts from reservoir DB...")
    while offset < load_count:
        limit = min(chunk_size, load_count - offset)
        cur.execute("SELECT text FROM sentences LIMIT ? OFFSET ?", (limit, offset))
        rows = cur.fetchall()
        if not rows: break
        all_sents.extend([r[0] for r in rows])
        offset += len(rows)
    conn.close()
    return all_sents

def load_texts_for_training(db_path: str, subsample_size: int, tokenizer_for_chunking):
    if os.path.exists(db_path):
        print(f"[+] Reservoir DB {db_path} found. Attempting to load texts.")
        texts = stream_reservoir_texts(db_path, subsample_size)
        if len(texts) == 0:
            print("Warning: DB is empty. Proceeding to create a new reservoir.")
            os.remove(db_path)
            return load_texts_for_training(db_path, subsample_size, tokenizer_for_chunking)
        return texts
    else:
        print(f"[-] Reservoir DB {db_path} not found. Creating a new reservoir...")
        build_reservoir_sqlite(db_path, sample_size=subsample_size, tokenizer_for_chunking=tokenizer_for_chunking)
        texts = stream_reservoir_texts(db_path, subsample_size)
        return texts

def train_mapper_weighted(m_src2tgt, m_tgt2src, dataloader, optimizer, device,
                          weights: Dict[str, float], n_epochs=AB_EPOCHS,
                          k_local=10, temperature=0.05, tlm_mask_prob=0.15):
    m_src2tgt.to(device).train()
    m_tgt2src.to(device).train()
    
    for epoch in range(1, n_epochs+1):
        epoch_loss = 0.0
        n_batches = 0
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            loss, logs = flexible_loss_weighted(
                            src, tgt, m_src2tgt, m_tgt2src,
                            weights=weights,
                            k_local=k_local, temperature=temperature,
                            tlm_mask_prob=tlm_mask_prob, device=device
            )
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        if n_batches > 0:
            print(f"  Epoch {epoch}/{n_epochs} avg loss: {epoch_loss / n_batches:.6f}")
    return m_src2tgt, m_tgt2src

def evaluate_mapper_bidirectional(mapper_src2tgt, mapper_tgt2src, 
                                 src_model, src_tok, tgt_model, tgt_tok, 
                                 texts, device, batch_size=64):
    mapper_src2tgt.to(device).eval()
    mapper_tgt2src.to(device).eval()
    
    src_embs = embed_sentences(texts, src_tok, src_model, device, batch_size)
    tgt_embs = embed_sentences(texts, tgt_tok, tgt_model, device, batch_size)
    
    metrics = {}
    
    # 1. src -> tgt
    src_embs_dev = src_embs.to(device)
    tgt_embs_dev = tgt_embs.to(device)
    mapped_src2tgt, _ = mapper_src2tgt(src_embs_dev, memory=tgt_embs_dev)
    mapped_src2tgt = F.normalize(mapped_src2tgt, dim=-1).cpu()
    tgt_embs_norm = F.normalize(tgt_embs, dim=-1)
    
    N = mapped_src2tgt.size(0)
    top1_correct = 0
    ranks = []
    cos_sims = []
    
    block = 4096
    for i0 in range(0, N, block):
        i1 = min(N, i0 + block)
        sub_m = mapped_src2tgt[i0:i1]
        sim_block = sub_m @ tgt_embs_norm.T
        for local_idx in range(sim_block.size(0)):
            global_idx = i0 + local_idx
            row = sim_block[local_idx]
            values, indices = torch.sort(row, descending=True)
            rank = (indices == global_idx).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(rank)
            cos_sims.append(row[global_idx].item())
            if rank == 1: top1_correct += 1
    
    struc_corr = compute_structure_preservation(src_embs, mapped_src2tgt, sample_size=1000)
    metrics['src2tgt'] = {
        "mean_cosine": float(np.mean(cos_sims)),
        "top1_acc": float(top1_correct / max(1, N)),
        "mean_rank": float(np.mean(ranks)),
        "structure_preservation": float(struc_corr)
    }
    
    # 2. tgt -> src
    mapped_tgt2src, _ = mapper_tgt2src(tgt_embs_dev, memory=src_embs_dev)
    mapped_tgt2src = F.normalize(mapped_tgt2src, dim=-1).cpu()
    src_embs_norm = F.normalize(src_embs, dim=-1)
    
    top1_correct = 0
    ranks = []
    cos_sims = []
    for i0 in range(0, N, block):
        i1 = min(N, i0 + block)
        sub_m = mapped_tgt2src[i0:i1]
        sim_block = sub_m @ src_embs_norm.T
        for local_idx in range(sim_block.size(0)):
            global_idx = i0 + local_idx
            row = sim_block[local_idx]
            values, indices = torch.sort(row, descending=True)
            rank = (indices == global_idx).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(rank)
            cos_sims.append(row[global_idx].item())
            if rank == 1: top1_correct += 1
            
    struc_corr = compute_structure_preservation(tgt_embs, mapped_tgt2src, sample_size=1000)
    metrics['tgt2src'] = {
        "mean_cosine": float(np.mean(cos_sims)),
        "top1_acc": float(top1_correct / max(1, N)),
        "mean_rank": float(np.mean(ranks)),
        "structure_preservation": float(struc_corr)
    }
    return metrics


# -------------------------
# Orchestrator: HIERARCHICAL GRID SEARCH
# -------------------------
def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("Loading embedding backends...")
    src_tok, src_model, src_backend = load_embedding_backend(SRC_NAME)
    tgt_tok, tgt_model, tgt_backend = load_embedding_backend(TGT_NAME)

    if src_tok is None:
        if isinstance(tgt_tok, type(None)):
            from transformers import AutoTokenizer as _AutoTokenizer
            chunk_tok = _AutoTokenizer.from_pretrained(SRC_NAME)
        else:
            chunk_tok = tgt_tok
    else:
        chunk_tok = src_tok

    # 1) Load training texts
    texts_train = load_texts_for_training(RESERVOIR_DB, SUBSAMPLE, chunk_tok)
    N_train = len(texts_train)
    if N_train == 0: raise RuntimeError("No training sequences loaded.")
    print(f"[+] Using {N_train} sequences for training (cap={SUBSAMPLE})")

    # 2) Embed training texts
    print("[*] Embedding training candidates...")
    src_emb = embed_sentences(texts_train, src_tok, src_model, DEVICE, batch_size=64)
    tgt_emb = embed_sentences(texts_train, tgt_tok, tgt_model, DEVICE, batch_size=64)
    ab_src_emb = src_emb.cpu()
    ab_tgt_emb = tgt_emb.cpu()
    del src_emb, tgt_emb

    print("[*] Preparing dataset...")
    ab_ds = EmbeddingPairsDataset(ab_src_emb, ab_tgt_emb)
    ab_loader = DataLoader(ab_ds, batch_size=TRAIN_BATCH, shuffle=True)

    print("[*] Loading eval texts...")
    eval_ds = load_dataset("jxm/nq_corpus_dpr", split="dev", streaming=True)
    eval_texts = [x["text"] for x in eval_ds.take(AB_EVAL_SIZE)]

    # -------------------------
    # HIERARCHICAL EXPERIMENT SETUP
    # -------------------------
    experiments = []

    # 1. BASELINE (Equal Weights)
    experiments.append({
        'id': 'Baseline',
        'weights': {'con': 0.2, 'tlm': 0.2, 'structure': 0.2, 'global': 0.2, 'local': 0.2}
    })

    # 2. HIERARCHICAL GENERATION (24 Variations)
    sim_configs = [
        {'id': 'SimEq',  'w': [0.5, 0.5]},         # Equal internal
        {'id': 'SimCon', 'w': [0.75, 0.25]},       # Con Dominant
        {'id': 'SimTlm', 'w': [0.25, 0.75]}        # Tlm Dominant
    ]
    struct_configs = [
        {'id': 'StrEq',   'w': [1/3, 1/3, 1/3]},     # Equal internal
        {'id': 'StrMain', 'w': [0.50, 0.25, 0.25]},  # Structure Dominant
        {'id': 'StrGlo',  'w': [0.25, 0.50, 0.25]},  # Global Dominant
        {'id': 'StrLoc',  'w': [0.25, 0.25, 0.50]}   # Local Dominante
    ]
    macro_scenarios = [
        {'id': 'MACRO_SIM',  'w_sim': 0.75, 'w_str': 0.25}, # Sim 75%
        {'id': 'MACRO_STR',  'w_sim': 0.25, 'w_str': 0.75}  # Struct 75%
    ]

    for macro in macro_scenarios:
        for sim_conf in sim_configs:
            for str_conf in struct_configs:
                row_weights = {}
                # Calculate real weights
                row_weights['con'] = macro['w_sim'] * sim_conf['w'][0]
                row_weights['tlm'] = macro['w_sim'] * sim_conf['w'][1]
                row_weights['structure'] = macro['w_str'] * str_conf['w'][0]
                row_weights['global']    = macro['w_str'] * str_conf['w'][1]
                row_weights['local']     = macro['w_str'] * str_conf['w'][2]
                
                exp_id = f"{macro['id']} | {sim_conf['id']} + {str_conf['id']}"
                experiments.append({'id': exp_id, 'weights': row_weights})

    print(f"Running Grid Search with {len(experiments)} configurations...")

    results = []
    D = ab_src_emb.shape[1]

    # Run Experiments
    for idx, cfg in enumerate(tqdm(experiments, desc="Grid Configs")):
        exp_id = cfg['id']
        weights = cfg['weights']
        
        # Ensure mse/cycle are 0
        weights['mse'] = 0.0
        weights['cycle'] = 0.0

        print(f"\n[{idx+1}/{len(experiments)}] {exp_id}")
        
        # Fresh mappers
        mapper_src2tgt = TransformerEmbeddingMapper(D, hidden_dim=D, n_layers=MAPPER_LAYERS, n_heads=MAPPER_HEADS).to(DEVICE)
        mapper_tgt2src = TransformerEmbeddingMapper(D, hidden_dim=D, n_layers=MAPPER_LAYERS, n_heads=MAPPER_HEADS).to(DEVICE)
        optimizer = torch.optim.AdamW(list(mapper_src2tgt.parameters()) + list(mapper_tgt2src.parameters()),
                                     lr=LR, weight_decay=1e-6)

        mapper_src2tgt, mapper_tgt2src = train_mapper_weighted(
            mapper_src2tgt, mapper_tgt2src, ab_loader, optimizer, DEVICE,
            weights=weights, n_epochs=AB_EPOCHS,
            k_local=200, temperature=0.05, tlm_mask_prob=0.15
        )

        metrics = evaluate_mapper_bidirectional(mapper_src2tgt, mapper_tgt2src,
                                                src_model, src_tok, tgt_model, tgt_tok,
                                                eval_texts, DEVICE, batch_size=64)

        for direction in ['src2tgt', 'tgt2src']:
            row = {
                'configuration_id': exp_id,
                'direction': 'src→tgt' if direction == 'src2tgt' else 'tgt→src',
                # Log weights
                'w_con': weights['con'],
                'w_tlm': weights['tlm'],
                'w_struct': weights['structure'],
                'w_global': weights['global'],
                'w_local': weights['local'],
                # Metrics
                'mean_cosine': metrics[direction]['mean_cosine'],
                'top1_acc': metrics[direction]['top1_acc'],
                'mean_rank': metrics[direction]['mean_rank'],
                'structure_preservation': metrics[direction]['structure_preservation']
            }
            results.append(row)
            
        # Log concise result
        s2t = metrics['src2tgt']
        print(f"  -> S2T: Cos={s2t['mean_cosine']:.3f} | Top1={s2t['top1_acc']:.3f} | Rank={s2t['mean_rank']:.1f} | Struct={s2t['structure_preservation']:.3f}")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUT_DIR, f"grid_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, f"grid_results_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[+] Results saved to {csv_path}")

    # Plot: Bar chart of Top-1 Acc per Config (S2T)
    df_s2t = df[df['direction'] == 'src→tgt'].copy()
    if not df_s2t.empty:
        df_s2t = df_s2t.sort_values(by='top1_acc', ascending=False)
        plt.figure(figsize=(12, 8))
        sns.barplot(data=df_s2t, y='configuration_id', x='top1_acc', palette='viridis')
        plt.title('Top-1 Accuracy by Grid Configuration')
        plt.xlabel('Top-1 Acc')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "top1_ranking.png"))
        plt.close()

    print("Grid search finished.")

if __name__ == "__main__":
    main()