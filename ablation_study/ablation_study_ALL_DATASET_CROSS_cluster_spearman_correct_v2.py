#!/usr/bin/env python3
"""
Alignment training on NQ with Cross-Attention Transformer Mapper.
Data loading: Loads the optimized sample from the .pt file (Spearman + K-Means).

Workflow:
 - Load pre-computed optimized embeddings from INPUT_SAMPLE_FILE.
 - Use this diverse subset for ablation training.
 - Run ablation: train mappers on the subset, evaluate on AB_EVAL_SIZE dev samples.
 - Save CSV and plots in saved_mappers/
"""

import os
import random
import sqlite3 
import json
from typing import List, Tuple, Optional
from tqdm import tqdm
from datetime import datetime
from itertools import combinations

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

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# ** REDIRECTED INPUT SOURCE **
INPUT_SAMPLE_FILE = "best_alignment_sample_10percent_spearman_gra_gte_300k.pt"#"best_alignment_sample_10percent_spearman_gra_gte_100k.pt"#"best_alignment_sample_10percent_spearman_qwen_gte_large_200k.pt"###"best_alignment_sample_10percent_spearman_qwen_gte_large.pt"

# embedding model names
SRC_NAME = "ibm-granite/granite-embedding-278m-multilingual"#"Qwen/Qwen3-Embedding-0.6B"#"sentence-transformers/gtr-t5-base"# #"intfloat/e5-base-v2" #"infgrad/stella-base-en-v2" #"thenlper/gte-base" ## 
TGT_NAME = "thenlper/gte-base"#"thenlper/gte-large"#"intfloat/e5-base-v2" #"sentence-transformers/gtr-t5-base"#"infgrad/stella-base-en-v2"# # #"ibm-granite/granite-embedding-278m-multilingual"# #"ibm-granite/granite-embedding-278m-multilingual"# # 

# training
TRAIN_BATCH = 256#32
TRAIN_EPOCHS = 3
LR = 1e-4

# ablation parameters
AB_EVAL_SIZE = 2000      
AB_EPOCHS = 3            

# mapper architecture params
MAPPER_LAYERS = 4
MAPPER_HEADS = 16

OUT_DIR = "saved_mappers_diverse_optimized_correct_gra_gte_300k" 
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# Helper: unified loader
# -------------------------
def load_embedding_backend(name):
    if name.startswith("sentence-transformers/") and SentenceTransformer is not None:
        model = SentenceTransformer(name)
        # SentenceTransformer objects store the HF tokenizer in the .tokenizer attribute
        tok = model.tokenizer 
        return tok, model, "sentencetransformer"
    else:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModel.from_pretrained(name)
        return tok, model, "hf"

@torch.no_grad()
def embed_sentences(texts, tokenizer, model, device, batch_size=64, max_length=512):
    embeddings = []
    
    # Check if we are using the SentenceTransformer wrapper
    is_st = isinstance(model, SentenceTransformer)

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        
        if is_st:
            # SentenceTransformer's encode is optimized and handles its own pooling
            emb = model.encode(batch, convert_to_tensor=True, device=device, show_progress_bar=False)
            embeddings.append(emb.cpu())
        else:
            # Manual HF pipeline (Mean Pooling)
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

        h = self.input_proj(x)
        if memory.dim() == 2:
            mem_proj = self.memory_proj(memory.unsqueeze(1))
        else:
            mem_proj = self.memory_proj(memory)

        h = self.transformer(tgt=h, memory=mem_proj) 
        
        out = self.output_proj(h)
        out = self.ln(out)
        return (out[:, 0, :], h) if single else (out, h)

def pairwise_dist(x):
    x = F.normalize(x, dim=1)
    return 1 - x @ x.T

def global_preservation_loss(orig, mapped, sample_size=200):
    N = orig.size(0)
    if N > sample_size:
        idx = torch.randperm(N, device=orig.device)[:sample_size]
        orig = orig[idx]
        mapped = mapped[idx]
    orig_n = F.normalize(orig, dim=1)
    mapped_n = F.normalize(mapped, dim=1)
    orig_sim = (orig_n @ orig_n.T).mean()
    mapped_sim = (mapped_n @ mapped_n.T).mean()
    return F.mse_loss(mapped_sim, orig_sim)

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

def flexible_loss(model_src_emb, model_tgt_emb,
                  mapper_src2tgt, mapper_tgt2src,
                  use_mse=True, use_cycle=True, use_con=True,
                  use_global=True, use_local=True, use_tlm=True,
                  use_structure=False,
                  structure_mode='pearson',
                  lambda_structure=1.0,
                  k_local=10, temperature=0.05, tlm_mask_prob=0.15,
                  device='cpu'):
    """
    Computes loss with selective components based on flags.
    ALL components are now fully symmetric (src2tgt and tgt2src).
    """
    model_src_emb = model_src_emb.to(device)
    model_tgt_emb = model_tgt_emb.to(device)
    
    total_loss = 0.0
    logs = {}

    # --- 1) Standard Forward Pass ---
    # src->tgt mapping: src is Query, tgt is Key/Value (Memory)
    mapped_src2tgt, _ = mapper_src2tgt(model_src_emb, memory=model_tgt_emb)
    # tgt->src mapping: tgt is Query, src is Key/Value (Memory)
    mapped_tgt2src, _ = mapper_tgt2src(model_tgt_emb, memory=model_src_emb)
    
    # --- 2) MSE (Pointwise) - Symmetric ---
    if use_mse:
        loss_mse = F.mse_loss(mapped_src2tgt, model_tgt_emb) + F.mse_loss(mapped_tgt2src, model_src_emb)
        total_loss = total_loss + loss_mse
        logs['mse'] = loss_mse.item()
    else:
        logs['mse'] = 0.0

    # --- 3) Cycle Consistency - Symmetric ---
    if use_cycle:
        # src -> tgt -> src
        src_cycle, _ = mapper_tgt2src(mapped_src2tgt.detach(), memory=model_src_emb)
        # tgt -> src -> tgt
        tgt_cycle, _ = mapper_src2tgt(mapped_tgt2src.detach(), memory=model_tgt_emb)
        
        loss_cycle = F.mse_loss(src_cycle, model_src_emb) + F.mse_loss(tgt_cycle, model_tgt_emb)
        total_loss = total_loss + loss_cycle
        logs['cycle'] = loss_cycle.item()
    else:
        logs['cycle'] = 0.0

    # --- 4) Contrastive NT-Xent - Symmetric ---
    if use_con:
        loss_con = nt_xent_loss(mapped_src2tgt, model_tgt_emb, temperature) + \
                   nt_xent_loss(mapped_tgt2src, model_src_emb, temperature)
        total_loss = total_loss + loss_con
        logs['contrastive'] = loss_con.item()
    else:
        logs['contrastive'] = 0.0

    # --- 5) Global similarity - Symmetric ---
    if use_global:
        # Preserve global structure of SRC in mapped S2T
        loss_global_s = global_similarity_loss(mapped_src2tgt, model_src_emb)
        # Preserve global structure of TGT in mapped T2S (Added)
        loss_global_t = global_similarity_loss(mapped_tgt2src, model_tgt_emb)
        
        loss_global = loss_global_s + loss_global_t
        total_loss = total_loss + loss_global
        logs['global'] = loss_global.item()
    else:
        logs['global'] = 0.0

    # --- 6) Local similarity - Symmetric ---
    if use_local:
        # Preserve local neighborhood of SRC in mapped S2T
        loss_local_s = local_similarity_loss(mapped_src2tgt, model_src_emb, k=k_local)
        # Preserve local neighborhood of TGT in mapped T2S (Added)
        loss_local_t = local_similarity_loss(mapped_tgt2src, model_tgt_emb, k=k_local)
        
        loss_local = loss_local_s + loss_local_t
        total_loss = total_loss + loss_local
        logs['local'] = loss_local.item()
    else:
        logs['local'] = 0.0

    # --- 7) TLM (Symmetric Cross-Reconstruction) ---
    if use_tlm:
        # 1. Generate Masks for both spaces
        # We mask random dimensions in Source and Target
        B, D = model_src_emb.shape
        #mask_s = (torch.rand(B, D, device=device) > tlm_mask_prob).float()
        #mask_t = (torch.rand(B, D, device=device) > tlm_mask_prob).float()
        mask = (torch.rand(B, D, device=device) < tlm_mask_prob).float()
        
        # 2. Create Noisy Inputs
        # We apply the mask to create the "context" for the mappers
        noisy_src = model_src_emb * mask
        noisy_tgt = model_tgt_emb * mask
        
        # 3. Run Mappers (Respecting their training direction)
        
        # Task A: Reconstruct the CLEAN TARGET
        # Query = Noisy Source, Memory = Noisy Target
        # The mapper must use the partial Source info AND partial Target info 
        # to hallucinates/reconstruct the complete Target vector.
        recon_tgt, _ = mapper_src2tgt(noisy_src, memory=noisy_tgt)
        
        # Task B: Reconstruct the CLEAN SOURCE
        # Query = Noisy Target, Memory = Noisy Source
        # The mapper must use the partial Target info AND partial Source info
        # to reconstruct the complete Source vector.
        recon_src, _ = mapper_tgt2src(noisy_tgt, memory=noisy_src)
        
        # 4. Loss
        # We compare the reconstruction to the ground truth (Clean vectors)
        loss_tlm_tgt = F.mse_loss(recon_tgt*mask, model_tgt_emb*mask)
        loss_tlm_src = F.mse_loss(recon_src*mask, model_src_emb*mask)
        
        loss_tlm = loss_tlm_tgt + loss_tlm_src
            
        total_loss = total_loss + loss_tlm
        logs['tlm'] = loss_tlm.item()
    else:
        logs['tlm'] = 0.0

    # --- 8) Structure Preservation Loss - Symmetric ---
    if use_structure:
        # Calculate structure loss for Src -> Mapped Src
        if structure_mode == 'spearman':
            loss_struct_s = structure_preservation_loss_spearman(model_src_emb, mapped_src2tgt)
            loss_struct_t = structure_preservation_loss_spearman(model_tgt_emb, mapped_tgt2src)
        else:
            loss_struct_s = structure_preservation_loss_pearson(model_src_emb, mapped_src2tgt)
            loss_struct_t = structure_preservation_loss_pearson(model_tgt_emb, mapped_tgt2src)
            
        loss_struct = loss_struct_s + loss_struct_t
        total_loss = total_loss + lambda_structure * loss_struct
        logs['structure'] = (lambda_structure * loss_struct).item()
    else:
        logs['structure'] = 0.0

    logs['total'] = total_loss.item()
    return total_loss, logs

@torch.no_grad()
def compute_structure_preservation(original_embs, mapped_embs, sample_size=1000):
    N = original_embs.size(0)
    if N > sample_size:
        indices = torch.randperm(N)[:sample_size]
        original_embs, mapped_embs = original_embs[indices], mapped_embs[indices]
    on, mn = F.normalize(original_embs, dim=1), F.normalize(mapped_embs, dim=1)
    os, ms = on @ on.T, mn @ mn.T
    triu = torch.triu_indices(os.size(0), os.size(1), offset=1)
    od, md = (1 - os[triu[0], triu[1]]).cpu().numpy(), (1 - ms[triu[0], triu[1]]).cpu().numpy()
    corr, _ = spearmanr(od, md)
    return corr

def train_mapper(m_src2tgt, m_tgt2src, dataloader, optimizer, device, n_epochs=AB_EPOCHS, **kwargs):
    m_src2tgt.to(device).train(); m_tgt2src.to(device).train()
    for epoch in range(1, n_epochs+1):
        for src, tgt in dataloader:
            optimizer.zero_grad()
            loss, _ = flexible_loss(src, tgt, m_src2tgt, m_tgt2src, device=device, **kwargs)
            loss.backward(); optimizer.step()
    return m_src2tgt, m_tgt2src

@torch.no_grad()
def evaluate_mapper_bidirectional(mapper_src2tgt, mapper_tgt2src, src_model, src_tok, tgt_model, tgt_tok, texts, device, batch_size=64):
    mapper_src2tgt.to(device).eval(); mapper_tgt2src.to(device).eval()
    se = embed_sentences(texts, src_tok, src_model, device, batch_size)
    te = embed_sentences(texts, tgt_tok, tgt_model, device, batch_size)
    metrics = {}
    for direction, mapper, queries, targets in [('src2tgt', mapper_src2tgt, se, te), ('tgt2src', mapper_tgt2src, te, se)]:
        mapped, _ = mapper(queries.to(device), memory=targets.to(device))
        mapped = F.normalize(mapped, dim=-1).cpu()
        targets_n = F.normalize(targets, dim=-1)
        sims = mapped @ targets_n.T
        ranks = [(torch.sort(sims[i], descending=True)[1] == i).nonzero().item() + 1 for i in range(len(texts))]
        metrics[direction] = {
            "mean_cosine": float(torch.diag(sims).mean()),
            "top1_acc": float(np.mean([1 if r == 1 else 0 for r in ranks])),
            "mean_rank": float(np.mean(ranks)),
            "structure_preservation": float(compute_structure_preservation(queries, mapped))
        }
    return metrics

# -------------------------
# Main Logic
# -------------------------
def main():
    if not os.path.exists(INPUT_SAMPLE_FILE):
        raise FileNotFoundError(f"Optimized sample file {INPUT_SAMPLE_FILE} not found.")

    print(f"[*] LOADING DATA FROM {INPUT_SAMPLE_FILE}...")
    data = torch.load(INPUT_SAMPLE_FILE)
    ab_src_emb, ab_tgt_emb = data["src_embeddings"], data["tgt_embeddings"]
    
    src_tok, src_model, _ = load_embedding_backend(SRC_NAME)
    tgt_tok, tgt_model, _ = load_embedding_backend(TGT_NAME)

    ab_loader = DataLoader(EmbeddingPairsDataset(ab_src_emb, ab_tgt_emb), batch_size=TRAIN_BATCH, shuffle=True)

    eval_ds = load_dataset("jxm/nq_corpus_dpr", split="dev", streaming=True)
    eval_texts = [x["text"] for x in eval_ds.take(AB_EVAL_SIZE)]

    # -------------------------
    # Ablation configurations
    # -------------------------
    loss_components = ['mse', 'cycle', 'con', 'global', 'local', 'tlm', 'structure']
    ablation_configs = []
    #ablation_configs_temp = []

    # singletons
    for comp in loss_components:
        cfg = {c: (c == comp) for c in loss_components}
        cfg['name'] = comp
        ablation_configs.append(cfg)
        #ablation_configs_temp.append(cfg)

    # combos sizes 2..len-1
    for r in range(2, len(loss_components) + 1):
        for combo in combinations(loss_components, r):
            cfg = {c: (c in combo) for c in loss_components}
            cfg['name'] = '+'.join(combo)
            ablation_configs.append(cfg)
            #ablation_configs_temp.append(cfg)
    
    #ablation_configs = []
    #for i in ablation_configs_temp:
    #    if i['tlm'] == True: #and i['con'] == True and i['local'] == True and i['structure'] == True and i['mse'] == False and i['cycle'] == False:
    #        ablation_configs.append(i)

    results = []
    D = ab_src_emb.shape[1]

    for cfg in tqdm(ablation_configs, desc="Ablation"):
        mapper_src2tgt = TransformerEmbeddingMapper(D, hidden_dim=D, n_layers=MAPPER_LAYERS, n_heads=MAPPER_HEADS).to(DEVICE)
        mapper_tgt2src = TransformerEmbeddingMapper(D, hidden_dim=D, n_layers=MAPPER_LAYERS, n_heads=MAPPER_HEADS).to(DEVICE)
        optimizer = torch.optim.AdamW(list(mapper_src2tgt.parameters()) + list(mapper_tgt2src.parameters()), lr=LR)

        train_mapper(mapper_src2tgt, mapper_tgt2src, ab_loader, optimizer, DEVICE,
                    use_mse=cfg['mse'], use_cycle=cfg['cycle'], use_con=cfg['con'],
                    use_global=cfg['global'], use_local=cfg['local'], use_tlm=cfg['tlm'],
                    use_structure=cfg['structure'], structure_mode='pearson', k_local=200)

        metrics = evaluate_mapper_bidirectional(mapper_src2tgt, mapper_tgt2src, src_model, src_tok, tgt_model, tgt_tok, eval_texts, DEVICE)

        for d in ['src2tgt', 'tgt2src']:
            results.append({'configuration': cfg['name'], 'direction': d, **{c: int(cfg[c]) for c in loss_components}, **metrics[d]})

    # Output Plots (Original Logic)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUT_DIR, f"ablation_{timestamp}"); os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(results); df.to_csv(os.path.join(out_dir, "ablation_results.csv"), index=False)
    
    df_avg = df.groupby('configuration').agg({'top1_acc': 'mean', 'structure_preservation': 'mean', **{c: 'first' for c in loss_components}}).reset_index()
    df_avg['num_components'] = df_avg[loss_components].sum(axis=1)

    plt.figure(figsize=(10,6))
    sns.lineplot(data=df_avg, x='num_components', y='top1_acc', marker='o'); plt.savefig(os.path.join(out_dir, "top1_vs_ncomp.png")); plt.close()
    plt.figure(figsize=(10,6))
    sns.lineplot(data=df_avg, x='num_components', y='structure_preservation', marker='s', color='green'); plt.savefig(os.path.join(out_dir, "structure_vs_ncomp.png")); plt.close()

if __name__ == "__main__":
    main()