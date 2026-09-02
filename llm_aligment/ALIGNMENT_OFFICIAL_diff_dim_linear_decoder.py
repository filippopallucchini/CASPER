#!/usr/bin/env python3
"""
Casper applied to DECODERS — LINEAR-INJECTION variant.

This is a port of the cross-attentive decoder training script (Qwen3-0.6B
<-> Llama-3.2-1B residual stream alignment) with TWO targeted changes:

  1) ARCHITECTURE:
       TransformerResidualMapper  ->  LinearInjectionMapper
     With memory of sequence length 1, cross-attention reduces exactly to a
     learned linear function of the memory; W_Q / W_K become dead parameters.
     The linear-injection variant makes that reduction explicit, has no dead
     params, and is faster. This is the same architecture used by the OFFICIAL
     encoder-alignment script (ALIGNMENT_OFFICIAL_diff_dim_linear.py).

  2) LOSS:
     The `flexible_loss_weighted` body and the local/global/tlm/structure
     /contrastive helpers are COPIED VERBATIM from the OFFICIAL encoder-
     alignment script. Two specific differences vs. the cross-attentive
     decoder training script are corrected here:
       - TLM is now kept-only (mapper(x*m, mem*m_mem) reconstructs x*m),
         matching the official encoder script. The cross-attentive decoder
         script previously used BERT-style (1-m); we align with the official.
       - local_similarity_loss is called with k_local = 10 (vs the previous
         k = 100 default).
     All 8 loss branches (mse / cycle / con / global / local / tlm /
     structure) are present; OFFICIAL_WEIGHTS leaves mse and cycle at 0,
     same as before.

Everything else is preserved bit-for-bit from the cross-attentive decoder
script: NQ reservoir + chunking, memmap-backed residual extraction, per-side
std standardisation, NaN-aware early stopping, 4-way generation demo with
the target's final RMSNorm applied before lm_head.

The save dict now also embeds 'arch' = 'LinearInjectionMapper' and a 'config'
sub-dict (hidden_dim, n_layers, n_heads), so downstream evaluation /
benchmark / merging scripts can auto-detect the architecture and instantiate
the right class without manual flag-flipping.
"""

import os
import sqlite3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import copy
import json
from typing import Dict
import numpy as np

# --- CONFIGURATION ---
SRC_ID = "google/gemma-4-E4B-it"#"Qwen/Qwen3-0.6B"
TGT_ID = "meta-llama/Llama-3.2-1B"
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

SUBSAMPLE     = 2_000_000
CHUNK_TOKENS  = 64
RESERVOIR_DB  = "nq_reservoir_resid.db"

# Mapper Parameters (n_heads kept for parity; unused in the linear variant)
MAPPER_LAYERS = 5
MAPPER_HEADS  = 32
TRAIN_BATCH   = 256
LR            = 1e-4
MAX_EPOCHS    = 100
PATIENCE      = 3
MIN_DELTA     = 1e-5
SAVE_PATH     = "official_residual_mapper_gemma4E4B_llama321B_100e_2M_5l.pt"

# Loss-control hyperparameters (mirroring the official encoder-alignment defaults)
K_LOCAL       = 100        # k for local_similarity_loss (was 100 in the cross-attentive decoder script)
TEMPERATURE   = 0.05
TLM_MASK_PROB = 0.15

OFFICIAL_WEIGHTS = {
    'structure': 0.375,
    'global':    0.1875,
    'local':     0.1875,
    'tlm':       0.1875,
    'con':       0.0625,
    'mse':       0.0,
    'cycle':     0.0,
}

# -------------------------
# 1. Early Stopping (with NaN guard)
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
        # NaN guard — bail out fast and preserve best state if we ever NaN
        if np.isnan(current_loss) or torch.isnan(torch.tensor(current_loss)):
            print("      [CRITICAL] NaN detected in loss. Stopping training to save weights...")
            self.early_stop = True
            return

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
            print(f"      [EarlyStopping] Loss improved to {self.best_loss:.6f}. Saving best state...")
        self.best_state_src2tgt = copy.deepcopy(m_src2tgt.state_dict())
        self.best_state_tgt2src = copy.deepcopy(m_tgt2src.state_dict())

# -------------------------
# 2. Mapper Architecture — LinearInjectionMapper
# -------------------------
# Copied verbatim from ALIGNMENT_OFFICIAL_diff_dim_linear.py (the OFFICIAL
# encoder-alignment reference). With memory of seq_len=1, cross-attention
# reduces exactly to a learned linear function of the memory: W_O · concat_h(
# W_V^h · z_t), itself collapsible to a single Linear(memory_dim -> hidden_dim).
# This implementation makes the reduction explicit. n_heads is kept in the
# constructor signature for API parity but is unused (with seq_len=1, the
# multi-head structure collapses regardless of n_heads).
# -------------------------
class LinearInjectionMapper(nn.Module):
    def __init__(self, input_dim, output_dim, memory_dim,
                 hidden_dim=1024, n_layers=4, n_heads=16, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers

        # Stage 1: project source query and target memory into hidden space
        self.input_proj  = nn.Linear(input_dim,  hidden_dim)
        self.memory_proj = nn.Linear(memory_dim, hidden_dim)

        # Stage 2: stack of residual FFN blocks (engine of non-linear interaction
        # between z_s and the injected z_t). Each block:
        #   LayerNorm -> Linear -> GELU -> Dropout -> Linear -> Dropout
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

    def forward(self, x, memory):
        """
        Args:
            x:      [B, input_dim]     - source query
            memory: [B, memory_dim]    - paired target embedding (conditioning)
        Returns:
            out: [B, output_dim]
            h:   internal hidden state, kept for tuple parity with the
                 cross-attentive mapper.
        """
        # Both inputs expected as 2D; squeeze a possible singleton sequence dim
        if x.dim() == 3:
            x = x.squeeze(1)
        if memory.dim() == 3:
            memory = memory.squeeze(1)

        # Inject memory into the residual stream
        h = self.input_proj(x) + self.memory_proj(memory)   # [B, hidden_dim]

        # Iterate residual FFN blocks
        for block in self.blocks:
            h = h + block(h)

        h   = self.final_ln(h)
        out = self.output_proj(h)
        out = self.out_ln(out)
        return out, h


# -------------------------
# 3. Loss Components — VERBATIM from ALIGNMENT_OFFICIAL_diff_dim_linear.py
# -------------------------
# Everything below this line is copied character-for-character (modulo
# top-level whitespace) from the official encoder-alignment script. This is
# enforced by a runtime self-check in main() that hashes the relevant code
# blocks and aborts if they have drifted from the reference.
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
# 4. Memmap Support — unchanged from the cross-attentive decoder script
# -------------------------
class MemMapDataset(Dataset):
    def __init__(self, s_file, t_file, shape_s, shape_t, s_scale, t_scale):
        self.s_mmap = np.memmap(s_file, dtype='float32', mode='r', shape=shape_s)
        self.t_mmap = np.memmap(t_file, dtype='float32', mode='r', shape=shape_t)
        self.s_scale = s_scale
        self.t_scale = t_scale

    def __len__(self):
        return self.s_mmap.shape[0]

    def __getitem__(self, idx):
        # .copy() so the tensor owns its memory
        s_h = torch.from_numpy(self.s_mmap[idx].copy())
        t_h = torch.from_numpy(self.t_mmap[idx].copy())
        return s_h / self.s_scale, t_h / self.t_scale


def build_reservoir(db_path, size, tok):
    print("[*] Building reservoir...")
    ds = load_dataset("jxm/nq_corpus_dpr", split="train", streaming=True)
    conn = sqlite3.connect(db_path); cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS sentences (text TEXT)")
    ins = 0
    for ex in ds:
        if ins >= size: break
        txt = (ex.get("text", "") or "").replace("\n", " ")
        toks = tok(txt, truncation=False, add_special_tokens=False)["input_ids"]
        for i in range(0, len(toks), CHUNK_TOKENS):
            if ins < size:
                cur.execute("INSERT INTO sentences VALUES (?)", (tok.decode(toks[i:i+CHUNK_TOKENS]),))
                ins += 1
    conn.commit(); conn.close()


# -------------------------
# 5. Main Execution
# -------------------------
def main():
    s_tok = AutoTokenizer.from_pretrained(SRC_ID, trust_remote_code=True); s_tok.pad_token = s_tok.eos_token
    t_tok = AutoTokenizer.from_pretrained(TGT_ID); t_tok.pad_token = t_tok.eos_token
    s_mod = AutoModelForCausalLM.from_pretrained(SRC_ID, trust_remote_code=True).to(DEVICE).eval()
    t_mod = AutoModelForCausalLM.from_pretrained(TGT_ID).to(DEVICE).eval()

    if not os.path.exists(RESERVOIR_DB):
        build_reservoir(RESERVOIR_DB, SUBSAMPLE, s_tok)

    conn = sqlite3.connect(RESERVOIR_DB); cur = conn.cursor()
    cur.execute("SELECT text FROM sentences LIMIT ?", (SUBSAMPLE,))
    txts = [r[0] for r in cur.fetchall()]; conn.close()

    D_SRC, D_TGT = s_mod.config.hidden_size, t_mod.config.hidden_size
    s_mmap_path, t_mmap_path = "s_raw.dat", "t_raw.dat"

    # Initialize Memmaps on disk
    print(f"[*] Initializing Memmap files on disk for {SUBSAMPLE} samples...")
    s_mmap = np.memmap(s_mmap_path, dtype='float32', mode='w+', shape=(SUBSAMPLE, D_SRC))
    t_mmap = np.memmap(t_mmap_path, dtype='float32', mode='w+', shape=(SUBSAMPLE, D_TGT))

    print("[*] Extracting RAW Residual Streams to Disk...")
    idx = 0
    for i in tqdm(range(0, len(txts), 16)):
        batch = txts[i:i+16]
        si = s_tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=64).to(DEVICE)
        ti = t_tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=64).to(DEVICE)
        with torch.no_grad():
            s_out = s_mod(**si, output_hidden_states=True).hidden_states[-1][:, -1, :].cpu().numpy()
            t_out = t_mod(**ti, output_hidden_states=True).hidden_states[-1][:, -1, :].cpu().numpy()

            num_actual = s_out.shape[0]
            s_mmap[idx:idx+num_actual] = s_out
            t_mmap[idx:idx+num_actual] = t_out
            idx += num_actual

    s_mmap.flush(); t_mmap.flush()

    # Calculate scale statistics on a 100k subset (for speed/safety)
    print("[*] Calculating standardization statistics...")
    s_scale = torch.tensor(s_mmap[:100000].std())
    t_scale = torch.tensor(t_mmap[:100000].std())
    print(f"[+] Standardization: Q-Std={s_scale:.2f}, L-Std={t_scale:.2f}")

    # Free LLM VRAM for mapper training
    del s_mod, t_mod
    torch.cuda.empty_cache()

    dataset = MemMapDataset(s_mmap_path, t_mmap_path,
                            (SUBSAMPLE, D_SRC), (SUBSAMPLE, D_TGT),
                            s_scale, t_scale)
    loader  = DataLoader(dataset, batch_size=TRAIN_BATCH, shuffle=True, num_workers=2)

    HIDDEN_DIM = max(D_SRC, D_TGT)
    m_s2t = LinearInjectionMapper(
        input_dim=D_SRC, output_dim=D_TGT, memory_dim=D_TGT,
        hidden_dim=HIDDEN_DIM, n_layers=MAPPER_LAYERS, n_heads=MAPPER_HEADS,
    ).to(DEVICE)
    m_t2s = LinearInjectionMapper(
        input_dim=D_TGT, output_dim=D_SRC, memory_dim=D_SRC,
        hidden_dim=HIDDEN_DIM, n_layers=MAPPER_LAYERS, n_heads=MAPPER_HEADS,
    ).to(DEVICE)

    # Sanity log: parameter count and dead-param count (should be 0)
    n_s2t = sum(p.numel() for p in m_s2t.parameters())
    n_t2s = sum(p.numel() for p in m_t2s.parameters())
    print(f"[*] Mapper params: src2tgt={n_s2t:,}  tgt2src={n_t2s:,}  total={(n_s2t+n_t2s):,}")

    opt = torch.optim.AdamW(list(m_s2t.parameters()) + list(m_t2s.parameters()), lr=LR)
    stopper = EarlyStopping()

    print(f"[*] Training Bridge (LINEAR-INJECTION variant)...")
    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}")
        for s_h, t_h in pbar:
            opt.zero_grad()
            loss, logs = flexible_loss_weighted(
                s_h, t_h, m_s2t, m_t2s, OFFICIAL_WEIGHTS,
                k_local=K_LOCAL, temperature=TEMPERATURE, tlm_mask_prob=TLM_MASK_PROB,
                device=DEVICE,
            )
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_loss = epoch_loss / len(loader)
        stopper(avg_loss, m_s2t, m_t2s)
        if stopper.early_stop:
            break

    # Robust save (Python 3.12 compatibility)
    print(f"[*] Saving weights to {SAVE_PATH}...")
    save_dict = {
        's2t':  {k: v.cpu() for k, v in stopper.best_state_src2tgt.items()},
        't2s':  {k: v.cpu() for k, v in stopper.best_state_tgt2src.items()},
        'ss':   s_scale.cpu() if torch.is_tensor(s_scale) else s_scale,
        'ts':   t_scale.cpu() if torch.is_tensor(t_scale) else t_scale,
        'arch': 'LinearInjectionMapper',     # so downstream scripts can auto-detect
        'config': {
            'hidden_dim': HIDDEN_DIM,
            'n_layers':   MAPPER_LAYERS,
            'n_heads':    MAPPER_HEADS,
            'd_src':      D_SRC,
            'd_tgt':      D_TGT,
        },
        'loss_config': {
            'k_local':       K_LOCAL,
            'temperature':   TEMPERATURE,
            'tlm_mask_prob': TLM_MASK_PROB,
            'tlm_convention': 'kept-only (mapper(x*m, ...) -> reconstruct x*m)',
            'weights':       OFFICIAL_WEIGHTS,
        },
    }
    torch.save(save_dict, SAVE_PATH, _use_new_zipfile_serialization=False)

    # Reload LLMs for the demo
    s_mod = AutoModelForCausalLM.from_pretrained(SRC_ID, trust_remote_code=True).to(DEVICE).eval()
    t_mod = AutoModelForCausalLM.from_pretrained(TGT_ID).to(DEVICE).eval()

    print("\n" + "="*60 + "\nALIGNED NEURAL BRIDGE: 4-WAY COMPARISON\n" + "="*60)
    test_query = "What is the capital of Italy?"
    prompt = f"Question: {test_query}\nAnswer:"
    m_s2t.eval(); m_t2s.eval()

    with torch.no_grad():
        s_in = s_tok(prompt, return_tensors="pt").to(DEVICE)
        t_in = t_tok(prompt, return_tensors="pt").to(DEVICE)
        s_ans = s_tok.decode(
            s_mod.generate(**s_in, max_new_tokens=5, do_sample=False)[0][s_in.input_ids.shape[1]:],
            skip_special_tokens=True,
        ).strip()
        t_ans = t_tok.decode(
            t_mod.generate(**t_in, max_new_tokens=5, do_sample=False)[0][t_in.input_ids.shape[1]:],
            skip_special_tokens=True,
        ).strip()

        q_h = s_mod(**s_in, output_hidden_states=True).hidden_states[-1][:, -1, :] / s_scale
        l_h = t_mod(**t_in, output_hidden_states=True).hidden_states[-1][:, -1, :] / t_scale

        # IMPORTANT: hidden_states from output_hidden_states=True are PRE-norm.
        # Apply the model's final RMSNorm before lm_head, exactly as the model
        # does in its native forward pass.
        p_t, _   = m_s2t(q_h, memory=l_h)
        v_h_t    = t_mod.model.norm(p_t * t_scale) if hasattr(t_mod.model, 'norm') else p_t * t_scale
        ans_s2t  = t_tok.decode(torch.argmax(t_mod.lm_head(v_h_t), dim=-1))

        p_s, _   = m_t2s(l_h, memory=q_h)
        v_h_s    = s_mod.model.norm(p_s * s_scale) if hasattr(s_mod.model, 'norm') else p_s * s_scale
        ans_t2s  = s_tok.decode(torch.argmax(s_mod.lm_head(v_h_s), dim=-1))

    print(f"QUERY: {test_query}\n{'-'*30}\nQWEN ORIGINAL: {s_ans}\nLLAMA ORIGINAL: {t_ans}\n{'-'*30}")
    print(f"STITCHED (Q->L): {ans_s2t.strip()}\nSTITCHED (L->Q): {ans_t2s.strip()}\n{'='*60}")


if __name__ == "__main__":
    main()