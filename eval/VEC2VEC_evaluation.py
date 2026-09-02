#!/usr/bin/env python3
"""
Standalone Evaluation of Pre-Trained Mappers on NQ Corpus.
Loads saved mappers and automatically detects model dimensions.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from scipy.stats import spearmanr
from tqdm import tqdm

# optional SentenceTransformer backend
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# -------------------------
# Configuration
# -------------------------
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# Model Names (Must match the ones used in training)
SRC_NAME = "Qwen/Qwen3-Embedding-0.6B"#"ibm-granite/granite-embedding-278m-multilingual"#"sentence-transformers/gtr-t5-base"# #"intfloat/e5-base-v2" #"infgrad/stella-base-en-v2" #"thenlper/gte-base" ## 
TGT_NAME = "intfloat/e5-base-v2"#"infgrad/stella-base-en-v2"#"thenlper/gte-base"#"thenlper/gte-large"# #"sentence-transformers/gtr-t5-base"# # #"ibm-granite/granite-embedding-278m-multilingual"# #"ibm-granite/granite-embedding-278m-multilingual"# # 

# Paths
RUN_TAG = "official_qwen3_e5_1M_MacroStr_SimTlm"
OUT_DIR = f"saved_mappers_{RUN_TAG}"
MAPPER_PATH_S2T = os.path.join(OUT_DIR, "official_mapper_src2tgt.pt")
MAPPER_PATH_T2S = os.path.join(OUT_DIR, "official_mapper_tgt2src.pt")

# Evaluation Settings
AB_EVAL_SIZE = 65_536  # Size of evaluation set
MAPPER_LAYERS = 4
MAPPER_HEADS = 16

# -------------------------
# Model Definitions
# -------------------------
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

    def forward(self, x, memory):
        if x.dim() == 2:
            x = x.unsqueeze(1)
            single = True
        else:
            single = False
        h = self.input_proj(x)
        mem_proj = self.memory_proj(memory.unsqueeze(1) if memory.dim() == 2 else memory)
        h = self.transformer(tgt=h, memory=mem_proj) 
        out = self.output_proj(h)
        out = self.ln(out)
        return (out[:, 0, :], h) if single else (out, h)

# -------------------------
# Helpers
# -------------------------
def load_embedding_backend(name):
    """
    Loads model and returns: tokenizer, model, backend_type ('hf' or 'sentencetransformer')
    """
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
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding Batches"):
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

@torch.no_grad()
def compute_structure_preservation_spearman(original_embs, mapped_embs, sample_size=1000):
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

def evaluate_mapper_bidirectional(m_s2t, m_t2s, s_m, s_t, t_m, t_t, texts, device):
    print("[*] Computing Embeddings for Evaluation...")
    s_emb = embed_sentences(texts, s_t, s_m, device)
    t_emb = embed_sentences(texts, t_t, t_m, device)
    
    res = {}
    print("[*] Evaluating Source -> Target...")
    res['src2tgt'] = evaluate_single_direction(m_s2t, s_emb, t_emb, device, "src2tgt")
    
    print("[*] Evaluating Target -> Source...")
    res['tgt2src'] = evaluate_single_direction(m_t2s, t_emb, s_emb, device, "tgt2src")
    
    return res

def evaluate_single_direction(mapper, query_embs, target_embs, device, name):
    mapper.eval()
    
    # Run Mapper
    batch_size = 1000
    mapped_list = []
    
    # Map Query using Target as Memory (since pairs are aligned in NQ)
    for i in range(0, query_embs.size(0), batch_size):
        q_batch = query_embs[i:i+batch_size].to(device)
        m_batch = target_embs[i:i+batch_size].to(device)
        with torch.no_grad():
            out, _ = mapper(q_batch, memory=m_batch)
        mapped_list.append(out.cpu())
        
    mapped_embs = torch.cat(mapped_list, dim=0)
    mapped_norm = F.normalize(mapped_embs, dim=-1)
    target_norm = F.normalize(target_embs, dim=-1)
    
    ranks, sims = [], []
    eval_bs = 256
    target_norm_gpu = target_norm.to(device)
    
    for i in tqdm(range(0, mapped_norm.size(0), eval_bs), desc=f"Ranking {name}"):
        batch_map = mapped_norm[i:i+eval_bs].to(device)
        
        # Similarity Matrix
        sim_matrix = torch.matmul(batch_map, target_norm_gpu.T)
        
        start_idx = i
        batch_size_actual = batch_map.size(0)
        
        for local_idx in range(batch_size_actual):
            global_idx = start_idx + local_idx
            
            scores = sim_matrix[local_idx]
            true_score = scores[global_idx].item()
            
            # Rank
            rank = (scores > true_score).sum().item() + 1
            ranks.append(rank)
            sims.append(true_score)

    return {
        "mean_cosine": np.mean(sims),
        "top1_acc": sum(1 for r in ranks if r == 1) / len(ranks),
        "mean_rank": np.mean(ranks),
        "structure_preservation": compute_structure_preservation_spearman(query_embs, mapped_embs)
    }

# -------------------------
# Main Execution
# -------------------------
def main():
    # 1. Load Models & Detect Type
    print("[*] Loading Embedding Models...")
    s_tok, s_mod, s_type = load_embedding_backend(SRC_NAME)
    t_tok, t_mod, t_type = load_embedding_backend(TGT_NAME)
    
    if hasattr(s_mod, 'to'): s_mod = s_mod.to(DEVICE)
    if hasattr(t_mod, 'to'): t_mod = t_mod.to(DEVICE)

    # 2. Extract Dimensions (The New Part)
    print("[*] Detecting Embedding Dimensions...")
    
    # Source Dim
    if s_type == "sentencetransformer":
        dim_s = s_mod.get_sentence_embedding_dimension()
    else:
        dim_s = s_mod.config.hidden_size
        
    # Target Dim
    if t_type == "sentencetransformer":
        dim_t = t_mod.get_sentence_embedding_dimension()
    else:
        dim_t = t_mod.config.hidden_size
        
    print(f"    Source Dim: {dim_s}")
    print(f"    Target Dim: {dim_t}")
    
    if dim_s != dim_t:
        print("[!] Warning: Dimension mismatch! The current Mapper requires equal dims.")
        # If your mapper handles different dims, you can proceed. 
        # But standard TransformerDecoder usually expects d_model == dim.
        # Proceeding assuming they are equal or mapper handles it (currently hardcoded to D=dim_s).
    
    D = dim_s

    # 3. Load Mappers
    print(f"[*] Loading Mappers from {OUT_DIR}...")
    m_s2t = TransformerEmbeddingMapper(D, hidden_dim=D, n_layers=MAPPER_LAYERS, n_heads=MAPPER_HEADS).to(DEVICE)
    m_t2s = TransformerEmbeddingMapper(D, hidden_dim=D, n_layers=MAPPER_LAYERS, n_heads=MAPPER_HEADS).to(DEVICE)

    try:
        m_s2t.load_state_dict(torch.load(MAPPER_PATH_S2T, map_location=DEVICE))
        m_t2s.load_state_dict(torch.load(MAPPER_PATH_T2S, map_location=DEVICE))
        print("[+] Mappers loaded successfully.")
    except Exception as e:
        print(f"[!] Error loading mappers: {e}")
        return

    # 4. Load Evaluation Data
    print(f"[*] Loading {AB_EVAL_SIZE} samples from NQ Corpus...")
    try:
        eval_ds = load_dataset("jxm/nq_corpus_dpr", split="dev")
    except Exception:
        print("Fallback: Using train split as dev is missing")
        eval_ds = load_dataset("jxm/nq_corpus_dpr", split="train")
        
    eval_texts = eval_ds.select(range(min(AB_EVAL_SIZE, len(eval_ds))))["text"]
    
    # 5. Run Evaluation
    print("[*] Starting Evaluation Loop...")
    metrics = evaluate_mapper_bidirectional(m_s2t, m_t2s, s_mod, s_tok, t_mod, t_tok, eval_texts, DEVICE)
    
    # 6. Save & Print
    res_df = pd.DataFrame([
        {'direction': 'src2tgt', **metrics['src2tgt']},
        {'direction': 'tgt2src', **metrics['tgt2src']}
    ])
    
    csv_path = os.path.join(OUT_DIR, "standalone_eval_results.csv")
    res_df.to_csv(csv_path, index=False)
    
    print("\n=== FINAL EVALUATION RESULTS ===")
    print(res_df.to_string())
    print(f"[+] Results saved to {csv_path}")

if __name__ == "__main__":
    main()