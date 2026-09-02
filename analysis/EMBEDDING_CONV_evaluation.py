#!/usr/bin/env python3
"""
Intrinsic Alignment Evaluation on BEIR Domains.
Applies the "NQ Alignment Test" (Cosine/Rank/Structure) to every BEIR dataset corpus.
Goal: Verify if the mapper aligns well across different domains (SciFact, FiQA, etc.)
"""

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from scipy.stats import spearmanr
from tqdm import tqdm
from beir.datasets.data_loader import GenericDataLoader

# optional SentenceTransformer backend
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# -------------------------
# Configuration
# -------------------------
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# Models
SRC_NAME = "infgrad/stella-base-en-v2" #"Qwen/Qwen3-Embedding-0.6B"#"sentence-transformers/gtr-t5-base"#"thenlper/gte-base" #"ibm-granite/granite-embedding-278m-multilingual"# #"intfloat/e5-base-v2" ## 
TGT_NAME = "thenlper/gte-base"#"thenlper/gte-large"#"infgrad/stella-base-en-v2"#"ibm-granite/granite-embedding-278m-multilingual"#"intfloat/e5-base-v2"# #"sentence-transformers/gtr-t5-base"# # #"ibm-granite/granite-embedding-278m-multilingual"# # # 

# Paths
RUN_TAG = "official_stella_gte_1M_MacroStr_SimTlm"
OUT_DIR = f"saved_mappers_{RUN_TAG}"
MAPPER_PATH_S2T = os.path.join(OUT_DIR, "official_mapper_src2tgt.pt")
MAPPER_PATH_T2S = os.path.join(OUT_DIR, "official_mapper_tgt2src.pt")
BEIR_DATA_ROOT = "/home/pallucchini/LLM_alignment/align_encoder/beir_datasets"

# Evaluation Settings
# We sample a subset of the corpus for the intrinsic test to keep it fast
SAMPLES_PER_DOMAIN = 5_000 
MAPPER_LAYERS = 4
MAPPER_HEADS = 16

# -------------------------
# Mapper Definition
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
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding Batches", leave=False):
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

def evaluate_mapper_bidirectional(m_s2t, m_t2s, s_m, s_t, t_m, t_t, texts, device, dataset_name="Unknown"):
    print(f"[*] Computing Embeddings for {dataset_name} ({len(texts)} samples)...")
    s_emb = embed_sentences(texts, s_t, s_m, device)
    t_emb = embed_sentences(texts, t_t, t_m, device)
    
    res = {}
    print(f"[*] Evaluating {dataset_name} Source -> Target...")
    res['src2tgt'] = evaluate_single_direction(m_s2t, s_emb, t_emb, device, f"{dataset_name} S2T")
    
    print(f"[*] Evaluating {dataset_name} Target -> Source...")
    res['tgt2src'] = evaluate_single_direction(m_t2s, t_emb, s_emb, device, f"{dataset_name} T2S")
    
    return res

def evaluate_single_direction(mapper, query_embs, target_embs, device, name):
    mapper.eval()
    
    # Run Mapper
    batch_size = 1000
    mapped_list = []
    
    # EXACT ALIGNMENT TEST: Map Query using its Target Pair as Memory
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
    
    for i in tqdm(range(0, mapped_norm.size(0), eval_bs), desc=f"Ranking {name}", leave=False):
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

def get_beir_corpus_sample(dataset_path, sample_size):
    """
    Loads BEIR corpus and returns a list of text strings.
    """
    try:
        corpus, _, _ = GenericDataLoader(dataset_path).load(split="test")
        doc_ids = list(corpus.keys())
        
        if len(doc_ids) > sample_size:
            sampled_ids = random.sample(doc_ids, sample_size)
        else:
            sampled_ids = doc_ids
            
        texts = []
        for did in sampled_ids:
            # Prefer title + text if title exists
            doc = corpus[did]
            txt = (doc.get("title", "") + " " + doc.get("text", "")).strip()
            if txt:
                texts.append(txt)
        return texts
    except Exception as e:
        print(f"[!] Error loading BEIR dataset at {dataset_path}: {e}")
        return []

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

    # 2. Extract Dimensions
    print("[*] Detecting Embedding Dimensions...")
    if s_type == "sentencetransformer":
        dim_s = s_mod.get_sentence_embedding_dimension()
    else:
        dim_s = s_mod.config.hidden_size
    D = dim_s
    print(f"    Dim: {D}")

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

    # 4. Iterate BEIR Datasets
    if not os.path.exists(BEIR_DATA_ROOT):
        print(f"[!] BEIR data root not found: {BEIR_DATA_ROOT}")
        return

    datasets = sorted([d for d in os.listdir(BEIR_DATA_ROOT) if os.path.isdir(os.path.join(BEIR_DATA_ROOT, d))])
    print(f"[*] Found {len(datasets)} BEIR datasets: {datasets}")

    all_results = []

    for ds_name in datasets:
        print(f"\n{'='*50}")
        print(f"PROCESSING: {ds_name}")
        print(f"{'='*50}")
        
        ds_path = os.path.join(BEIR_DATA_ROOT, ds_name)
        texts = get_beir_corpus_sample(ds_path, SAMPLES_PER_DOMAIN)
        
        if not texts or len(texts) < 100:
            print(f"[!] Skipping {ds_name} (insufficient data).")
            continue

        print(f"[+] Loaded {len(texts)} texts from {ds_name}")
        
        # Run Evaluation
        metrics = evaluate_mapper_bidirectional(
            m_s2t, m_t2s, s_mod, s_tok, t_mod, t_tok, texts, DEVICE, dataset_name=ds_name
        )
        
        # Collect
        for direction, m in metrics.items():
            row = {'dataset': ds_name, 'direction': direction, **m}
            all_results.append(row)
            print(f"   -> {direction}: Cosine={m['mean_cosine']:.4f}, Top1={m['top1_acc']:.4f}")

    # 5. Save & Print Summary
    res_df = pd.DataFrame(all_results)
    csv_path = os.path.join(OUT_DIR, "beir_domain_alignment_results.csv")
    res_df.to_csv(csv_path, index=False)
    
    print("\n=== FINAL DOMAIN ALIGNMENT RESULTS ===")
    print(res_df.to_string())
    print(f"[+] Results saved to {csv_path}")

if __name__ == "__main__":
    main()