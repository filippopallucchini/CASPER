#!/usr/bin/env python3
"""
SOTA Alignment Evaluation: Linear Mapper (Wu & Dredze)
Bidirectional Evaluation on NQ Corpus with Auto-Filename Detection.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from scipy.stats import spearmanr
from tqdm import tqdm

# Ensure sklearn is available
try:
    from sklearn.preprocessing import StandardScaler
except ImportError:
    raise ImportError("Please install scikit-learn: pip install scikit-learn")

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

# Model Names (Use the EXACT names used during training)
# The script will use the last part of the name (e.g., 'gtr-t5-base') to find files.

SRC_NAME = "ibm-granite/granite-embedding-278m-multilingual"#"sentence-transformers/gtr-t5-base"#"Qwen/Qwen3-Embedding-0.6B"# #"intfloat/e5-base-v2" #"infgrad/stella-base-en-v2" #"thenlper/gte-base" ## 
TGT_NAME = "thenlper/gte-base"#"intfloat/e5-base-v2"#"infgrad/stella-base-en-v2"##"thenlper/gte-large"# #"sentence-transformers/gtr-t5-base"# # #"ibm-granite/granite-embedding-278m-multilingual"# #"ibm-granite/granite-embedding-278m-multilingual"# # 


# Directory containing the .pt mapper files
MAPPERS_DIR = "wu_dredze_alignment_method"

# Evaluation Settings
AB_EVAL_SIZE = 65_536  # Official evaluation size
# AB_EVAL_SIZE = 1000  # Uncomment for debugging

# -------------------------
# Helpers
# -------------------------
def get_clean_model_name(full_name):
    """
    Extracts 'gtr-t5-base' from 'sentence-transformers/gtr-t5-base'
    """
    return full_name.strip("/").split("/")[-1]

def load_embedding_backend(name):
    print(f"[*] Loading Model: {name}")
    if name.startswith("sentence-transformers/") and SentenceTransformer is not None:
        model = SentenceTransformer(name, trust_remote_code=True)
        tok = model.tokenizer 
        return tok, model, "sentencetransformer"
    else:
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        model = AutoModel.from_pretrained(name, trust_remote_code=True)
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
    """
    Computes Spearman correlation of pairwise distances before and after mapping.
    """
    N = original_embs.size(0)
    if N > sample_size:
        indices = torch.randperm(N)[:sample_size]
        original_embs = original_embs[indices]
        mapped_embs = mapped_embs[indices]
    
    # Normalize
    original_norm = F.normalize(original_embs, dim=1)
    mapped_norm = F.normalize(mapped_embs, dim=1)
    
    # Distance Matrices (1 - Cosine)
    original_sim = original_norm @ original_norm.T
    mapped_sim = mapped_norm @ mapped_norm.T
    
    triu_indices = torch.triu_indices(original_sim.size(0), original_sim.size(1), offset=1)
    
    # Extract upper triangle values
    original_dists = (1 - original_sim[triu_indices[0], triu_indices[1]]).cpu().numpy()
    mapped_dists = (1 - mapped_sim[triu_indices[0], triu_indices[1]]).cpu().numpy()
    
    correlation, _ = spearmanr(original_dists, mapped_dists)
    return correlation

def evaluate_linear_projection(src_emb_raw, tgt_emb_raw, checkpoint, device, description=""):
    """
    Applies Scikit-Learn Scalers + Linear Projection W and evaluates.
    """
    print(f"\n--- Evaluating: {description} ---")
    
    # 1. Unpack Checkpoint
    W = checkpoint["W"]      # (D, D) numpy
    scX = checkpoint["scX"]  # Source Scaler
    scY = checkpoint["scY"]  # Target Scaler
    
    # 2. Scale Inputs (Convert to Numpy -> Scale)
    print("    Applying Scalers...")
    X_np = src_emb_raw.numpy()
    Y_np = tgt_emb_raw.numpy()
    
    X_scaled = scX.transform(X_np)
    Y_scaled = scY.transform(Y_np)
    
    # 3. Apply Linear Mapping
    # Logic: Mapped_Source = Scaled_Source @ W
    print("    Applying Projection W...")
    X_mapped_np = X_scaled @ W
    
    # 4. Convert to Tensors for GPU Metrics
    # We compare Mapped Source vs Scaled Target (as the 'Ground Truth' space)
    mapped_tensor = torch.tensor(X_mapped_np, dtype=torch.float32)
    target_tensor = torch.tensor(Y_scaled, dtype=torch.float32)
    
    mapped_norm = F.normalize(mapped_tensor, dim=-1)
    target_norm = F.normalize(target_tensor, dim=-1)
    
    # 5. Ranking Evaluation
    ranks, sims = [], []
    eval_bs = 256
    target_norm_gpu = target_norm.to(device)
    
    print("    Computing Retrieval Metrics...")
    for i in tqdm(range(0, mapped_norm.size(0), eval_bs), desc="    Ranking"):
        batch_map = mapped_norm[i:i+eval_bs].to(device)
        
        # Dense Retrieval: Dot product with ALL targets
        sim_matrix = torch.matmul(batch_map, target_norm_gpu.T)
        
        batch_size_actual = batch_map.size(0)
        for local_idx in range(batch_size_actual):
            global_idx = i + local_idx
            scores = sim_matrix[local_idx]
            true_score = scores[global_idx].item()
            
            # Rank = 1 + count(scores > true_score)
            rank = (scores > true_score).sum().item() + 1
            ranks.append(rank)
            sims.append(true_score)

    return {
        "direction": description,
        "mean_cosine": np.mean(sims),
        "top1_acc": sum(1 for r in ranks if r == 1) / len(ranks),
        "mean_rank": np.mean(ranks),
        "structure_preservation": compute_structure_preservation_spearman(src_emb_raw, mapped_tensor)
    }

# -------------------------
# Main Execution
# -------------------------
def main():
    # 1. Construct File Paths
    clean_src = get_clean_model_name(SRC_NAME)
    clean_tgt = get_clean_model_name(TGT_NAME)
    
    # Pattern: mapper_{SOURCE}_{TARGET}.pt
    file_s2t = f"mapper_{clean_src}_{clean_tgt}.pt"
    file_t2s = f"mapper_{clean_tgt}_{clean_src}.pt"
    
    path_s2t = os.path.join(MAPPERS_DIR, file_s2t)
    path_t2s = os.path.join(MAPPERS_DIR, file_t2s)
    
    print(f"[*] Looking for mappers in: {MAPPERS_DIR}")
    print(f"    Src->Tgt File: {file_s2t}")
    print(f"    Tgt->Src File: {file_t2s}")

    chk_s2t, chk_t2s = None, None

    if os.path.exists(path_s2t):
        print(f"[+] Found Src->Tgt mapper.")
        chk_s2t = torch.load(path_s2t, weights_only=False)
    else:
        print(f"[!] Warning: Src->Tgt mapper not found.")

    if os.path.exists(path_t2s):
        print(f"[+] Found Tgt->Src mapper.")
        chk_t2s = torch.load(path_t2s, weights_only=False)
    else:
        print(f"[!] Warning: Tgt->Src mapper not found.")
        
    if not chk_s2t and not chk_t2s:
        print("[!] No mappers found. Exiting.")
        return

    # 2. Load Models
    print("\n[*] Loading Embedding Models...")
    s_tok, s_mod, _ = load_embedding_backend(SRC_NAME)
    t_tok, t_mod, _ = load_embedding_backend(TGT_NAME)
    
    if hasattr(s_mod, 'to'): s_mod = s_mod.to(DEVICE)
    if hasattr(t_mod, 'to'): t_mod = t_mod.to(DEVICE)

    # 3. Load Data
    print(f"\n[*] Loading {AB_EVAL_SIZE} samples from NQ Corpus...")
    try:
        eval_ds = load_dataset("jxm/nq_corpus_dpr", split="dev")
    except Exception:
        print("Fallback: Using train split as dev is missing")
        eval_ds = load_dataset("jxm/nq_corpus_dpr", split="train")
        
    eval_texts = eval_ds.select(range(min(AB_EVAL_SIZE, len(eval_ds))))["text"]
    
    # 4. Compute Raw Embeddings (Once)
    print("[*] Computing Raw Embeddings (Source)...")
    s_emb = embed_sentences(eval_texts, s_tok, s_mod, DEVICE)
    print("[*] Computing Raw Embeddings (Target)...")
    t_emb = embed_sentences(eval_texts, t_tok, t_mod, DEVICE)
    
    results = []

    # 5. Evaluate Src -> Tgt
    if chk_s2t:
        # Input: Source Emb, Target: Target Emb, Mapper: S2T
        metrics_s2t = evaluate_linear_projection(s_emb, t_emb, chk_s2t, DEVICE, "src2tgt")
        results.append(metrics_s2t)

    # 6. Evaluate Tgt -> Src
    if chk_t2s:
        # Input: Target Emb, Target: Source Emb, Mapper: T2S
        metrics_t2s = evaluate_linear_projection(t_emb, s_emb, chk_t2s, DEVICE, "tgt2src")
        results.append(metrics_t2s)
    
    # 7. Save & Print
    if results:
        res_df = pd.DataFrame(results)
        print("\n=== FINAL EVALUATION RESULTS (SOTA - Wu & Dredze) ===")
        print(res_df.to_string())
        
        csv_path = os.path.join(MAPPERS_DIR, f"sota_eval_results_{clean_src}_{clean_tgt}.csv")
        res_df.to_csv(csv_path, index=False)
        print(f"\n[+] Results saved to {csv_path}")
    else:
        print("[!] No results to display.")

if __name__ == "__main__":
    main()