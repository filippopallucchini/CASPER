#!/usr/bin/env python3
"""
Optimized BEIR evaluation using Guided Alignment (Parallel Query Projection).
Architecture matches the Official Training Validation exactly.

Logic:
Since we have the query text, we generate both Source and Target embeddings.
We map Source -> Target using the Target Embedding as MEMORY.
This evaluates the alignment quality specifically.
"""

import os
import json
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from tqdm import tqdm
import pandas as pd

# ----------------------------
# Defaults / Config
# ----------------------------
DEFAULT_SRC_MODEL = "ibm-granite/granite-embedding-278m-multilingual"
DEFAULT_TGT_MODEL = "thenlper/gte-base"

# Paths to your saved mappers
SRC2TGT_PATH = "/home/pallucchini/NEW_ALIGNMENT_METHOD/align_new/saved_mappers_official_gra_gte_1M_MacroStr_SimTlm/official_mapper_src2tgt.pt"
TGT2SRC_PATH = "/home/pallucchini/NEW_ALIGNMENT_METHOD/align_new/saved_mappers_official_gra_gte_1M_MacroStr_SimTlm/official_mapper_tgt2src.pt"

DEFAULT_DATA_PATH = "/home/pallucchini/LLM_alignment/align_encoder/beir_datasets"
DEFAULT_OUTPUT = "./alignment_validation_lowmem"
DEFAULT_EMB_BATCH = 128    
DEFAULT_MAP_BATCH = 1024   

# ----------------------------
# 1. OFFICIAL MAPPER DEFINITION
# ----------------------------
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

# ----------------------------
# 2. Utilities
# ----------------------------
def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def sample_corpus_for_dataset(corpus: dict, dataset_name: str, cache_dir: str, debug: bool = False) -> Tuple[List[str], List[str]]:
    ensure_dir(cache_dir)
    cache_file = os.path.join(cache_dir, f"{dataset_name}_sample.json")
    if os.path.exists(cache_file):
        with open(cache_file, "r") as fh:
            obj = json.load(fh)
        return obj["ids"], obj["texts"]

    docids = list(corpus.keys())
    n = len(docids)
    if debug:
        sample_size = min(5000, n)
    else:
        sample_size = n 

    sampled = docids if sample_size >= n else random.sample(docids, sample_size)
    texts = [corpus[d].get("text", "") for d in sampled]

    with open(cache_file, "w") as fh:
        json.dump({"ids": sampled, "texts": texts}, fh)

    return sampled, texts

# ----------------------------
# 3. Optimized Embedding (CPU/Disk Caching)
# ----------------------------
@torch.inference_mode()
def embed_texts_cpu_optimized(
    model: AutoModel, 
    tokenizer: AutoTokenizer, 
    texts: List[str], 
    batch_size: int = 128,
    show_progress: bool = True,
    cache_file: Optional[str] = None
) -> np.ndarray:
    if cache_file and os.path.exists(cache_file):
        return np.load(cache_file, mmap_mode='r')
    
    device = next(model.parameters()).device
    all_embs = []
    
    hidden_size = model.config.hidden_size
    n_batches = (len(texts) + batch_size - 1) // batch_size
    
    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, total=n_batches, desc="Embedding")
    
    for i in iterator:
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc, return_dict=True)
        mask = enc["attention_mask"].unsqueeze(-1)
        hidden = out.last_hidden_state * mask
        pooled = hidden.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        all_embs.append(pooled.cpu().numpy().astype(np.float32))
        del out, hidden, pooled, enc, mask
        
    if len(all_embs) == 0:
        result = np.zeros((0, hidden_size), dtype=np.float32)
    else:
        result = np.vstack(all_embs)
    
    if cache_file:
        ensure_dir(os.path.dirname(cache_file))
        np.save(cache_file, result)
    
    return result

# ----------------------------
# 4. Optimized Mapping WITH MEMORY (Corrected)
# ----------------------------
@torch.inference_mode()
def map_embeddings_with_memory(
    mapper: torch.nn.Module,
    input_np: np.ndarray,
    memory_np: np.ndarray,
    batch_size: int,
    device: str = "cuda",
    normalize: bool = True
) -> np.ndarray:
    """
    Projects embeddings using Cross-Attention.
    input_np: The embeddings to transform (e.g., Granite)
    memory_np: The reference embeddings to guide projection (e.g., GTE)
    """
    if input_np.shape[0] == 0:
        return input_np
    
    assert input_np.shape[0] == memory_np.shape[0], "Input and Memory must have same number of samples!"
    
    target_device = torch.device(device if device != "cpu" and torch.cuda.is_available() else "cpu")
    mapper = mapper.to(target_device)
    mapper.eval()
    
    all_out = []
    
    for i in range(0, input_np.shape[0], batch_size):
        # Load blocks
        block_in = torch.from_numpy(input_np[i:i+batch_size]).to(target_device, non_blocking=True)
        block_mem = torch.from_numpy(memory_np[i:i+batch_size]).to(target_device, non_blocking=True)
        
        # Cross-Attention Mapping
        out, _ = mapper(block_in, memory=block_mem)
        
        if normalize:
            out = F.normalize(out, dim=-1)
        
        all_out.append(out.cpu().numpy())
        del block_in, block_mem, out
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return np.vstack(all_out).astype(np.float32)

# ----------------------------
# 5. Optimized Retrieval
# ----------------------------
def retrieve_topk_optimized(
    query_embs: np.ndarray, 
    corpus_embs: np.ndarray, 
    query_ids: List[str], 
    corpus_ids: List[str],
    top_k: int = 1000, 
    chunk_size: int = 32768 
) -> dict:
    
    nq, d = query_embs.shape
    nd = corpus_embs.shape[0]
    
    eps = 1e-12
    qnorm = np.linalg.norm(query_embs, axis=1, keepdims=True)
    query_embs = query_embs / np.maximum(qnorm, eps)
    
    cnorm = np.linalg.norm(corpus_embs, axis=1, keepdims=True)
    corpus_embs = corpus_embs / np.maximum(cnorm, eps)
    
    results = {}
    
    all_scores = np.full((nq, top_k), -np.inf, dtype=np.float32)
    all_indices = np.full((nq, top_k), -1, dtype=np.int32)
    
    for cs in tqdm(range(0, nd, chunk_size), desc="Retrieval", disable=nq < 100):
        ce = min(cs + chunk_size, nd)
        corpus_chunk = corpus_embs[cs:ce]
        
        sims = query_embs @ corpus_chunk.T
        
        for qi in range(nq):
            current_scores = sims[qi]
            
            combined_scores = np.concatenate([all_scores[qi], current_scores])
            combined_indices = np.concatenate([all_indices[qi], np.arange(cs, ce, dtype=np.int32)])
            
            if combined_scores.size > top_k:
                top_idx = np.argpartition(-combined_scores, top_k-1)[:top_k]
                all_scores[qi] = combined_scores[top_idx]
                all_indices[qi] = combined_indices[top_idx]
        
        del sims, corpus_chunk
    
    for qi in range(nq):
        scores = all_scores[qi]
        indices = all_indices[qi]
        
        valid_mask = indices >= 0
        scores = scores[valid_mask]
        indices = indices[valid_mask]
        
        if len(scores) > 0:
            sort_idx = np.argsort(-scores)
            results[query_ids[qi]] = {
                corpus_ids[int(indices[i])]: float(scores[i])
                for i in sort_idx
            }
        else:
            results[query_ids[qi]] = {}
    
    return results

def compute_beir_metrics(qrels: dict, results: dict) -> dict:
    ev = EvaluateRetrieval(score_function="cos_sim")
    ndcg, _map, recall, precision = ev.evaluate(qrels, results, ev.k_values)
    return {
        "nDCG@10": ndcg.get("NDCG@10", 0.0),
        "Recall@10": recall.get("Recall@10", 0.0),
        "Recall@100": recall.get("Recall@100", 0.0),
    }

# ----------------------------
# 6. Main Evaluation Loop
# ----------------------------
def evaluate_dataset_optimized(
    dataset_name: str,
    src_model: AutoModel,
    src_tok: AutoTokenizer,
    tgt_model: AutoModel,
    tgt_tok: AutoTokenizer,
    mapper_s2t: torch.nn.Module,
    mapper_t2s: torch.nn.Module,
    data_path: str,
    cache_dir: str,
    embed_batch: int,
    map_batch: int,
    top_k: int,
    chunk_size_docs: int,
    device_for_mapping: str,
    debug: bool = False
):
    print(f"\n{'='*60}")
    print(f"Evaluating {dataset_name}")
    print(f"{'='*60}")
    
    dataset_path = os.path.join(data_path, dataset_name)
    corpus, queries, qrels = GenericDataLoader(dataset_path).load(split="test")

    sampled_ids, sampled_texts = sample_corpus_for_dataset(corpus, dataset_name, cache_dir, debug=debug)
    print(f"Corpus: {len(sampled_ids):,} docs")

    cache_subdir = os.path.join(cache_dir, dataset_name)
    ensure_dir(cache_subdir)
    
    # Sanitize model names for filenames (e.g., "ibm-granite/granite..." -> "ibm-granite_granite...")
    s_name = src_model.config._name_or_path.replace("/", "_")
    t_name = tgt_model.config._name_or_path.replace("/", "_")

    # Define cache names unique to the model
    src_corpus_cache = os.path.join(cache_subdir, f"corpus_emb_{s_name}.npy")
    tgt_corpus_cache = os.path.join(cache_subdir, f"corpus_emb_{t_name}.npy")
    src_query_cache = os.path.join(cache_subdir, f"query_emb_{s_name}.npy")
    tgt_query_cache = os.path.join(cache_subdir, f"query_emb_{t_name}.npy")
    # 1. Embed Corpus
    print("\n[1/4] Embedding corpus...")
    src_model.to("cpu"); tgt_model.to("cpu")
    src_c_emb = embed_texts_cpu_optimized(src_model, src_tok, sampled_texts, batch_size=embed_batch, cache_file=src_corpus_cache)
    tgt_c_emb = embed_texts_cpu_optimized(tgt_model, tgt_tok, sampled_texts, batch_size=embed_batch, cache_file=tgt_corpus_cache)

    # 2. Embed Queries (Source AND Target)
    print("\n[2/4] Embedding queries...")
    qids = list(queries.keys())
    qtexts = [queries[q] for q in qids]
    
    src_q_emb = embed_texts_cpu_optimized(src_model, src_tok, qtexts, batch_size=embed_batch, cache_file=src_query_cache, show_progress=False)
    tgt_q_emb = embed_texts_cpu_optimized(tgt_model, tgt_tok, qtexts, batch_size=embed_batch, cache_file=tgt_query_cache, show_progress=False)
    print(f"Queries: {len(qids):,}")

    # 3. Evaluate Systems
    mapper_s2t.eval().to("cpu")
    mapper_t2s.eval().to("cpu")
    rows = []

    print("\n[3/4] Evaluating systems...")
    
    # A. Source Baseline (Granite -> Granite Corpus)
    print("  → src_original...")
    res = retrieve_topk_optimized(src_q_emb, src_c_emb, qids, sampled_ids, top_k=top_k, chunk_size=chunk_size_docs)
    rows.append({"dataset": dataset_name, "system": "src_original", **compute_beir_metrics(qrels, res)})
    del res; gc.collect()

    # B. Target Baseline (GTE -> GTE Corpus)
    print("  → tgt_original...")
    res = retrieve_topk_optimized(tgt_q_emb, tgt_c_emb, qids, sampled_ids, top_k=top_k, chunk_size=chunk_size_docs)
    rows.append({"dataset": dataset_name, "system": "tgt_original", **compute_beir_metrics(qrels, res)})
    del res; gc.collect()

    # C. Mapped Retrieval (Granite Query + GTE Memory -> GTE Corpus)
    print("  → src2tgt_mapped (Granite Query guided by GTE Memory)...")
    mapped_src_q = map_embeddings_with_memory(
        mapper_s2t, 
        input_np=src_q_emb, 
        memory_np=tgt_q_emb,  # <--- GUIDED BY TARGET
        batch_size=map_batch, 
        device=device_for_mapping
    )
    res = retrieve_topk_optimized(mapped_src_q, tgt_c_emb, qids, sampled_ids, top_k=top_k, chunk_size=chunk_size_docs)
    rows.append({"dataset": dataset_name, "system": "src2tgt_mapped", **compute_beir_metrics(qrels, res)})
    del mapped_src_q, res; gc.collect()

    # D. Mapped Retrieval (GTE Query + Granite Memory -> Granite Corpus)
    print("  → tgt2src_mapped (GTE Query guided by Granite Memory)...")
    mapped_tgt_q = map_embeddings_with_memory(
        mapper_t2s, 
        input_np=tgt_q_emb, 
        memory_np=src_q_emb,  # <--- GUIDED BY SOURCE
        batch_size=map_batch, 
        device=device_for_mapping
    )
    res = retrieve_topk_optimized(mapped_tgt_q, src_c_emb, qids, sampled_ids, top_k=top_k, chunk_size=chunk_size_docs)
    rows.append({"dataset": dataset_name, "system": "tgt2src_mapped", **compute_beir_metrics(qrels, res)})
    del mapped_tgt_q, res; gc.collect()

    # Print Results
    print(f"\n[4/4] Results for {dataset_name}:")
    for row in rows:
        print(f"  {row['system']:20s} | nDCG@10: {row['nDCG@10']:.4f} | Recall@100: {row['Recall@100']:.4f}")

    del src_c_emb, tgt_c_emb, src_q_emb, tgt_q_emb
    gc.collect()
    
    return rows

# ----------------------------
# 7. Main CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Optimized BEIR evaluation")
    parser.add_argument("--src_model", type=str, default=DEFAULT_SRC_MODEL)
    parser.add_argument("--tgt_model", type=str, default=DEFAULT_TGT_MODEL)
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--embed_batch", type=int, default=DEFAULT_EMB_BATCH)
    parser.add_argument("--map_batch", type=int, default=DEFAULT_MAP_BATCH)
    parser.add_argument("--top_k", type=int, default=1000)
    parser.add_argument("--chunk_docs", type=int, default=32768)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clear_cache", action="store_true", help="Clear embedding cache")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ensure_dir(args.output_dir)
    
    if args.clear_cache:
        print("Clearing embedding cache...")
        import shutil
        for ds in os.listdir(args.data_path):
            cache_path = os.path.join(args.output_dir, ds)
            if os.path.exists(cache_path):
                shutil.rmtree(cache_path)

    print("="*60)
    print("OPTIMIZED BEIR EVALUATION (Guided Alignment)")
    print("="*60)
    print(f"Source model: {args.src_model}")
    print(f"Target model: {args.tgt_model}")
    print(f"Device: {args.device}")
    
    # 1. Load Models
    print("\nLoading models...")
    src_tok = AutoTokenizer.from_pretrained(args.src_model, use_fast=True)
    src_model = AutoModel.from_pretrained(args.src_model).to("cpu").eval()
    
    tgt_tok = AutoTokenizer.from_pretrained(args.tgt_model, use_fast=True)
    tgt_model = AutoModel.from_pretrained(args.tgt_model).to("cpu").eval()

    dim_s = src_model.config.hidden_size
    dim_t = tgt_model.config.hidden_size
    print(f"Dimensions detected -> Source: {dim_s}, Target: {dim_t}")

    # 2. Load Mappers
    print("Loading mappers...")
    mapper_s2t = TransformerEmbeddingMapper(dim_s, hidden_dim=dim_s, n_layers=4, n_heads=16)
    mapper_t2s = TransformerEmbeddingMapper(dim_s, hidden_dim=dim_s, n_layers=4, n_heads=16)

    if os.path.exists(os.path.expanduser(SRC2TGT_PATH)):
        mapper_s2t.load_state_dict(torch.load(SRC2TGT_PATH, map_location="cpu"))
        print(f"  ✓ Loaded src2tgt mapper")
    else:
        print(f"  ⚠ SRC2TGT checkpoint not found")

    if os.path.exists(os.path.expanduser(TGT2SRC_PATH)):
        mapper_t2s.load_state_dict(torch.load(TGT2SRC_PATH, map_location="cpu"))
        print(f"  ✓ Loaded tgt2src mapper")
    else:
        print(f"  ⚠ TGT2SRC checkpoint not found")

    # 3. Evaluate Datasets
    datasets = sorted([d for d in os.listdir(args.data_path) 
                      if os.path.isdir(os.path.join(args.data_path, d))])
    print(f"\nFound {len(datasets)} datasets: {', '.join(datasets)}")

    all_rows = []
    for i, ds in enumerate(datasets, 1):
        try:
            rows = evaluate_dataset_optimized(
                ds, src_model, src_tok, tgt_model, tgt_tok,
                mapper_s2t, mapper_t2s,
                data_path=args.data_path,
                cache_dir=args.output_dir,
                embed_batch=args.embed_batch,
                map_batch=args.map_batch,
                top_k=args.top_k,
                chunk_size_docs=args.chunk_docs,
                device_for_mapping=args.device,
                debug=args.debug
            )
            all_rows.extend(rows)
            df = pd.DataFrame(all_rows)
            df.to_csv(os.path.join(args.output_dir, "results_partial.csv"), index=False)
        except Exception as e:
            print(f"✗ SKIPPED {ds}: {e}")
            import traceback
            traceback.print_exc()

    # 4. Final Save
    df = pd.DataFrame(all_rows)
    output_file = os.path.join(args.output_dir, "results_full.csv")
    df.to_csv(output_file, index=False)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"Results saved to: {output_file}")
    if not df.empty:
        summary = df.groupby('system')[['nDCG@10', 'Recall@10', 'Recall@100']].mean()
        print(summary.to_string())

if __name__ == "__main__":
    main()