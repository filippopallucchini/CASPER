import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from scipy.stats import spearmanr
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from sentence_transformers import SentenceTransformer

# ============================
#   Model Loading & Embedding
# ============================

def load_any_embedding_model(model_name):
    if model_name.startswith("sentence-transformers/"):
        print(f"[Loader] Using SentenceTransformer backend for: {model_name}")
        model = SentenceTransformer(model_name)
        # SentenceTransformer objects store the HF tokenizer in the .tokenizer attribute
        tok = model.tokenizer 
        return tok, model, "sentencetransformer"
    print(f"[Loader] Using HuggingFace AutoModel backend for: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name)
    return tok, mdl, False

@torch.no_grad()
def embed_sentences_any(texts, tokenizer, model, is_st, device, batch_size=64):
    if is_st:
        model = model.to(device)
        e = model.encode(texts, convert_to_tensor=True, device=device, show_progress_bar=True)
        return e.cpu()
    
    model = model.to(device).eval()
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        out = model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1)
        hidden = out.last_hidden_state * mask
        pooled = hidden.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        embeddings.append(pooled.cpu())
    return torch.cat(embeddings, dim=0)

@torch.no_grad()
def compute_structure_preservation(original_embs, mapped_embs, sample_size=1000):
    N = original_embs.size(0)
    if N > sample_size:
        indices = torch.randperm(N)[:sample_size]
        original_embs = original_embs[indices]
        mapped_embs = mapped_embs[indices]
    
    orig_norm = F.normalize(original_embs.float(), dim=1)
    map_norm = F.normalize(mapped_embs.float(), dim=1)
    orig_sim = orig_norm @ orig_norm.T
    map_sim = map_norm @ map_norm.T
    
    triu = torch.triu_indices(orig_sim.size(0), orig_sim.size(1), offset=1)
    orig_d = (1 - orig_sim[triu[0], triu[1]]).cpu().numpy()
    map_d = (1 - map_sim[triu[0], triu[1]]).cpu().numpy()
    corr, _ = spearmanr(orig_d, map_d)
    return float(corr if not np.isnan(corr) else 0.0)

# ============================
#   UPDATED Mapper Definition
# ============================

class TransformerEmbeddingMapper(nn.Module):
    """
    Updated to match the Cross-Attention architecture from your training code.
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

        h = self.input_proj(x)
        
        if memory.dim() == 2:
            mem_proj = self.memory_proj(memory.unsqueeze(1))
        else:
            mem_proj = self.memory_proj(memory)

        # Cross-Attend: tgt=h (Query), memory=mem_proj (Key/Value)
        h = self.transformer(tgt=h, memory=mem_proj) 
        out = self.output_proj(h)
        out = self.ln(out)
        return (out[:, 0, :], h) if single else (out, h)

# ============================
#   Evaluation logic
# ============================

@torch.no_grad()
def evaluate_alignment(mapper, src_embs, tgt_embs, direction="src→tgt", device="cuda", recall_ks=[1, 5, 10, 50]):
    """
    Directional evaluation using cross-attention.
    """
    mapper.to(device).eval()
    src_embs_dev = src_embs.to(device)
    tgt_embs_dev = tgt_embs.to(device)

    if direction == "src→tgt":
        # Query = Source, Memory = Target
        mapped, _ = mapper(src_embs_dev, memory=tgt_embs_dev)
        mapped = F.normalize(mapped, dim=-1).cpu()
        reference = F.normalize(tgt_embs, dim=-1)
        original = src_embs
    else:
        # Query = Target, Memory = Source
        mapped, _ = mapper(tgt_embs_dev, memory=src_embs_dev)
        mapped = F.normalize(mapped, dim=-1).cpu()
        reference = F.normalize(src_embs, dim=-1)
        original = tgt_embs

    # Calculate Retrieval Metrics
    sim = mapped @ reference.T
    N = sim.size(0)
    diag = torch.arange(N)
    
    # Efficient rank calculation
    ranks = []
    for i in range(N):
        row = sim[i]
        rank = (row > row[i]).sum().item() + 1
        ranks.append(rank)
    
    ranks = torch.tensor(ranks).float()
    top1_acc = (ranks == 1).float().mean().item()
    mean_rank = ranks.mean().item()
    mean_cosine = sim.diag().mean().item()
    recall_at_k = {f"R@{k}": (ranks <= k).float().mean().item() for k in recall_ks}
    struct_corr = compute_structure_preservation(original, mapped)

    return {
        "direction": direction,
        "mean_cosine": mean_cosine,
        "top1_acc": top1_acc,
        "mean_rank": mean_rank,
        "structure_preservation": struct_corr,
        **recall_at_k
    }

# ============================
#   MAIN
# ============================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Config
    src_name = "sentence-transformers/gtr-t5-base"
    tgt_name = "thenlper/gte-base"
    path_mapper = "/home/pallucchini/NEW_ALIGNMENT_METHOD/align_new/ablation_study/saved_mappers_cross_attention_no_filter/ablation_1M_gtr_gte/nq_sentence-transformers_gtr-t5-base_thenlper_gte-base" # Update to your specific timestamp folder
    
    # Load Models
    src_tok, src_model, src_is_st = load_any_embedding_model(src_name)
    tgt_tok, tgt_model, tgt_is_st = load_any_embedding_model(tgt_name)

    # Extract embedding dimension
    if tgt_is_st:
        D = tgt_model.get_sentence_embedding_dimension()
    else:
        D = tgt_model.config.hidden_size


    # Load Cross-Attention Mappers
    # Note: Training used hidden_dim=D, n_layers=4, n_heads=16
    mapper_src2tgt = TransformerEmbeddingMapper(D, hidden_dim=D, n_layers=4, n_heads=16).to(device)
    mapper_src2tgt.load_state_dict(torch.load(os.path.join(path_mapper, "mapper_src2tgt.pt"), map_location=device))
    
    mapper_tgt2src = TransformerEmbeddingMapper(D, hidden_dim=D, n_layers=4, n_heads=16).to(device)
    mapper_tgt2src.load_state_dict(torch.load(os.path.join(path_mapper, "mapper_tgt2src.pt"), map_location=device))

    # Load Data
    print("Loading NQ dev dataset...")
    eval_dset = load_dataset("jxm/nq_corpus_dpr", split="dev", streaming=True)
    texts = [x["text"] for x in eval_dset.take(65536)] # Using 2k for speed, increase if needed

    print("Embedding texts...")
    src_embs = embed_sentences_any(texts, src_tok, src_model, src_is_st, device)
    tgt_embs = embed_sentences_any(texts, tgt_tok, tgt_model, tgt_is_st, device)

    # Evaluate
    results = []
    print("Evaluating src -> tgt...")
    results.append(evaluate_alignment(mapper_src2tgt, src_embs, tgt_embs, "src→tgt", device))
    print("Evaluating tgt -> src...")
    results.append(evaluate_alignment(mapper_tgt2src, src_embs, tgt_embs, "tgt→src", device))

    df = pd.DataFrame(results)
    print("\n=== Final Alignment Results ===")
    print(df.to_string())