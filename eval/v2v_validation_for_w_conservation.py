import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from tqdm import tqdm
import torch.nn as nn
import csv
from datetime import datetime

# -------------------
# Mapper class (must match training)
# -------------------
class TransformerEmbeddingMapper(nn.Module):
    def __init__(self, emb_dim, hidden_dim=512, n_layers=3, n_heads=8, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(emb_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim*4, dropout=dropout,
            activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(hidden_dim, emb_dim)
        self.ln = nn.LayerNorm(emb_dim)

    def forward(self, x):
        h = self.input_proj(x.unsqueeze(1))   # (B, 1, hidden_dim)
        h = self.transformer(h)[:, 0, :]      # (B, hidden_dim)
        mapped = self.ln(self.output_proj(h)) # (B, D)
        return mapped, h

# -------------------
# Load embedding models
# -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
src_name = "Qwen/Qwen3-Embedding-0.6B"   # source model
tgt_name = "thenlper/gte-large"          # target model

src_tok = AutoTokenizer.from_pretrained(src_name)
src_model = AutoModel.from_pretrained(src_name).to(device).eval()

tgt_tok = AutoTokenizer.from_pretrained(tgt_name)
tgt_model = AutoModel.from_pretrained(tgt_name).to(device).eval()

# helper: get safe max_length from model config
def model_max_length_or_default(tok, model, default=512):
    # prefer model.config.max_position_embeddings if present, else tokenizer/model config
    max_pos = getattr(model.config, "max_position_embeddings", None)
    if max_pos is not None:
        return int(max_pos)
    # fallback to tokenizer.model_max_length if available and reasonable
    tok_len = getattr(tok, "model_max_length", None)
    if tok_len is not None and tok_len > 0 and tok_len < 10_000:
        return int(tok_len)
    return default

# -------------------
# Robust embed function with batching + attention-weighted mean pooling
# -------------------
def embed_sentences(texts, tokenizer, model, batch_size=64, device=device):
    """
    texts: list[str]
    returns: torch.FloatTensor shape (len(texts), D) on CPU
    """
    max_len = model_max_length_or_default(tokenizer, model, default=512)
    embeddings = []
    model = model.to(device)
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        # ensure truncation to model max length to avoid position-emb mismatch
        inputs = tokenizer(batch_texts,
                           return_tensors="pt",
                           truncation=True,
                           max_length=max_len,
                           padding=True)  # pad to longest in batch
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            outs = model(input_ids=input_ids, attention_mask=attention_mask)
            # prefer last_hidden_state; some models use 'last_hidden_state'
            hidden = outs.last_hidden_state  # (B, L, D)

        # attention-weighted mean
        mask = attention_mask.unsqueeze(-1)  # (B, L, 1)
        hidden_masked = hidden * mask
        summed = hidden_masked.sum(dim=1)  # (B, D)
        lengths = mask.sum(dim=1).clamp(min=1)  # (B, 1)
        mean_pooled = summed / lengths
        embeddings.append(mean_pooled.cpu())

    embeddings = torch.cat(embeddings, dim=0)
    return embeddings  # CPU tensor

# -------------------
# Generic evaluation function (uses batched embeddings)
# -------------------
def evaluate_mapper_bidirectional(mapper, src_model, src_tok, tgt_model, tgt_tok, texts, 
                                  src_model_name, tgt_model_name, embed_batch_size=64):
    """
    Evaluate one mapping direction: src_model → tgt_model
    """
    direction = f"{src_model_name} → {tgt_model_name}"
    
    # embed all texts in both spaces (batched)
    print(f"Encoding source ({src_model_name}): {len(texts)} texts ...")
    src_embs = embed_sentences(texts, src_tok, src_model, batch_size=embed_batch_size)  # CPU
    print(f"Encoding target ({tgt_model_name}): {len(texts)} texts ...")
    tgt_embs = embed_sentences(texts, tgt_tok, tgt_model, batch_size=embed_batch_size)  # CPU

    # Move to device for mapping & similarity
    mapper = mapper.to(device).eval()
    src_embs_dev = src_embs.to(device)
    with torch.no_grad():
        mapped, _ = mapper(src_embs_dev)
        mapped = F.normalize(mapped, dim=-1).cpu()
    tgt_embs = F.normalize(tgt_embs, dim=-1)

    # Similarity matrix (careful with memory; compute in blocks if very large)
    N = mapped.size(0)
    # If N is large, compute top1 with block-wise matmul
    top1_correct = 0
    ranks = []
    cos_sims = []

    # We'll compute similarity in chunks to avoid O(N^2) memory in one go
    block = 4096  # adjust depending on RAM (CPU)
    # Pre-move target to cpu (already cpu)
    for i0 in range(0, N, block):
        i1 = min(N, i0 + block)
        sub_m = mapped[i0:i1]  # (b, D) CPU
        sim_block = sub_m @ tgt_embs.T  # (b, N)
        for local_idx in range(sim_block.size(0)):
            global_idx = i0 + local_idx
            row = sim_block[local_idx]
            values, indices = torch.sort(row, descending=True)
            # rank is 1-based
            rank = (indices == global_idx).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(rank)
            cos_sims.append(row[global_idx].item())
            if rank == 1:
                top1_correct += 1

    return {
        "direction": direction,
        "source_model": src_model_name,
        "target_model": tgt_model_name,
        "mean_cosine": sum(cos_sims) / N,
        "top1_acc": top1_correct / N,
        "mean_rank": sum(ranks) / N,
        "checkpoint_path": None,  # Will be filled in later
    }

# -------------------
# Main (entry)
# -------------------
if __name__ == "__main__":
    # Load Natural Questions dev subset (streaming)
    dset = load_dataset("jxm/nq_corpus_dpr", split="dev", streaming=True)
    texts = [x["text"] for x in dset.take(65536)]

    # Detect embedding dim by encoding a short string
    D = embed_sentences(["hello world"], src_tok, src_model, batch_size=1).shape[-1]

    # Load mapper checkpoint (adjust path)
    checkpoint_path = "/home/pallucchini/NEW_ALIGNMENT_METHOD/align_TLM_and_NT/model_aligned_qwen_gte/tlm_alignment_six_losses_015fil_111111.pt"
    tlm_ckpt = torch.load(checkpoint_path, map_location="cpu")

    def init_mapper(state_dict):
        m = TransformerEmbeddingMapper(D, hidden_dim=1024, n_layers=5, n_heads=16, dropout=0.1)
        m.load_state_dict(state_dict)
        return m.to(device).eval()

    mapper_src2tgt = init_mapper(tlm_ckpt["qwen32gte"])
    mapper_tgt2src = init_mapper(tlm_ckpt["gte2qwen3"])

    print("\n=== Evaluating TLM-trained mappers ===")
    res_src2tgt = evaluate_mapper_bidirectional(
        mapper_src2tgt, src_model, src_tok, tgt_model, tgt_tok, texts,
        src_model_name=src_name, tgt_model_name=tgt_name, embed_batch_size=64
    )
    res_src2tgt["checkpoint_path"] = checkpoint_path
    
    res_tgt2src = evaluate_mapper_bidirectional(
        mapper_tgt2src, tgt_model, tgt_tok, src_model, src_tok, texts,
        src_model_name=tgt_name, tgt_model_name=src_name, embed_batch_size=64
    )
    res_tgt2src["checkpoint_path"] = checkpoint_path
    
    print(res_src2tgt)
    print(res_tgt2src)

    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"embedding_mapper_evaluation_{timestamp}.csv"
    
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['direction', 'source_model', 'target_model', 'mean_cosine', 'top1_acc', 'mean_rank', 'checkpoint_path']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerow(res_src2tgt)
        writer.writerow(res_tgt2src)
    
    print(f"\n✓ Results saved to {csv_filename}")