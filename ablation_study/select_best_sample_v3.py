import os
import random
import sqlite3
import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

# optional SentenceTransformer backend
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# -------------------------
# Configurazione
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SRC_NAME = "Qwen/Qwen3-Embedding-0.6B"#"ibm-granite/granite-embedding-278m-multilingual"#"sentence-transformers/gtr-t5-base"# #"intfloat/e5-base-v2" #"infgrad/stella-base-en-v2" #"thenlper/gte-base" ## 
TGT_NAME = "thenlper/gte-large"#"thenlper/gte-base"#"intfloat/e5-base-v2" #"sentence-transformers/gtr-t5-base"#"infgrad/stella-base-en-v2"# # #"ibm-granite/granite-embedding-278m-multilingual"# #"ibm-granite/granite-embedding-278m-multilingual"# # 

RESERVOIR_DB = "nq_reservoir.db"
OUTPUT_FILE = "best_alignment_sample_10percent_spearman_qwen_gte_large_300k.pt"

SUBSAMPLE_MAX = 1_000_000
TARGET_RATIO = 0.30
FILTER_K_NEIGHBORS = 200  
N_BUCKETS = 300_000
CHUNK_TOKENS = 64           

# -------------------------
# Helper: unified loader (from second code)
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

# -------------------------
# Chunking (EXACT copy from second code)
# -------------------------
def sentence_split_into_64_token_chunks(text: str, tokenizer, chunk_tokens=CHUNK_TOKENS):
    """Tokenize text into non-overlapping chunks of exactly chunk_tokens tokens and decode them."""
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
    """
    Build reservoir of chunked sequences into sqlite up to sample_size.
    EXACT copy from second code.
    """
    ds = load_dataset(dataset_name, split=split, streaming=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # Delete and recreate table just in case of failed prior run
    cur.execute("DROP TABLE IF EXISTS sentences")
    cur.execute("CREATE TABLE sentences (id INTEGER PRIMARY KEY, text TEXT)")
    conn.commit()

    inserted = 0

    print(f"[*] Streaming dataset {dataset_name}:{split} to collect {sample_size} chunks and build DB...")
    for ex in tqdm(ds, desc="collecting chunks"):
        if inserted >= sample_size:
            break
            
        text = ex.get("text") or ex.get("document_text") or (ex.get("question", "") + " " + ex.get("answer", ""))
        if not text:
            continue
        
        chunks = sentence_split_into_64_token_chunks(text.replace("\n", " "), tokenizer_for_chunking, CHUNK_TOKENS)
        
        for ch in chunks:
            if inserted < sample_size:
                cur.execute("INSERT INTO sentences (id, text) VALUES (?, ?)", (inserted + 1, ch))
                inserted += 1
            if inserted % 10000 == 0:
                conn.commit()
            if inserted >= sample_size:
                break
                
    conn.commit()
    cur.execute("SELECT COUNT(*) FROM sentences")
    cnt = cur.fetchone()[0]
    conn.close()
    print(f"[+] Reservoir finished. Stored {cnt} samples in DB.")
    return cnt


def stream_reservoir_texts(db_path: str, subsample_size: int, chunk_size: int = 50000):
    """Load texts from existing reservoir DB. EXACT copy from second code."""
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
    
    print(f"[*] Loading {load_count} texts from reservoir DB (out of {total} available)...")
    
    while offset < load_count:
        limit = min(chunk_size, load_count - offset)
        cur.execute("SELECT text FROM sentences LIMIT ? OFFSET ?", (limit, offset))
        rows = cur.fetchall()
        if not rows:
            break
        all_sents.extend([r[0] for r in rows])
        offset += len(rows)
    conn.close()
    return all_sents


def load_texts_for_training(db_path: str, subsample_size: int, tokenizer_for_chunking):
    """
    Combined function to either load from existing DB or create one.
    EXACT copy from second code.
    """
    if os.path.exists(db_path):
        print(f"[+] Reservoir DB {db_path} found. Attempting to load texts.")
        texts = stream_reservoir_texts(db_path, subsample_size)
        
        if len(texts) == 0:
            print("Warning: DB is empty. Proceeding to create a new reservoir.")
            os.remove(db_path) # Remove empty/corrupt DB
            return load_texts_for_training(db_path, subsample_size, tokenizer_for_chunking) # Recursive call to create
        
        if len(texts) < subsample_size:
            print(f"Warning: DB only contained {len(texts)} texts. Using all available.")

        return texts
    else:
        print(f"[-] Reservoir DB {db_path} not found. Creating a new reservoir...")
        # Create the DB by sampling from the dataset
        build_reservoir_sqlite(db_path, sample_size=subsample_size, tokenizer_for_chunking=tokenizer_for_chunking)
        
        # Load the newly created texts
        texts = stream_reservoir_texts(db_path, subsample_size)
        return texts

# -------------------------
# Utility Embedding (from second code)
# -------------------------
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
# Logica di Correlazione
# -------------------------
def compute_spearman_scores(src_embs, tgt_embs, k=200):
    N = src_embs.shape[0]
    scores = np.zeros(N)
    
    # Normalizzazione per dot product (più veloce di cosine_similarity)
    # NOTE: These inputs must be numpy arrays for .dot() to broadcast correctly
    s_norm = src_embs / (np.linalg.norm(src_embs, axis=1, keepdims=True) + 1e-12)
    t_norm = tgt_embs / (np.linalg.norm(tgt_embs, axis=1, keepdims=True) + 1e-12)

    for i in tqdm(range(N), desc="Calcolo Spearman Similarity"):
        idx_neighbors = np.random.choice(N, k, replace=False)
        
        # Numpy .dot handles vector-matrix multiplication properly
        dists_src = 1 - s_norm[i].dot(s_norm[idx_neighbors].T)
        dists_tgt = 1 - t_norm[i].dot(t_norm[idx_neighbors].T)
        
        corr, _ = spearmanr(dists_src, dists_tgt)
        scores[i] = np.nan_to_num(corr)
    return scores

def main():
    # Load embedding backends first (needed for chunking tokenizer)
    print("[*] Caricamento embedding backends...")
    src_tok, src_model, src_backend = load_embedding_backend(SRC_NAME)
    tgt_tok, tgt_model, tgt_backend = load_embedding_backend(TGT_NAME)
    
    print(f"  Source model: {SRC_NAME} (backend: {src_backend})")
    print(f"  Target model: {TGT_NAME} (backend: {tgt_backend})")

    # For chunking we need a tokenizer. Prefer the src tokenizer if HF, else fallback to tgt HF tokenizer.
    if src_tok is None:
        if isinstance(tgt_tok, type(None)):
            from transformers import AutoTokenizer as _AutoTokenizer
            chunk_tok = _AutoTokenizer.from_pretrained(SRC_NAME)
        else:
            chunk_tok = tgt_tok
    else:
        chunk_tok = src_tok

    # 1. Load texts using the EXACT same workflow as second code
    texts = load_texts_for_training(RESERVOIR_DB, SUBSAMPLE_MAX, chunk_tok)
    
    if len(texts) == 0:
        raise RuntimeError("No training sequences were loaded from DB or generated. Check dataset/file permissions.")

    print(f"[+] Caricati {len(texts)} testi dal database")

    # 2. Generazione Embedding (using unified function from second code)
    print("[*] Embedding con source model...")
    # FIXED: Added .numpy() to convert Tensor to NumPy array
    src_embs = embed_sentences(texts, src_tok, src_model, DEVICE, batch_size=128, max_length=512).numpy()
    
    print("[*] Embedding con target model...")
    # FIXED: Added .numpy() to convert Tensor to NumPy array
    tgt_embs = embed_sentences(texts, tgt_tok, tgt_model, DEVICE, batch_size=128, max_length=512).numpy()

    # 3. Calcolo Correlazione Spearman
    corr_scores = compute_spearman_scores(src_embs, tgt_embs, k=FILTER_K_NEIGHBORS)

    # 4. Stratified Diversity Selection
    print(f"[*] Avvio campionamento diversificato (Target: {TARGET_RATIO*100}%)...")
    
    # Opzionale: PCA per velocizzare drasticamente il clustering senza perdere struttura macroscopica
    print("[*] Riduzione dimensionalità per clustering veloce...")
    pca = IncrementalPCA(n_components=32)
    low_dim_embs = pca.fit_transform(src_embs)

    # Clustering leggero per dividere lo spazio in "bucket"
    print(f"[*] Partizionamento spazio in {N_BUCKETS} bucket...")
    kmeans = MiniBatchKMeans(n_clusters=N_BUCKETS, batch_size=2048, random_state=SEED, n_init=3)
    labels = kmeans.fit_predict(low_dim_embs)

    target_n = int(len(texts) * TARGET_RATIO)
    samples_per_bucket = target_n // N_BUCKETS
    final_indices = []

    for i in tqdm(range(N_BUCKETS), desc="Selezione top-quality per bucket"):
        # Indici dei punti che appartengono a questo cluster
        bucket_indices = np.where(labels == i)[0]
        
        if len(bucket_indices) == 0:
            continue
            
        # Ordina i punti nel bucket per score di Spearman (dal migliore al peggiore)
        bucket_scores = corr_scores[bucket_indices]
        sorted_indices = bucket_indices[np.argsort(-bucket_scores)]
        
        # Prendi i migliori N da questa specifica regione dello spazio
        selection = sorted_indices[:max(1, samples_per_bucket)]
        final_indices.extend(selection)

    # Riempimento se mancano campioni causa arrotondamenti
    if len(final_indices) < target_n:
        diff = target_n - len(final_indices)
        already_selected = set(final_indices)
        best_overall = np.argsort(-corr_scores)
        
        added = 0
        for idx in best_overall:
            if idx not in already_selected:
                final_indices.append(idx)
                added += 1
            if added >= diff:
                break

    final_indices = np.array(final_indices[:target_n])

    # 5. Salvataggio
    print(f"[*] Salvataggio di {len(final_indices)} campioni in {OUTPUT_FILE}...")
    
    output_data = {
        "texts": [texts[i] for i in final_indices],
        "src_embeddings": torch.from_numpy(src_embs[final_indices]),
        "tgt_embeddings": torch.from_numpy(tgt_embs[final_indices]),
        "spearman_scores": torch.from_numpy(corr_scores[final_indices]),
        "metadata": {
            "src_model": SRC_NAME,
            "tgt_model": TGT_NAME,
            "src_backend": src_backend,
            "tgt_backend": tgt_backend,
            "selection_method": "Stratified-Spearman-Bucketing",
            "chunk_tokens": CHUNK_TOKENS,
            "max_length": 512,
            "seed": SEED
        }
    }
    
    torch.save(output_data, OUTPUT_FILE)
    avg_corr = corr_scores[final_indices].mean()
    print(f"[SUCCESS] Completato. Correlazione media del campione: {avg_corr:.4f}")

if __name__ == "__main__":
    main()