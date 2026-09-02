#!/usr/bin/env python3
"""
Functional Structure-Preservation tests for CASPER / CASPER-TC
==============================================================
Turns SP from a descriptive metric into a falsifiable functional property.
Two tests, run for EVERY mapper checkpoint found (missing ones are skipped):

  1) k-NN LABEL TRANSFER (TweetTopic only, it has topic labels).
     Leave-one-out k-NN topic classification computed in:
       (a) the native source space  z_s
       (b) the mapped space         phi(z_s)
       (c) the native target space  z_t   (reference ceiling)
     Prediction: if SP is functional, acc(mapped) ~ acc(source); the gap
     acc(source) - acc(mapped) quantifies how much task-relevant local
     geometry the mapping destroys.

  2) NEIGHBOURHOOD JACCARD@k (NQ dev + TweetTopic).
     Overlap of each point's top-k neighbour set between the source space
     and the mapped space, k in {5, 10, 50, 200}. This is the LOCAL
     complement of the global Spearman SP: high SP with destroyed local
     neighbourhoods would be exposed here.

  3) SP AS A PREDICTOR (printed summary).
     Across all evaluated (pair, variant, direction) rows, the Spearman
     correlation between measured SP and (i) k-NN transfer retention
     acc(mapped)/acc(source), (ii) Jaccard@10. High correlation = SP is a
     validated proxy of a deployable property, not a benchmark artefact.

Everything reuses the embedding cache of the OOD script
(eval_ood_vec2vec_protocol/emb_cache/), so with warm cache the whole run is
mapper-forwards + CPU matrix ops. NQ uses a subsample (default 8192) of the
held-out dev split, embedded once and cached.
"""

import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from datasets import load_dataset

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# -------------------------
# Config
# -------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

VEC2VEC_MODEL_IDS = {
    "e5":     "intfloat/e5-base-v2",
    "gra":    "ibm-granite/granite-embedding-278m-multilingual",
    "gte":    "thenlper/gte-base",
    "gtr":    "sentence-transformers/gtr-t5-base",
    "stella": "infgrad/stella-base-en-v2",
}
VARIANT_SUFFIX = {"casper": "unsupervised", "casper_tc": "linear"}

_PAIR_SPECS = [
    ("e5",  "stella", "casper"), ("gra", "e5", "casper"), ("gra", "gte", "casper"),
    ("gra", "stella", "casper"), ("gte", "e5", "casper"), ("gte", "stella", "casper"),
    ("gtr", "e5", "casper"), ("gtr", "gra", "casper"), ("gtr", "gte", "casper"),
    ("gtr", "stella", "casper"),
    ("e5",  "stella", "casper_tc"), ("gra", "e5", "casper_tc"), ("gra", "gte", "casper_tc"),
    ("gra", "stella", "casper_tc"), ("gte", "e5", "casper_tc"), ("gte", "stella", "casper_tc"),
    ("gtr", "e5", "casper_tc"), ("gtr", "gra", "casper_tc"), ("gtr", "gte", "casper_tc"),
    ("gtr", "stella", "casper_tc"),
]
PAIRS = [
    {"src": VEC2VEC_MODEL_IDS[s], "tgt": VEC2VEC_MODEL_IDS[t], "variant": v,
     "ckpt_dir": f"saved_mappers_official_{s}_{t}_1M_MacroStr_SimTlm_{VARIANT_SUFFIX[v]}"}
    for (s, t, v) in _PAIR_SPECS
]
# Optional extra checkpoints (e.g. mse+con ablation mappers) — same dir layout:
# PAIRS.append({"src": ..., "tgt": ..., "variant": "mse_con",
#               "ckpt_dir": "saved_mappers_ablation_..."})

# Datasets
TWEETTOPIC = ("cardiffnlp/tweet_topic_multilingual", "en", "test")   # 800, labelled
NQ_EVAL_SIZE = 8192          # subsample of jxm/nq_corpus_dpr dev split
KNN_K = 10                   # k for label transfer
JACCARD_KS = [5, 10, 50, 200]
SP_SAMPLE = 1000             # sample size for the Spearman SP metric

EMB_BATCH = 64
OUT_DIR   = "eval_functional_sp"
CACHE_DIR = os.path.join("eval_ood_vec2vec_protocol", "emb_cache")  # shared cache
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# -------------------------
# Mapper architectures (verbatim)
# -------------------------
class LinearInjectionMapper(nn.Module):
    def __init__(self, input_dim, output_dim, memory_dim,
                 hidden_dim=1024, n_layers=4, n_heads=16, dropout=0.1):
        super().__init__()
        self.input_proj  = nn.Linear(input_dim,  hidden_dim)
        self.memory_proj = nn.Linear(memory_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim * 4),
                          nn.GELU(), nn.Dropout(dropout),
                          nn.Linear(hidden_dim * 4, hidden_dim), nn.Dropout(dropout))
            for _ in range(n_layers)])
        self.final_ln    = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.out_ln      = nn.LayerNorm(output_dim)

    def forward(self, x, memory):
        if x.dim() == 3: x = x.squeeze(1)
        if memory.dim() == 3: memory = memory.squeeze(1)
        h = self.input_proj(x) + self.memory_proj(memory)
        for block in self.blocks:
            h = h + block(h)
        h = self.final_ln(h)
        return self.out_ln(self.output_proj(h)), h


class StandaloneMapper(nn.Module):
    def __init__(self, input_dim, output_dim, memory_dim=None,
                 hidden_dim=1024, n_layers=4, n_heads=16, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim * 4),
                          nn.GELU(), nn.Dropout(dropout),
                          nn.Linear(hidden_dim * 4, hidden_dim), nn.Dropout(dropout))
            for _ in range(n_layers)])
        self.final_ln    = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.out_ln      = nn.LayerNorm(output_dim)

    def forward(self, x, memory=None):
        if x.dim() == 3: x = x.squeeze(1)
        del memory
        h = self.input_proj(x)
        for block in self.blocks:
            h = h + block(h)
        h = self.final_ln(h)
        return self.out_ln(self.output_proj(h)), h


def build_mapper_from_state_dict(state):
    hidden_dim = state["input_proj.weight"].shape[0]
    input_dim  = state["input_proj.weight"].shape[1]
    output_dim = state["output_proj.weight"].shape[0]
    n_layers   = len({k.split(".")[1] for k in state if k.startswith("blocks.")})
    if "memory_proj.weight" in state:
        mapper = LinearInjectionMapper(input_dim, output_dim,
                                       state["memory_proj.weight"].shape[1],
                                       hidden_dim=hidden_dim, n_layers=n_layers)
        arch = "LinearInjectionMapper"
    else:
        mapper = StandaloneMapper(input_dim, output_dim,
                                  hidden_dim=hidden_dim, n_layers=n_layers)
        arch = "StandaloneMapper"
    mapper.load_state_dict(state)
    return mapper, arch


def load_pair_mappers(ckpt_dir):
    src_ckpt = os.path.join(ckpt_dir, "official_mapper_src2tgt.pt")
    tgt_ckpt = os.path.join(ckpt_dir, "official_mapper_tgt2src.pt")
    if not (os.path.exists(src_ckpt) and os.path.exists(tgt_ckpt)):
        raise FileNotFoundError(f"Missing checkpoints in '{ckpt_dir}'")
    m_s2t, arch = build_mapper_from_state_dict(torch.load(src_ckpt, map_location="cpu"))
    m_t2s, _    = build_mapper_from_state_dict(torch.load(tgt_ckpt, map_location="cpu"))
    return m_s2t.to(DEVICE).eval(), m_t2s.to(DEVICE).eval(), arch

# -------------------------
# Encoders + cached embeddings
# -------------------------
_ENCODER_CACHE = {}

def load_embedding_backend(name):
    if name.startswith("sentence-transformers/") and SentenceTransformer is not None:
        model = SentenceTransformer(name)
        return model.tokenizer, model
    tok = AutoTokenizer.from_pretrained(name)
    try:
        model = AutoModel.from_pretrained(name, trust_remote_code=True)
    except Exception:
        model = AutoModelForSequenceClassification.from_pretrained(name, trust_remote_code=True)
    return tok, model


def get_encoder(name):
    if name not in _ENCODER_CACHE:
        _ENCODER_CACHE[name] = load_embedding_backend(name)
    return _ENCODER_CACHE[name]


@torch.no_grad()
def embed_sentences(texts, tokenizer, model, device, batch_size=EMB_BATCH, max_length=512):
    embeddings = []
    is_st = SentenceTransformer is not None and isinstance(model, SentenceTransformer)
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i + batch_size]
        if is_st:
            emb = model.encode(batch, convert_to_tensor=True, device=device, show_progress_bar=False)
            embeddings.append(emb.cpu())
        else:
            model = model.to(device).eval()
            inputs = tokenizer(batch, return_tensors="pt", padding=True,
                               truncation=True, max_length=max_length).to(device)
            out = model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1)
            hidden = out.last_hidden_state * mask
            pooled = hidden.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            embeddings.append(pooled.cpu())
    return torch.cat(embeddings, dim=0).float()


def embed_with_cache(texts, dataset_key, model_name):
    safe = model_name.replace("/", "__")
    cache_path = os.path.join(CACHE_DIR, f"{dataset_key}__{safe}.pt")
    if os.path.exists(cache_path):
        emb = torch.load(cache_path, map_location="cpu").float()
        if emb.shape[0] == len(texts):
            return emb
    tok, mod = get_encoder(model_name)
    emb = embed_sentences(texts, tok, mod, DEVICE)
    torch.save(emb, cache_path)
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return emb

# -------------------------
# Datasets
# -------------------------
def load_tweettopic():
    name, config, split = TWEETTOPIC
    ds = load_dataset(name, config)[split]
    texts, labels = [], []
    # label column autodetect: prefer explicit names, else first list/int col
    cand = [c for c in ("gold_label_list", "label_name", "labels", "label")
            if c in ds.column_names]
    lab_col = cand[0] if cand else None
    for ex in ds:
        t = ex["text"]
        if not t or not t.strip():
            continue
        texts.append(t)
        if lab_col is None:
            labels.append(frozenset())
        else:
            v = ex[lab_col]
            labels.append(frozenset(v) if isinstance(v, (list, tuple)) else frozenset([v]))
    print(f"[+] TweetTopic: {len(texts)} texts, label column: {lab_col}")
    return texts, labels


def load_nq_texts():
    ds = load_dataset("jxm/nq_corpus_dpr", split="dev")
    n = min(NQ_EVAL_SIZE, len(ds))
    texts = [t for t in ds.select(range(n))["text"] if t and t.strip()]
    print(f"[+] NQ dev: {len(texts)} texts")
    return texts

# -------------------------
# Functional metrics
# -------------------------
@torch.no_grad()
def topk_neighbours(emb, k):
    """Indices of top-k cosine neighbours per row, self excluded. [N, k]"""
    e = F.normalize(emb.float(), dim=1)
    sim = e @ e.T
    sim.fill_diagonal_(-2.0)
    return sim.topk(k, dim=1).indices


@torch.no_grad()
def knn_label_accuracy(emb, labels, k=KNN_K):
    """
    Leave-one-out k-NN multi-label accuracy: predict the most frequent label
    among the k neighbours' label sets; correct if it belongs to the gold set.
    """
    nn_idx = topk_neighbours(emb, k)
    correct = 0
    for i in range(len(labels)):
        votes = {}
        for j in nn_idx[i].tolist():
            for lab in labels[j]:
                votes[lab] = votes.get(lab, 0) + 1
        if not votes:
            continue
        pred = max(votes, key=votes.get)
        if pred in labels[i]:
            correct += 1
    return correct / len(labels)


@torch.no_grad()
def jaccard_at_k(emb_a, emb_b, k):
    """Mean Jaccard overlap of top-k neighbour sets between two spaces."""
    na = topk_neighbours(emb_a, k)
    nb = topk_neighbours(emb_b, k)
    scores = []
    for i in range(na.size(0)):
        sa, sb = set(na[i].tolist()), set(nb[i].tolist())
        scores.append(len(sa & sb) / len(sa | sb))
    return float(np.mean(scores))


@torch.no_grad()
def sp_spearman(orig, mapped, sample_size=SP_SAMPLE):
    orig, mapped = orig.float(), mapped.float()
    N = orig.size(0)
    if N > sample_size:
        idx = torch.randperm(N)[:sample_size]
        orig, mapped = orig[idx], mapped[idx]
    on, mn = F.normalize(orig, dim=1), F.normalize(mapped, dim=1)
    os_, ms_ = on @ on.T, mn @ mn.T
    triu = torch.triu_indices(os_.size(0), os_.size(1), offset=1)
    od = (1 - os_[triu[0], triu[1]]).numpy()
    md = (1 - ms_[triu[0], triu[1]]).numpy()
    corr, _ = spearmanr(od, md)
    return float(corr)

# -------------------------
# Main
# -------------------------
def main():
    # 1) Datasets (shared across all pairs)
    tt_texts, tt_labels = load_tweettopic()
    nq_texts = load_nq_texts()
    datasets = {"tweettopic": tt_texts, "nq_dev": nq_texts}
    has_labels = {"tweettopic": tt_labels, "nq_dev": None}

    with open(os.path.join(OUT_DIR, "run_meta.json"), "w") as f:
        json.dump({"knn_k": KNN_K, "jaccard_ks": JACCARD_KS,
                   "nq_eval_size": NQ_EVAL_SIZE, "seed": SEED,
                   "started_at": datetime.now().isoformat(timespec="seconds")}, f, indent=2)

    rows = []
    knn_native_cache = {}   # (dataset, model) -> LOO kNN acc in native space

    for pi, pair in enumerate(PAIRS):
        src, tgt, variant, ckpt_dir = pair["src"], pair["tgt"], pair["variant"], pair["ckpt_dir"]
        pair_tag = f"{src.split('/')[-1]}__{tgt.split('/')[-1]}"
        print(f"\n[*] ===== {pi+1}/{len(PAIRS)}: {pair_tag} [{variant}] =====")
        try:
            m_s2t, m_t2s, arch = load_pair_mappers(ckpt_dir)
        except FileNotFoundError as e:
            print(f"  [!] {e} — skipping.")
            continue

        for ds_key, texts in datasets.items():
            s_emb = embed_with_cache(texts, ds_key, src)
            t_emb = embed_with_cache(texts, ds_key, tgt)
            labels = has_labels[ds_key]

            for direction, mapper, q_emb, mem_emb in [
                ("src2tgt", m_s2t, s_emb, t_emb),
                ("tgt2src", m_t2s, t_emb, s_emb),
            ]:
                with torch.no_grad():
                    mapped, _ = mapper(q_emb.to(DEVICE).float(),
                                       memory=mem_emb.to(DEVICE).float())
                mapped = mapped.cpu().float()

                rec = {"pair": pair_tag, "variant": variant, "arch": arch,
                       "dataset": ds_key, "direction": direction,
                       "sp": sp_spearman(q_emb, mapped)}

                # Jaccard@k: source geometry vs mapped geometry
                for k in JACCARD_KS:
                    kk = min(k, len(texts) - 1)
                    rec[f"jaccard@{k}"] = jaccard_at_k(q_emb, mapped, kk)

                # k-NN label transfer (labelled datasets only)
                if labels is not None:
                    # native accuracies cached per (dataset, encoder)
                    q_name = src if direction == "src2tgt" else tgt
                    m_name = tgt if direction == "src2tgt" else src
                    if (ds_key, q_name) not in knn_native_cache:
                        knn_native_cache[(ds_key, q_name)] = knn_label_accuracy(q_emb, labels)
                    if (ds_key, m_name) not in knn_native_cache:
                        knn_native_cache[(ds_key, m_name)] = knn_label_accuracy(mem_emb, labels)
                    acc_src = knn_native_cache[(ds_key, q_name)]
                    acc_tgt = knn_native_cache[(ds_key, m_name)]
                    acc_map = knn_label_accuracy(mapped, labels)
                    rec["knn_acc_source"] = acc_src
                    rec["knn_acc_mapped"] = acc_map
                    rec["knn_acc_target_native"] = acc_tgt
                    rec["knn_gap"] = acc_src - acc_map
                    rec["knn_retention"] = acc_map / acc_src if acc_src > 0 else float("nan")

                rows.append(rec)
                jac10 = rec.get("jaccard@10", float("nan"))
                extra = (f" | kNN src={rec['knn_acc_source']:.3f} "
                         f"map={rec['knn_acc_mapped']:.3f} "
                         f"gap={rec['knn_gap']:+.3f}") if labels is not None else ""
                print(f"    {ds_key:11s} {direction}: SP={rec['sp']:.3f} "
                      f"J@10={jac10:.3f}{extra}")

        del m_s2t, m_t2s
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    if not rows:
        print("\n[!] No mappers found. Nothing saved.")
        return

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "functional_sp_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[+] Saved: {csv_path}")

    # 3) SP as a predictor: correlate SP with functional outcomes
    print("\n=== SP AS A PREDICTOR (across all evaluated rows) ===")
    summary = {}
    tt = df[df.dataset == "tweettopic"].dropna(subset=["knn_retention"])
    if len(tt) >= 3:
        rho_knn, p_knn = spearmanr(tt["sp"], tt["knn_retention"])
        print(f"TweetTopic: Spearman(SP, kNN retention) = {rho_knn:.3f} (p={p_knn:.2e}, n={len(tt)})")
        summary["rho_sp_knn_retention_tweettopic"] = rho_knn
        summary["p_sp_knn_retention_tweettopic"] = p_knn
    for ds_key in df.dataset.unique():
        sub = df[df.dataset == ds_key]
        if len(sub) >= 3:
            rho_j, p_j = spearmanr(sub["sp"], sub["jaccard@10"])
            print(f"{ds_key}: Spearman(SP, Jaccard@10) = {rho_j:.3f} (p={p_j:.2e}, n={len(sub)})")
            summary[f"rho_sp_jaccard10_{ds_key}"] = rho_j
            summary[f"p_sp_jaccard10_{ds_key}"] = p_j
    with open(os.path.join(OUT_DIR, "sp_predictor_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Per-variant means for a quick read
    print("\n=== Per-variant means ===")
    metric_cols = [c for c in df.columns
                   if c.startswith("jaccard@") or c.startswith("knn_") or c == "sp"]
    print(df.groupby(["dataset", "variant"])[metric_cols].mean().round(3).to_string())


if __name__ == "__main__":
    main()