#!/usr/bin/env python3
"""
Adversarial Sentence Detection via Cross-Model Disagreement — EXTENDED
=======================================================================

Estensione del protocollo originale con:

NUOVE METRICHE (tutte attivabili/disattivabili da `METRIC_FLAGS`)
- `lid_mean`              : Local Intrinsic Dimensionality (Ma et al., 2018,
                            stimatore MLE di Amsaleg). Cattura "tasche locali
                            ad alta dimensionalità" — un segnale ortogonale
                            sia a struttura-vicinato che a densità. Restituito
                            con segno invertito così che "higher = clean".
- `embedding_consensus`   : cosine media a coppie tra le view del punto
                            proiettato (NON sui vicini). Misura il disaccordo
                            dei mapper su DOVE sta il punto.
- `hybrid_triple`         : media rank-normalizzata di Spearman + Density + LID.
                            Robusto alle scale eterogenee.

NUOVO PREPROCESSING (attivabile da `ENABLE_CENTERING` / `ENABLE_ABTT`)
- Centratura per-view con la media del corpus di riferimento: mitiga
  l'anisotropia dei sentence embedding ed estende il range dinamico delle
  cosine, migliorando S/N di TUTTE le metriche a valle.
- All-But-The-Top (Mu & Viswanath, 2018): dopo la centratura, rimuove le
  prime `ABTT_N_COMPONENTS` direzioni principali del reference. Spesso
  aggiunge un piccolo guadagno sopra la sola centratura.

Tutto il resto (loader AdvGLUE, valutazione paired/pooled, auto-detect
architettura mapper, tabella e plot) è identico all'originale per piena
compatibilità.
"""

# ============================================================================
# Imports
# ============================================================================
import os
import json
import sqlite3
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# CASPER mapper class (only the class is imported; no side effects)
from ALIGNMENT_OFFICIAL_diff_dim_linear import LinearInjectionMapper


# ============================================================================
# Section 1 — CONFIGURATION
# ============================================================================
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Target encoder defines the shared embedding space Z_T
TGT_NAME = "sentence-transformers/gtr-t5-base"

# Source encoders; each must have a matching CASPER mapper checkpoint.
SOURCES = [
    {
        "name":        "intfloat/e5-base-v2",
        "mapper_ckpt": "saved_mappers_official_gtr_e5_1M_MacroStr_SimTlm_linear/official_mapper_tgt2src.pt",
    },
    #{
    #    "name":        "ibm-granite/granite-embedding-278m-multilingual",
    #    "mapper_ckpt": "saved_mappers_official_gtr_gra_1M_MacroStr_SimTlm_linear/official_mapper_tgt2src.pt",
    #},
    #{
    #    "name":        "thenlper/gte-base",
    #    "mapper_ckpt": "saved_mappers_official_gtr_gte_1M_MacroStr_SimTlm_linear/official_mapper_tgt2src.pt",
    #},
    
]

# Inference-time mapper parameters
MAPPER_DROPOUT = 0.1
MAPPER_N_HEADS = 16

# ----------------------------------------------------------------------------
# *** TOGGLES: PREPROCESSING ***
# ----------------------------------------------------------------------------
# Sottrai la media del reference per ciascuna view prima di calcolare le sim.
# Mitiga l'anisotropia degli embedding (cono stretto attorno alla media globale)
# e aumenta il potere discriminante delle cosine. Costo: una sottrazione.
ENABLE_CENTERING = False

# All-But-The-Top: dopo la centratura, rimuove le prime ABTT_N_COMPONENTS
# direzioni principali del reference (PCA su 10k vettori, una volta sola).
# Spesso aggiunge un +0.02-0.05 di Cohen's d sopra la sola centratura.
# Quando ENABLE_ABTT=True, la centratura è inclusa automaticamente.
ENABLE_ABTT       = False
ABTT_N_COMPONENTS = 2

# ----------------------------------------------------------------------------
# *** TOGGLES: METRICHE ***
# ----------------------------------------------------------------------------
# Tutte le metriche sono nella convenzione "higher = clean".
# Spegnerne una salta sia il calcolo che la presenza nei report/plot.
# Gli hybrid sono calcolati solo se le rispettive componenti sono attive.
METRIC_FLAGS = {
    # --- segnali strutturali (originali) ---
    "spearman_pw_mean":      True,
    "jaccard_pw_mean":       False,
    # --- tipicità del manifold (originale) ---
    "density_topk_mean":     True,
    # --- NUOVI ---
    "lid_mean":              True,
    "embedding_consensus":   False,
    # --- hybrid ---
    "hybrid_struct_density": False,    # 0.5*Spearman + 0.5*Density (rescaled)
    "hybrid_triple":         True,    # rank-norm mean of Spearman+Density+LID
    "hybrid_knn_balanced": False,
}

# Peso del segnale strutturale nell'hybrid binario; 1-alpha va alla densità.
HYBRID_ALPHA = 0.5

# ----------------------------------------------------------------------------
# Modes / Datasets / Grids (invariati)
# ----------------------------------------------------------------------------
N_MODELS_GRID = [2, 3, 4]
DATASETS      = ["advglue_sst2", "advglue_qnli", "advglue_mnli"]
K_GRID      = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 170, 200, 250, 300, 500, 800, 1000, 1300, 1500, 1800, 2000, 2300, 2500, 3000, 5000]
GRID_POINTS = 50000

REF_CORPUS_SIZE = 10_000
REF_DB_PATH     = "nq_reservoir.db"
REF_HF_FALLBACK = "jxm/nq_corpus_dpr"

OUT_DIR = "adv_detection_results_EXTENDED"
os.makedirs(OUT_DIR, exist_ok=True)

# Pretty labels
DATASET_PRETTY = {
    "advglue_sst2": "AdvGLUE SST-2",
    "advglue_qnli": "AdvGLUE QNLI",
    "advglue_mnli": "AdvGLUE MNLI",
}
METRIC_PRETTY = {
    "spearman_pw_mean":      "Spearman",
    "jaccard_pw_mean":       "Jaccard",
    "density_topk_mean":     "Density",
    "lid_mean":              "LID",
    "embedding_consensus":   "EmbConsensus",
    "hybrid_struct_density": "Hybrid",
    "hybrid_triple":         "HybridTriple",
    "hybrid_knn_balanced": "HybridKNN-Bal",
}
# Stili per il plot — un colore/marker per metrica, usati solo se attive.
METRIC_STYLES = {
    "Spearman":     {"color": "#1f77b4", "marker": "o", "ls": "-"},
    "Jaccard":      {"color": "#d62728", "marker": "s", "ls": "--"},
    "Density":      {"color": "#2ca02c", "marker": "^", "ls": "-."},
    "LID":          {"color": "#ff7f0e", "marker": "v", "ls": "-"},
    "EmbConsensus": {"color": "#17becf", "marker": "P", "ls": "--"},
    "Hybrid":       {"color": "#9467bd", "marker": "D", "ls": ":"},
    "HybridTriple": {"color": "#8c564b", "marker": "X", "ls": "-"},
    "HybridKNN-Bal": {"color": "#e377c2", "marker": "h", "ls": "--"},
}


# ============================================================================
# Section 2 — ENCODER & EMBEDDING UTILITIES (invariate)
# ============================================================================
def load_embedding_backend(name: str):
    if name.startswith("sentence-transformers/") and SentenceTransformer is not None:
        model = SentenceTransformer(name)
        return model.tokenizer, model, "sentencetransformer"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name, trust_remote_code=True)
    return tok, model, "hf"


@torch.no_grad()
def embed_sentences(texts: List[str], tokenizer, model, device,
                    batch_size: int = 64, max_length: int = 256) -> torch.Tensor:
    is_st = (SentenceTransformer is not None) and isinstance(model, SentenceTransformer)
    if not is_st:
        model = model.to(device).eval()
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding", leave=False):
        batch = texts[i : i + batch_size]
        if is_st:
            e = model.encode(batch, convert_to_tensor=True, device=device,
                             show_progress_bar=False)
            embs.append(e.cpu())
        else:
            inp = tokenizer(batch, return_tensors="pt", padding=True,
                            truncation=True, max_length=max_length).to(device)
            out = model(**inp)
            mask = inp["attention_mask"].unsqueeze(-1)
            hidden = out.last_hidden_state * mask
            pooled = hidden.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            embs.append(pooled.cpu())
    return torch.cat(embs, dim=0)


# ============================================================================
# Section 3 — REFERENCE CORPUS (invariata)
# ============================================================================
def load_reference_corpus(n: int = REF_CORPUS_SIZE) -> List[str]:
    sents: List[str] = []
    if os.path.exists(REF_DB_PATH):
        conn = sqlite3.connect(REF_DB_PATH); cur = conn.cursor()
        cur.execute("SELECT text FROM sentences LIMIT ?", (n,))
        sents = [r[0] for r in cur.fetchall()]
        conn.close()
        if len(sents) >= n:
            return sents[:n]
    print(f"[*] Streaming {n - len(sents)} ref sentences from {REF_HF_FALLBACK}")
    ds = load_dataset(REF_HF_FALLBACK, split="train", streaming=True)
    for ex in ds:
        if len(sents) >= n: break
        text = (ex.get("text") or "").strip().replace("\n", " ")
        if text and len(text.split()) >= 5:
            sents.append(text[:300])
    return sents[:n]


# ============================================================================
# Section 4 — PAIRED DATASET LOADERS (invariati)
# ============================================================================
def _concat_pair(a: str, b: str) -> str:
    a, b = (a or "").strip(), (b or "").strip()
    if not a and not b: return ""
    if not a: return b
    if not b: return a
    if a[-1] not in ".!?": a += "."
    return f"{a} {b}"


def _build_paired_dataset(adv_rows, clean_by_idx, render_fn):
    sentences, labels, pair_ids = [], [], []
    pid = 0
    for ex in adv_rows:
        idx = ex.get("idx")
        if idx is None or idx not in clean_by_idx: continue
        benign, adv = clean_by_idx[idx], render_fn(ex)
        if (not benign) or (not adv) or (benign == adv): continue
        sentences += [benign, adv]
        labels   += [0, 1]
        pair_ids += [pid, pid]
        pid += 1
    return (sentences,
            np.array(labels, dtype=np.int64),
            np.array(pair_ids, dtype=np.int64))


def load_advglue_sst2():
    adv = load_dataset("adv_glue", "adv_sst2", split="validation")
    clean = load_dataset("glue", "sst2", split="validation")
    clean_by_idx = {ex["idx"]: ex["sentence"].strip() for ex in clean}
    return _build_paired_dataset(adv, clean_by_idx,
                                 render_fn=lambda ex: ex["sentence"].strip())


def load_advglue_qnli():
    adv = load_dataset("adv_glue", "adv_qnli", split="validation")
    clean = load_dataset("glue", "qnli", split="validation")
    clean_by_idx = {ex["idx"]: _concat_pair(ex["question"], ex["sentence"])
                    for ex in clean}
    return _build_paired_dataset(adv, clean_by_idx,
                                 render_fn=lambda ex: _concat_pair(ex["question"],
                                                                  ex["sentence"]))


def load_advglue_mnli():
    adv = load_dataset("adv_glue", "adv_mnli", split="validation")
    clean = load_dataset("glue", "mnli", split="validation_matched")
    clean_by_idx = {ex["idx"]: ex["hypothesis"].strip() for ex in clean}
    return _build_paired_dataset(adv, clean_by_idx,
                                 render_fn=lambda ex: ex["hypothesis"].strip())


DATASET_LOADERS = {
    "advglue_sst2": load_advglue_sst2,
    "advglue_qnli": load_advglue_qnli,
    "advglue_mnli": load_advglue_mnli,
}


# ============================================================================
# Section 5 — PREPROCESSING (NUOVO: centratura + ABTT)
# ============================================================================
def _center_per_view(views_test: List[torch.Tensor],
                     views_ref:  List[torch.Tensor]
                     ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Sottrae a ogni view la media del proprio reference. Riduce
    l'anisotropia ('cono' centrato attorno alla media globale) e amplia
    il range dinamico delle cosine, migliorando il S/N di tutte le metriche."""
    out_t, out_r = [], []
    for vt, vr in zip(views_test, views_ref):
        mu = vr.float().mean(dim=0, keepdim=True)
        out_t.append(vt.float() - mu)
        out_r.append(vr.float() - mu)
    return out_t, out_r


def _abtt_per_view(views_test: List[torch.Tensor],
                   views_ref:  List[torch.Tensor],
                   n_components: int
                   ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """All-But-The-Top (Mu & Viswanath, 2018):
       1) centra ogni view con la media del proprio reference;
       2) calcola le prime `n_components` direzioni principali del reference
          centrato (una PCA-lowrank, O(N*D*q));
       3) sottrae la proiezione su quelle direzioni da entrambi test e ref.
    Risultato: gli embedding diventano isotropi e privi delle direzioni
    dominanti tipicamente non-semantiche (frequenza dei token, lunghezza,
    formattazione)."""
    out_t, out_r = [], []
    for vt, vr in zip(views_test, views_ref):
        vt, vr = vt.float(), vr.float()
        mu = vr.mean(dim=0, keepdim=True)
        vt_c, vr_c = vt - mu, vr - mu
        # PCA lowrank sul reference centrato
        q = min(n_components, min(vr_c.shape) - 1)
        if q < 1:
            out_t.append(vt_c); out_r.append(vr_c); continue
        _, _, V = torch.pca_lowrank(vr_c, q=q, center=False)
        # V: [D, q] direzioni principali; rimuovile da test e ref
        proj_t = (vt_c @ V) @ V.T
        proj_r = (vr_c @ V) @ V.T
        out_t.append(vt_c - proj_t)
        out_r.append(vr_c - proj_r)
    return out_t, out_r


def apply_preprocessing(views_test: List[torch.Tensor],
                        views_ref:  List[torch.Tensor]
                        ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Routing del preprocessing in base ai toggle globali.
    Ordine: ABTT (che include la centratura) PRIORITARIO; altrimenti
    sola centratura se attivata; altrimenti no-op."""
    if ENABLE_ABTT:
        return _abtt_per_view(views_test, views_ref, ABTT_N_COMPONENTS)
    if ENABLE_CENTERING:
        return _center_per_view(views_test, views_ref)
    return views_test, views_ref


# ============================================================================
# Section 6 — METRICHE DI AGREEMENT
# ============================================================================
@torch.no_grad()
def compute_sim_cross(query_emb: torch.Tensor,
                      ref_emb: torch.Tensor) -> torch.Tensor:
    q = F.normalize(query_emb.float(), dim=-1).to(DEVICE)
    r = F.normalize(ref_emb.float(),   dim=-1).to(DEVICE)
    return (q @ r.T).cpu()


def topk_indices(sim: torch.Tensor, k: int) -> np.ndarray:
    return sim.topk(k, dim=-1).indices.numpy()


def jaccard_per_row(idx_a: np.ndarray, idx_b: np.ndarray) -> np.ndarray:
    out = np.empty(idx_a.shape[0], dtype=np.float32)
    for i in range(idx_a.shape[0]):
        a, b = set(idx_a[i].tolist()), set(idx_b[i].tolist())
        out[i] = len(a & b) / max(len(a | b), 1)
    return out


def spearman_per_row(sim_a: torch.Tensor, sim_b: torch.Tensor,
                     idx_a: np.ndarray, idx_b: np.ndarray) -> np.ndarray:
    sa, sb = sim_a.numpy(), sim_b.numpy()
    out = np.empty(idx_a.shape[0], dtype=np.float32)
    for i in range(idx_a.shape[0]):
        union = list(set(idx_a[i].tolist()) | set(idx_b[i].tolist()))
        if len(union) < 3:
            out[i] = 0.0; continue
        rho, _ = spearmanr(sa[i, union], sb[i, union])
        out[i] = 0.0 if np.isnan(rho) else rho
    return out


def _view_pairs(n_views: int):
    return [(i, j) for i in range(n_views) for j in range(i + 1, n_views)]


def multiview_jaccard_mean(idx_list: List[np.ndarray]) -> np.ndarray:
    pairs = _view_pairs(len(idx_list))
    per_pair = [jaccard_per_row(idx_list[i], idx_list[j]) for i, j in pairs]
    return np.mean(per_pair, axis=0)


def multiview_spearman_mean(sim_list: List[torch.Tensor],
                            idx_list: List[np.ndarray]) -> np.ndarray:
    pairs = _view_pairs(len(sim_list))
    per_pair = [spearman_per_row(sim_list[i], sim_list[j],
                                 idx_list[i], idx_list[j]) for i, j in pairs]
    return np.mean(per_pair, axis=0)


def multiview_density_mean(sim_list: List[torch.Tensor], K: int) -> np.ndarray:
    per_view = []
    for sim in sim_list:
        topk_sims = sim.topk(K, dim=-1).values
        per_view.append(topk_sims.mean(dim=-1).numpy())
    return np.mean(per_view, axis=0)


# ----------------------------------------------------------------------------
# NUOVO: Local Intrinsic Dimensionality (Ma et al. 2018, Amsaleg MLE)
# ----------------------------------------------------------------------------
def multiview_lid_mean(sim_list: List[torch.Tensor], K: int) -> np.ndarray:
    """LID via MLE:  LID(x) = -1 / mean_{i<K} log(d_i / d_K).

    Punti adversariali tendono a vivere in 'tasche' a dimensionalità
    intrinseca più alta — segnale ortogonale sia ai vicinati condivisi
    (Spearman/Jaccard) sia alla densità.  Restituiamo -LID così che
    'higher = clean' rimanga la convenzione comune."""
    eps = 1e-8
    per_view = []
    for sim in sim_list:
        dist = (1.0 - sim).clamp(min=eps)                      # [N, N_ref]
        topk = dist.topk(K, dim=-1, largest=False).values       # [N, K]
        d_max = topk[:, -1:].clamp(min=eps)
        ratios = (topk[:, :-1] / d_max).clamp(min=eps, max=1.0) # i = 1..K-1
        mean_log = ratios.log().mean(dim=-1).clamp(max=-eps)
        lid = -1.0 / mean_log                                   # positivo
        per_view.append(lid.numpy())
    return -np.mean(per_view, axis=0)


# ----------------------------------------------------------------------------
# NUOVO: Consensus diretto sull'embedding proiettato
# ----------------------------------------------------------------------------
def embedding_consensus_mean(views_test: List[torch.Tensor]) -> np.ndarray:
    """Cosine media a coppie tra le view del punto proiettato (non sui vicini).
    Sotto attacco i mapper diventano meno coerenti su DOVE collocare il
    punto: cattura un disaccordo che è invisibile alle metriche basate
    sul vicinato."""
    normed = [F.normalize(v.float(), dim=-1) for v in views_test]
    pairs = _view_pairs(len(normed))
    sims = [(normed[i] * normed[j]).sum(dim=-1).cpu().numpy() for i, j in pairs]
    return np.mean(sims, axis=0)


# ----------------------------------------------------------------------------
# Hybrid scores
# ----------------------------------------------------------------------------
def hybrid_struct_density_score(spearman_agreement: np.ndarray,
                                density: np.ndarray,
                                alpha: float = HYBRID_ALPHA) -> np.ndarray:
    s_norm = (spearman_agreement + 1.0) / 2.0
    d_norm = (density            + 1.0) / 2.0
    return alpha * s_norm + (1.0 - alpha) * d_norm


def _rank_normalize(x: np.ndarray) -> np.ndarray:
    """Empirical-CDF rescale a [0, 1]. Scale-free; preserva ordering
    quindi non altera direction_acc e AUC ma rende coerente l'aggregazione."""
    order = np.argsort(np.argsort(x))
    return order.astype(np.float32) / max(len(x) - 1, 1)


def hybrid_triple_score(spearman: np.ndarray,
                        density:  np.ndarray,
                        lid:      np.ndarray) -> np.ndarray:
    """Media rank-normalizzata dei tre segnali principali.
    LID ha scala diversa da Spearman/Density, quindi rank-norm è la fusione
    'sicura': dipende solo dall'ordinamento di ciascun segnale nel batch."""
    return ( _rank_normalize(spearman)
           + _rank_normalize(density)
           + _rank_normalize(lid) ) / 3.0

def hybrid_knn_balanced(jaccard: np.ndarray, #hybrid_knn_balanced
                        spearman: np.ndarray,
                        lid: np.ndarray,
                        struct_weight: float = 0.5) -> np.ndarray:
    """Balanced hybrid: avoids double-counting the K-NN structural signal.

    structural = mean(rank_norm(Jaccard), rank_norm(Spearman))   <- identity + order
    dimension  = rank_norm(LID)                                  <- manifold shape
    final      = struct_weight * structural + (1 - struct_weight) * dimension

    `struct_weight = 0.5` weights structural and dimensional facets equally,
    correcting the implicit 2:1 ratio you'd get from a uniform 3-way mean."""
    struct = 0.5 * (_rank_normalize(jaccard) + _rank_normalize(spearman))
    dim    = _rank_normalize(lid)
    return struct_weight * struct + (1.0 - struct_weight) * dim


# ============================================================================
# Section 7 — CASPER MAPPER LOADING (invariata)
# ============================================================================
def infer_mapper_arch_from_state(state: Dict[str, torch.Tensor]) -> Dict[str, int]:
    return {
        "input_dim" : state["input_proj.weight"].shape[1],
        "hidden_dim": state["input_proj.weight"].shape[0],
        "memory_dim": state["memory_proj.weight"].shape[1],
        "output_dim": state["output_proj.weight"].shape[0],
        "n_layers"  : 1 + max(int(k.split(".")[1])
                              for k in state if k.startswith("blocks.")),
    }


def load_mapper(ckpt_path: str, device,
                expected_src_dim: int = None,
                expected_tgt_dim: int = None):
    if not os.path.exists(ckpt_path):
        print(f"[!] Mapper checkpoint missing: {ckpt_path}")
        return None, None
    state = torch.load(ckpt_path, map_location=device)
    arch  = infer_mapper_arch_from_state(state)
    if expected_src_dim is not None and arch["input_dim"] != expected_src_dim:
        print(f"[!] Mapper input_dim mismatch: ckpt={arch['input_dim']}, "
              f"source encoder={expected_src_dim}. Skipping {ckpt_path}.")
        return None, None
    if expected_tgt_dim is not None and arch["output_dim"] != expected_tgt_dim:
        print(f"[!] Mapper output_dim mismatch: ckpt={arch['output_dim']}, "
              f"target encoder={expected_tgt_dim}. Skipping {ckpt_path}.")
        return None, None
    mp = LinearInjectionMapper(
        input_dim  = arch["input_dim"],
        output_dim = arch["output_dim"],
        memory_dim = arch["memory_dim"],
        hidden_dim = arch["hidden_dim"],
        n_layers   = arch["n_layers"],
        n_heads    = MAPPER_N_HEADS,
        dropout    = MAPPER_DROPOUT,
    ).to(device)
    mp.load_state_dict(state)
    mp.eval()
    return mp, arch


@torch.no_grad()
def project_into_target(mapper, e_src: torch.Tensor,
                        e_tgt_memory: torch.Tensor, device,
                        chunk: int = 512) -> torch.Tensor:
    out = []
    mapper = mapper.float()
    for i in range(0, e_src.size(0), chunk):
        x   = e_src[i:i + chunk].to(device).float()
        mem = e_tgt_memory[i:i + chunk].to(device).float()
        y, _ = mapper(x, memory=mem)
        out.append(y.cpu())
    return torch.cat(out, dim=0)


# ============================================================================
# Section 8 — VALUTAZIONE PAIRED / POOLED (invariata)
# ============================================================================
def paired_evaluation(agreement: np.ndarray, labels: np.ndarray,
                      pair_ids: np.ndarray) -> Dict[str, float]:
    bag: Dict[int, Dict[str, float]] = {}
    for i in range(len(labels)):
        pid = int(pair_ids[i])
        key = "benign" if labels[i] == 0 else "adv"
        bag.setdefault(pid, {})[key] = float(agreement[i])
    benign = np.array([d["benign"] for d in bag.values() if "benign" in d and "adv" in d])
    adv    = np.array([d["adv"]    for d in bag.values() if "benign" in d and "adv" in d])
    if len(benign) == 0:
        return dict(direction_acc=np.nan, mean_delta=np.nan, cohens_d=np.nan,
                    wilcoxon_p=np.nan, n_pairs=0)
    diff = benign - adv
    sd = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
    cohens_d = float(diff.mean() / sd) if sd > 0 else float("nan")
    try:
        _, p = wilcoxon(benign, adv, alternative="greater", zero_method="zsplit")
    except ValueError:
        p = float("nan")
    return dict(
        direction_acc=float((adv < benign).mean()),
        mean_delta=float(diff.mean()),
        cohens_d=cohens_d,
        wilcoxon_p=float(p),
        n_pairs=int(len(benign)),
    )


def pooled_evaluation(agreement: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    try:
        auc = roc_auc_score(labels, -agreement)
    except ValueError:
        auc = float("nan")
    lo, hi = float(agreement.min()), float(agreement.max())
    thresholds = np.linspace(lo, hi, GRID_POINTS)
    best_f1, best_thr = 0.0, np.nan
    for t in thresholds:
        pred = (agreement < t).astype(np.int64)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        if tp == 0: continue
        prec = tp / (tp + fp); rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        if f1 > best_f1: best_f1, best_thr = f1, float(t)
    return {"AUC": auc, "F1_best": best_f1, "threshold_best": best_thr}


# ============================================================================
# Section 9 — EXPERIMENT RUNNER (registry-driven)
# ============================================================================
def run_experiment(views_test: List[torch.Tensor],
                   views_ref:  List[torch.Tensor],
                   labels: np.ndarray, pair_ids: np.ndarray,
                   k_grid: List[int] = K_GRID) -> pd.DataFrame:
    """Sweep su K; ogni metrica è calcolata SOLO se il suo flag è attivo.
    Gli hybrid sono calcolati solo se le rispettive componenti sono attive."""
    sim_list = [compute_sim_cross(vt, vr) for vt, vr in zip(views_test, views_ref)]

    # embedding consensus è K-independent: una volta sola
    emb_cons = (embedding_consensus_mean(views_test)
                if METRIC_FLAGS.get("embedding_consensus", False) else None)

    rows = []
    for K in k_grid:
        idx_list = [topk_indices(sim, K) for sim in sim_list]
        agreements: Dict[str, np.ndarray] = {}

        if METRIC_FLAGS.get("spearman_pw_mean", False):
            agreements["spearman_pw_mean"] = multiview_spearman_mean(sim_list, idx_list)
        if METRIC_FLAGS.get("jaccard_pw_mean", False):
            agreements["jaccard_pw_mean"] = multiview_jaccard_mean(idx_list)
        if METRIC_FLAGS.get("density_topk_mean", False):
            agreements["density_topk_mean"] = multiview_density_mean(sim_list, K)
        if METRIC_FLAGS.get("lid_mean", False):
            agreements["lid_mean"] = multiview_lid_mean(sim_list, K)
        if emb_cons is not None:
            agreements["embedding_consensus"] = emb_cons

        # Hybrid binario richiede Spearman + Density
        if (METRIC_FLAGS.get("hybrid_struct_density", False)
                and "spearman_pw_mean" in agreements
                and "density_topk_mean" in agreements):
            agreements["hybrid_struct_density"] = hybrid_struct_density_score(
                agreements["spearman_pw_mean"], agreements["density_topk_mean"])

        # Hybrid triplo richiede Spearman + Density + LID
        if (METRIC_FLAGS.get("hybrid_triple", False)
                and "spearman_pw_mean" in agreements
                and "density_topk_mean" in agreements
                and "lid_mean" in agreements):
            agreements["hybrid_triple"] = hybrid_triple_score(
                agreements["spearman_pw_mean"],
                agreements["density_topk_mean"],
                agreements["lid_mean"])
            
        if (METRIC_FLAGS.get("hybrid_knn_balanced", False)
                and {"jaccard_pw_mean", "spearman_pw_mean", "lid_mean"} <= agreements.keys()):
            agreements["hybrid_knn_balanced"] = hybrid_knn_balanced(
                agreements["jaccard_pw_mean"],
                agreements["spearman_pw_mean"],
                agreements["lid_mean"])

        for metric_name, agr in agreements.items():
            paired = paired_evaluation(agr, labels, pair_ids)
            pooled = pooled_evaluation(agr, labels)
            rows.append({"K": K, "metric": metric_name, **paired, **pooled})
    return pd.DataFrame(rows)


# ============================================================================
# Section 10 — PER-DATASET PIPELINE
# ============================================================================
def evaluate_dataset(dataset_name: str,
                     target_encoder, mappers,
                     e_tgt_ref: torch.Tensor,
                     ref_source_views: List[Dict[str, Any]],
                     out_subdir: str) -> pd.DataFrame:
    os.makedirs(out_subdir, exist_ok=True)
    print(f"\n>>> Dataset: {dataset_name}")
    sentences, labels, pair_ids = DATASET_LOADERS[dataset_name]()
    n_pairs = int(pair_ids.max()) + 1 if len(pair_ids) else 0
    print(f"    {n_pairs} pairs ({len(sentences)} sentences)")
    if n_pairs == 0: return pd.DataFrame()

    t_tok, t_mod = target_encoder
    e_tgt_test = embed_sentences(sentences, t_tok, t_mod, DEVICE)

    test_source_views = []
    for cfg, mapper in mappers:
        s_tok, s_mod, _ = load_embedding_backend(cfg["name"])
        e_src = embed_sentences(sentences, s_tok, s_mod, DEVICE)
        e_proj = project_into_target(mapper, e_src, e_tgt_test, DEVICE)
        test_source_views.append({"name": cfg["name"], "e_proj_test": e_proj})
        del s_mod, e_src
        if DEVICE.type == "cuda": torch.cuda.empty_cache()

    all_rows = []
    for n_models in N_MODELS_GRID:
        n_sources = n_models - 1
        if n_sources > len(test_source_views):
            print(f"    [!] skipping {n_models}-model: only {len(test_source_views)} sources")
            continue
        views_test = [e_tgt_test] + [sv["e_proj_test"]
                                     for sv in test_source_views[:n_sources]]
        views_ref  = [e_tgt_ref]  + [rv["e_proj_ref"]
                                     for rv in ref_source_views[:n_sources]]

        # *** PREPROCESSING applicato per-view (centratura / ABTT / no-op) ***
        views_test, views_ref = apply_preprocessing(views_test, views_ref)

        print(f"    Running {n_models}-model mode "
              f"[center={ENABLE_CENTERING}, abtt={ENABLE_ABTT}]...")
        df = run_experiment(views_test, views_ref, labels, pair_ids)
        df["mode"]     = f"{n_models}-model"
        df["n_models"] = n_models
        df["dataset"]  = dataset_name
        all_rows.append(df)

    full = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    full.to_csv(os.path.join(out_subdir, "results_per_K.csv"), index=False)
    return full


# ============================================================================
# Section 11 — REFERENCE VIEWS (invariata)
# ============================================================================
def precompute_reference_views(ref_sents: List[str], target_encoder, mappers):
    t_tok, t_mod = target_encoder
    print(f"[*] Embedding {len(ref_sents)} reference sentences with TARGET...")
    e_tgt_ref = embed_sentences(ref_sents, t_tok, t_mod, DEVICE)
    print(f"    D_tgt = {e_tgt_ref.shape[1]}")
    ref_views = []
    for cfg, mapper in mappers:
        print(f"[*] Reference via {cfg['name']}...")
        s_tok, s_mod, _ = load_embedding_backend(cfg["name"])
        e_src = embed_sentences(ref_sents, s_tok, s_mod, DEVICE)
        e_proj = project_into_target(mapper, e_src, e_tgt_ref, DEVICE)
        ref_views.append({"name": cfg["name"], "e_proj_ref": e_proj})
        del s_mod, e_src
        if DEVICE.type == "cuda": torch.cuda.empty_cache()
    return e_tgt_ref, ref_views


# ============================================================================
# Section 12 — SUMMARY + PLOT (registry-driven)
# ============================================================================
def _sig_stars(p: float) -> str:
    if pd.isna(p): return ""
    if p < 1e-4:   return "***"
    if p < 1e-2:   return "**"
    if p < 0.05:   return "*"
    return ""


def _d_label(d: float) -> str:
    if pd.isna(d): return ""
    ad = abs(d)
    if ad >= 0.8: return "large"
    if ad >= 0.5: return "medium"
    if ad >= 0.2: return "small"
    return "negligible"


def build_summary_table(full_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, n_models, metric), g in full_results.groupby(
            ["dataset", "n_models", "metric"]):
        best = g.loc[g["direction_acc"].idxmax()]
        rows.append({
            "Dataset"    : DATASET_PRETTY.get(dataset, dataset),
            "Metric"     : METRIC_PRETTY.get(metric, metric),
            "#models"    : int(n_models),
            "K*"         : int(best["K"]),
            "DirAcc"     : round(float(best["direction_acc"]), 3),
            "meanΔ"      : round(float(best["mean_delta"]), 4),
            "Cohen d"    : round(float(best["cohens_d"]), 3),
            "Effect"     : _d_label(best["cohens_d"]),
            "Wilcoxon p" : float(best["wilcoxon_p"]),
            "Sig"        : _sig_stars(best["wilcoxon_p"]),
            "F1"         : round(float(best["F1_best"]), 3),
            "AUC"        : round(float(best["AUC"]), 3),
            "n_pairs"    : int(best["n_pairs"]),
        })
    df = pd.DataFrame(rows).sort_values(
        ["Dataset", "Metric", "#models"]).reset_index(drop=True)
    df["Wilcoxon p"] = df["Wilcoxon p"].map(
        lambda p: "n/a" if pd.isna(p)
        else (f"{p:.1e}" if p < 0.01 else f"{p:.3f}"))
    return df


def plot_marginal_contribution(summary: pd.DataFrame, save_path: str):
    """DirAcc vs n_encoders, facet per dataset, una curva per metrica
    presente nella summary. Adatta automaticamente la legenda."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    datasets_order = ["AdvGLUE SST-2", "AdvGLUE QNLI", "AdvGLUE MNLI"]
    present_metrics = list(summary["Metric"].unique())
    for ax, ds in zip(axes, datasets_order):
        for metric_name in present_metrics:
            style = METRIC_STYLES.get(metric_name,
                                      {"color": "grey", "marker": ".", "ls": "-"})
            sub = (summary[(summary["Dataset"] == ds) &
                           (summary["Metric"]  == metric_name)]
                   .sort_values("#models"))
            if sub.empty: continue
            ax.plot(sub["#models"], sub["DirAcc"],
                    color=style["color"], marker=style["marker"],
                    ls=style["ls"], lw=2, markersize=9,
                    label=metric_name)
        ax.axhline(0.5, ls=":", color="grey", alpha=0.6)
        ax.set_xticks([2, 3, 4]); ax.set_xlim(1.5, 4.5)
        ax.set_xlabel("Number of encoders")
        if ax is axes[0]: ax.set_ylabel("Paired direction accuracy")
        ax.set_title(ds)
        ax.set_ylim(0.45, 0.85)
        ax.legend(loc="lower right", fontsize=8, ncol=2)
    fig.suptitle("Marginal contribution: structural + density + LID + consensus",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path + ".pdf", dpi=300, bbox_inches="tight")
    fig.savefig(save_path + ".png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Section 13 — MAIN
# ============================================================================
def main():
    # ---- Target encoder ----
    print(f"[*] TARGET encoder: {TGT_NAME}")
    t_tok, t_mod, _ = load_embedding_backend(TGT_NAME)
    D_tgt = embed_sentences(["probe"], t_tok, t_mod, DEVICE).shape[1]
    print(f"    D_tgt = {D_tgt}")

    # ---- Preprocessing & metric configuration banner ----
    print("\n[*] ===== CONFIGURATION =====")
    print(f"    Preprocessing : centering={ENABLE_CENTERING}, "
          f"abtt={ENABLE_ABTT} (n={ABTT_N_COMPONENTS})")
    print(f"    Metrics       : "
          + ", ".join([k for k, v in METRIC_FLAGS.items() if v]))
    print("[*] =========================\n")

    # ---- Load mappers ----
    print("[*] Loading source encoders + CASPER mappers...")
    mappers = []
    for cfg in SOURCES:
        s_tok, s_mod, _ = load_embedding_backend(cfg["name"])
        D_src = embed_sentences(["probe"], s_tok, s_mod, DEVICE).shape[1]
        del s_mod
        if DEVICE.type == "cuda": torch.cuda.empty_cache()
        mp, arch = load_mapper(cfg["mapper_ckpt"], DEVICE,
                               expected_src_dim=D_src,
                               expected_tgt_dim=D_tgt)
        if mp is not None:
            mappers.append((cfg, mp))
            print(f"    [+] {cfg['name']}: D_src={D_src}, arch={arch}")
    print(f"[+] {len(mappers)} usable encoder/mapper pairs.")
    if not mappers: raise RuntimeError("No usable mappers.")

    # ---- Reference views ----
    ref_sents = load_reference_corpus()
    e_tgt_ref, ref_source_views = precompute_reference_views(
        ref_sents, (t_tok, t_mod), mappers)

    # ---- Run all datasets ----
    all_results = []
    for ds_name in DATASETS:
        out_sub = os.path.join(OUT_DIR, ds_name)
        df = evaluate_dataset(
            dataset_name=ds_name,
            target_encoder=(t_tok, t_mod),
            mappers=mappers,
            e_tgt_ref=e_tgt_ref,
            ref_source_views=ref_source_views,
            out_subdir=out_sub,
        )
        if not df.empty: all_results.append(df)
    if not all_results:
        print("[!] No results produced."); return

    # ---- Summary + plot ----
    full = pd.concat(all_results, ignore_index=True)
    full.to_csv(os.path.join(OUT_DIR, "full_results.csv"), index=False)

    summary = build_summary_table(full)
    summary.to_csv(os.path.join(OUT_DIR, "summary_table.csv"), index=False)
    summary.to_markdown(os.path.join(OUT_DIR, "summary_table.md"), index=False)
    try:
        summary.to_latex(os.path.join(OUT_DIR, "summary_table.tex"),
                         index=False, escape=False,
                         caption=("Cross-model disagreement detection (extended). "
                                  "Metric families: structural (Spearman, Jaccard), "
                                  "manifold-density (Density), local intrinsic "
                                  "dimensionality (LID), embedding consensus, and "
                                  "their hybrids. Cohen's d = paired effect size; "
                                  "significance (* p$<$0.05, ** p$<$1e-2, "
                                  "*** p$<$1e-4) via Wilcoxon signed-rank "
                                  "(H1: benign$>$adv)."),
                         label="tab:summary_extended")
    except Exception as e:
        print(f"[!] LaTeX export failed: {e}")

    plot_marginal_contribution(
        summary, os.path.join(OUT_DIR, "marginal_contribution"))

    # ---- Console summary ----
    print("\n" + "=" * 100)
    print("SUMMARY TABLE (paper-ready)")
    print("=" * 100)
    print(summary.to_string(index=False))
    print(f"\n[+] Saved to {OUT_DIR}/")

    # ---- Winner ranking by Cohen's d ----
    print("\n" + "=" * 100)
    print("SIGNAL ANALYSIS: top metric per dataset by Cohen's d")
    print("=" * 100)
    for ds in DATASET_PRETTY.values():
        sub = summary[summary["Dataset"] == ds].copy()
        if sub.empty: continue
        sub = sub.sort_values("Cohen d", ascending=False).head(5)
        print(f"\n{ds}:")
        print(sub[["Metric", "#models", "K*", "DirAcc",
                   "Cohen d", "Effect", "Sig"]].to_string(index=False))

    # ---- Run metadata ----
    with open(os.path.join(OUT_DIR, "run_meta.json"), "w") as f:
        json.dump({
            "target_encoder": TGT_NAME,
            "source_encoders": [cfg["name"] for cfg, _ in mappers],
            "mapper_arch_per_source": "auto-detected per checkpoint",
            "mapper_dropout_at_load": MAPPER_DROPOUT,
            "mapper_n_heads_parity": MAPPER_N_HEADS,
            "datasets": DATASETS,
            "n_models_grid": N_MODELS_GRID,
            "K_grid": K_GRID,
            "grid_points": GRID_POINTS,
            "ref_corpus_size": REF_CORPUS_SIZE,
            "preprocessing": {
                "enable_centering": ENABLE_CENTERING,
                "enable_abtt": ENABLE_ABTT,
                "abtt_n_components": ABTT_N_COMPONENTS,
            },
            "metric_flags": METRIC_FLAGS,
            "hybrid_alpha": HYBRID_ALPHA,
        }, f, indent=2)


if __name__ == "__main__":
    main()