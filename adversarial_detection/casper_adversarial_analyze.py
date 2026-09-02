#!/usr/bin/env python3
"""
CASPER — Adversarial Application: Step 2, Analyze.

Consumes triples (orig, clean, adv) produced by casper_adversarial_generate.py,
then for each version of each input:

  1. Embeds with src and tgt encoders (gtr, e5).
  2. Applies the trained CASPER mapper to get displacement Δ_CASPER.
  3. Computes the raw-cosine baseline Δ_raw = 1 - cos(z_s, z_t).
  4. Compares discriminability of {orig, clean} vs {adv} for each method:
     - Cohen's d with bootstrap 95% CI (1000 resamples)
     - Mann-Whitney U test
     - ROC-AUC (treating adv=1, not-adv=0)

Saves results JSON and a multi-panel figure suitable for the paper application
section, mirroring the conceptual layout of the introductory figure.

Usage:
  python casper_adversarial_analyze.py --device cuda:0
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, roc_curve

from rich.console import Console
from rich.table import Table

console = Console()

import os
BASE_DIR  = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "casper_adversarial_cache"
OUT_DIR   = BASE_DIR / "temporal_drift_results"  # match the paper's figures dir
OUT_DIR.mkdir(exist_ok=True)

ATTACK_NAME  = os.environ.get("CASPER_ADV_ATTACK", "deepwordbug")
PATH_TRIPLES = CACHE_DIR / f"triples_{ATTACK_NAME}.json"
PATH_EMB     = CACHE_DIR / f"embeddings_{ATTACK_NAME}.pt"
PATH_RESULTS = OUT_DIR / f"casper_adversarial_results_{ATTACK_NAME}.json"

# CASPER mapper for the gtr→e5 pair (linear-injection variant, matching paper's
# Eq. (1)–(3): h0 = Ws*z_s + Wt*z_t  →  4 residual FFN blocks  →  out_proj + LN)
SRC_KEY = "gtr"
TGT_KEY = "e5"
SRC_HF  = "sentence-transformers/gtr-t5-base"
TGT_HF  = "intfloat/e5-base-v2"
MAPPER_PATH = BASE_DIR / "saved_mappers_official_gtr_e5_1M_MacroStr_SimTlm_linear/official_mapper_src2tgt.pt"

SEED = 42
N_BOOT = 1000


# ─── CASPER mapper architecture (linear-injection variant) ───────
# Matches the LinearInjectionMapper defined in
# ALIGNMENT_OFFICIAL_diff_dim_linear.py, exactly:
#   h0 = W_s @ z_s + W_t @ z_t  (with biases)        — Eq. (1)
#   for each block:  h = h + FFN(LN(h))               — Eq. (2)
#       FFN = Linear(h*4) → GELU → Dropout → Linear → Dropout
#   h = final_ln(h);  out = output_proj(h);  out = out_ln(out)   — Eq. (3)
#
# Each block's Sequential has 6 elements (.0 LN, .1 Linear, .2 GELU,
# .3 Dropout, .4 Linear, .5 Dropout); only .0/.1/.4 carry parameters.
# The forward returns (out, h) for parity with the cross-attentive
# legacy API; we unpack at the call site.
class LinearInjectionMapper(nn.Module):
    def __init__(self, input_dim, output_dim, memory_dim,
                 hidden_dim=1024, n_layers=4, n_heads=16, dropout=0.1):
        super().__init__()
        # n_heads kept for API parity; unused in this variant.
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.input_proj  = nn.Linear(input_dim,  hidden_dim)
        self.memory_proj = nn.Linear(memory_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),                # .0
                nn.Linear(hidden_dim, hidden_dim * 4),   # .1
                nn.GELU(),                                # .2 (no params)
                nn.Dropout(dropout),                      # .3 (no params)
                nn.Linear(hidden_dim * 4, hidden_dim),   # .4
                nn.Dropout(dropout),                      # .5 (no params)
            )
            for _ in range(n_layers)
        ])

        self.final_ln    = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.out_ln      = nn.LayerNorm(output_dim)

    def forward(self, x, memory):
        if x.dim() == 3:
            x = x.squeeze(1)
        if memory.dim() == 3:
            memory = memory.squeeze(1)
        h = self.input_proj(x) + self.memory_proj(memory)
        for block in self.blocks:
            h = h + block(h)
        h = self.final_ln(h)
        out = self.output_proj(h)
        out = self.out_ln(out)
        return out, h


# Alias kept so any external code that referenced the old name still works
TransformerEmbeddingMapper = LinearInjectionMapper


def load_mapper(ckpt_path, device):
    """Instantiate LinearInjectionMapper with all dims auto-inferred."""
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    hidden_dim, input_dim = sd["input_proj.weight"].shape
    _, memory_dim         = sd["memory_proj.weight"].shape
    output_dim, _         = sd["output_proj.weight"].shape
    # Count blocks by collecting unique block indices
    n_layers = len({int(k.split(".")[1]) for k in sd if k.startswith("blocks.")})
    m = LinearInjectionMapper(input_dim, output_dim, memory_dim,
                              hidden_dim=hidden_dim, n_layers=n_layers)
    m.load_state_dict(sd)
    m.to(device).eval()
    return m


# ─── Embedding ───────────────────────────────────────────────────
@torch.no_grad()
def embed_with_hf(texts, hf_name, device, batch_size=32, max_len=128):
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
    mod = AutoModel.from_pretrained(hf_name, trust_remote_code=True).to(device).eval()
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len).to(device)
        out = mod(**enc)
        mask = enc["attention_mask"].unsqueeze(-1)
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
        embs.append(pooled.cpu())
    del mod, tok
    torch.cuda.empty_cache()
    return torch.cat(embs)


@torch.no_grad()
def embed_with_st(texts, hf_name, device, batch_size=32):
    from sentence_transformers import SentenceTransformer
    mod = SentenceTransformer(hf_name, device="cpu").to(device)
    return mod.encode(texts, convert_to_tensor=True, device=device,
                      batch_size=batch_size, show_progress_bar=False).cpu()


def get_embeddings(triples, device):
    """Embed all three text variants with both encoders."""
    if PATH_EMB.exists():
        console.log(f"[green]Embeddings cached[/green] → {PATH_EMB.name}")
        return torch.load(PATH_EMB, map_location="cpu", weights_only=True)

    console.rule("[bold]Embedding all three variants with gtr and e5[/bold]")
    out = {}
    for variant in ("orig", "clean", "adv"):
        texts = [t[variant] for t in triples]
        console.log(f"  Embedding {len(texts)} {variant} sentences with {SRC_KEY}…")
        out[f"src_{variant}"] = embed_with_st(texts, SRC_HF, device)
        console.log(f"  Embedding {len(texts)} {variant} sentences with {TGT_KEY}…")
        out[f"tgt_{variant}"] = embed_with_hf(texts, TGT_HF, device)
        torch.cuda.empty_cache()
    torch.save(out, PATH_EMB)
    console.log(f"  Saved → {PATH_EMB.name}")
    return out


# ─── Displacement computation ────────────────────────────────────
@torch.no_grad()
def casper_displacement(mapper, src_emb, tgt_emb, device, batch_size=512):
    """Δ = 1 − cos(mapper(z_s, z_t), z_t).

    LinearInjectionMapper.forward returns (out, h_compat); we keep only out.
    """
    parts = []
    for i in range(0, len(src_emb), batch_size):
        s = src_emb[i:i+batch_size].to(device)
        t = tgt_emb[i:i+batch_size].to(device)
        mapped, _ = mapper(s, t)
        parts.append((1 - F.cosine_similarity(mapped, t)).cpu())
    return torch.cat(parts).numpy()


def raw_displacement(src_emb, tgt_emb):
    return (1 - F.cosine_similarity(src_emb, tgt_emb, dim=1)).numpy()


# ─── Statistics ──────────────────────────────────────────────────
def bootstrap_d(a, b, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n_a, n_b = len(a), len(b)
    ds = np.empty(n_boot)
    for i in range(n_boot):
        sa = a[rng.integers(0, n_a, n_a)]
        sb = b[rng.integers(0, n_b, n_b)]
        pool = np.sqrt((sa.std()**2 + sb.std()**2) / 2)
        ds[i] = (sb.mean() - sa.mean()) / pool if pool > 0 else 0.0
    return float(ds.mean()), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


def analyze_method(disp_orig, disp_clean, disp_adv, method_name):
    benign = np.concatenate([disp_orig, disp_clean])  # adv=0 class
    advers = disp_adv                                  # adv=1 class

    d_mean, d_lo, d_hi = bootstrap_d(benign, advers)
    u, p = mannwhitneyu(benign, advers, alternative="less")

    y_true = np.concatenate([np.zeros(len(benign)), np.ones(len(advers))])
    y_score = np.concatenate([benign, advers])
    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)

    return {
        "method": method_name,
        "mean_orig":  float(disp_orig.mean()),
        "mean_clean": float(disp_clean.mean()),
        "mean_adv":   float(disp_adv.mean()),
        "median_orig":  float(np.median(disp_orig)),
        "median_clean": float(np.median(disp_clean)),
        "median_adv":   float(np.median(disp_adv)),
        "n":             int(len(disp_orig)),
        "cohens_d":      d_mean,
        "cohens_d_ci":   [d_lo, d_hi],
        "mannwhitney_p": float(p),
        "auc":           float(auc),
        "fpr":           fpr.tolist(),
        "tpr":           tpr.tolist(),
    }


# ─── Figure ──────────────────────────────────────────────────────
def make_figure(results, out_path):
    """
    3-panel figure mirroring the intro:
      (A) Schematic recap (text-only annotation, since the actual diagram lives
          in the intro figure already in the paper)
      (B) Distribution of Δ per method, faceted by variant
      (C) ROC curves comparing CASPER vs raw cosine
    """
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(15, 9))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)

    method_colors = {"CASPER": "#c44e52", "raw_cosine": "#2e64a0"}
    variant_colors = {"orig": "#4c72b0", "clean": "#55a868", "adv": "#c44e52"}
    variant_labels = {"orig": "original", "clean": "clean paraphrase",
                      "adv": "adversarial paraphrase"}

    # ── (A) Schematic block (text annotation, top-left) ───────────
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    ax.set_title("(A)  Cross-encoder consensus protocol",
                 fontsize=13, fontweight="bold", loc="left")
    schema = (
        "Input  x\n"
        "  ├─►  E_src(x)  =  z_s\n"
        "  └─►  E_tgt(x)  =  z_t\n"
        "\n"
        "Δ_CASPER  =  1 − cos(f_θ(z_s; z_t),  z_t)\n"
        "Δ_raw     =  1 − cos(z_s,  z_t)\n"
        "\n"
        "Hypothesis: adversarial paraphrases that fool one\n"
        "encoder but not the other produce systematically\n"
        "larger Δ_CASPER than benign paraphrases.\n"
        "\n"
        f"src = {SRC_HF.split('/')[-1]}\n"
        f"tgt = {TGT_HF.split('/')[-1]}\n"
        f"victim classifier: bert-base-uncased-imdb\n"
        f"attack: TextFooler"
    )
    ax.text(0.02, 0.95, schema, transform=ax.transAxes,
            fontsize=10, family="monospace", verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", fc="#f8f8f8", ec="#666"))

    # ── (B) Distributions, top-right ──────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    casper = results["CASPER"]
    raw    = results["raw_cosine"]
    rows = []
    for variant in ("orig", "clean", "adv"):
        for d in casper["disp"][variant]:
            rows.append({"method": "CASPER", "variant": variant, "Δ": d})
    import pandas as pd
    df = pd.DataFrame(rows)
    sns.violinplot(data=df, x="variant", y="Δ", ax=ax,
                   order=["orig", "clean", "adv"],
                   palette=[variant_colors[v] for v in ["orig", "clean", "adv"]],
                   inner="quartile", cut=0)
    ax.set_xticklabels([variant_labels[v] for v in ["orig", "clean", "adv"]])
    ax.set_xlabel("")
    ax.set_ylabel("CASPER displacement  Δ", fontsize=11)
    ax.set_title("(B)  Distribution of Δ_CASPER by input variant",
                 fontsize=13, fontweight="bold", loc="left")
    # Annotate Cohen's d
    d_lo, d_hi = casper["cohens_d_ci"]
    ax.text(0.02, 0.96,
            f"Cohen's d  =  {casper['cohens_d']:+.3f}  [{d_lo:+.3f}, {d_hi:+.3f}]\n"
            f"AUC (adv vs benign)  =  {casper['auc']:.3f}",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fff8e7", ec="#444"))

    # ── (C) Same for raw cosine, bottom-left ──────────────────────
    ax = fig.add_subplot(gs[1, 0])
    rows = []
    for variant in ("orig", "clean", "adv"):
        for d in raw["disp"][variant]:
            rows.append({"variant": variant, "Δ": d})
    df = pd.DataFrame(rows)
    sns.violinplot(data=df, x="variant", y="Δ", ax=ax,
                   order=["orig", "clean", "adv"],
                   palette=[variant_colors[v] for v in ["orig", "clean", "adv"]],
                   inner="quartile", cut=0)
    ax.set_xticklabels([variant_labels[v] for v in ["orig", "clean", "adv"]])
    ax.set_xlabel("")
    ax.set_ylabel("Raw cosine displacement  Δ", fontsize=11)
    ax.set_title("(C)  Same, with raw-cosine baseline (no alignment)",
                 fontsize=13, fontweight="bold", loc="left")
    d_lo, d_hi = raw["cohens_d_ci"]
    ax.text(0.02, 0.96,
            f"Cohen's d  =  {raw['cohens_d']:+.3f}  [{d_lo:+.3f}, {d_hi:+.3f}]\n"
            f"AUC (adv vs benign)  =  {raw['auc']:.3f}",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fff8e7", ec="#444"))

    # ── (D) ROC curves, bottom-right ──────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    for method, info in results.items():
        ax.plot(info["fpr"], info["tpr"], lw=2.5,
                color=method_colors.get(method, "gray"),
                label=f"{method}  (AUC = {info['auc']:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1, label="chance")
    ax.set_xlabel("False positive rate", fontsize=11)
    ax.set_ylabel("True positive rate", fontsize=11)
    ax.set_title("(D)  ROC: detecting adversarial paraphrases",
                 fontsize=13, fontweight="bold", loc="left")
    ax.legend(loc="lower right", fontsize=10)

    fig.suptitle(
        "CASPER — Cross-Encoder Consensus for Adversarial-Paraphrase Detection\n"
        f"IMDB sentiment, TextFooler attack against BERT classifier  "
        f"(N = {results['CASPER']['n']} triples)",
        fontsize=14, fontweight="bold", y=0.995,
    )

    for fmt in ("pdf", "png"):
        p = out_path.with_suffix(f".{fmt}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        console.log(f"[green]Saved {p.name}[/green]")
    plt.close(fig)


# ─── Main ────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda:0")
    args = p.parse_args()

    if not PATH_TRIPLES.exists():
        console.log(f"[red]No triples found at {PATH_TRIPLES}.\n"
                    f"Run casper_adversarial_generate.py first.[/red]")
        return

    triples = json.loads(PATH_TRIPLES.read_text())
    console.rule("[bold blue]CASPER — adversarial analysis[/bold blue]")
    console.log(f"Loaded {len(triples)} triples")
    console.log(f"Device: {args.device}")

    # 1. Embed
    embs = get_embeddings(triples, args.device)

    # 2. Apply CASPER and raw cosine
    console.rule("[bold]Computing displacements[/bold]")
    mapper = load_mapper(MAPPER_PATH, args.device)

    results = {}
    casper_disp = {}
    raw_disp = {}
    for variant in ("orig", "clean", "adv"):
        s = embs[f"src_{variant}"].float()
        t = embs[f"tgt_{variant}"].float()
        casper_disp[variant] = casper_displacement(mapper, s, t, args.device)
        raw_disp[variant]    = raw_displacement(s, t)
        console.log(f"  {variant}: Δ_CASPER mean = {casper_disp[variant].mean():.5f}  "
                    f"Δ_raw mean = {raw_disp[variant].mean():.5f}")

    del mapper
    torch.cuda.empty_cache()

    # 3. Analyze
    console.rule("[bold]Analyzing[/bold]")
    res_casper = analyze_method(casper_disp["orig"], casper_disp["clean"],
                                 casper_disp["adv"], "CASPER")
    res_raw    = analyze_method(raw_disp["orig"], raw_disp["clean"],
                                 raw_disp["adv"], "raw_cosine")

    res_casper["disp"] = {k: v.tolist() for k, v in casper_disp.items()}
    res_raw["disp"]    = {k: v.tolist() for k, v in raw_disp.items()}

    results = {"CASPER": res_casper, "raw_cosine": res_raw}

    # 4. Print summary
    tbl = Table(title="Adversarial discrimination summary")
    tbl.add_column("Method", style="bold")
    tbl.add_column("N", justify="right")
    tbl.add_column("Δ̄ orig", justify="right")
    tbl.add_column("Δ̄ clean", justify="right")
    tbl.add_column("Δ̄ adv", justify="right")
    tbl.add_column("Cohen's d", justify="right")
    tbl.add_column("95% CI", justify="center")
    tbl.add_column("AUC", justify="right")
    for k in ("CASPER", "raw_cosine"):
        r = results[k]
        ci = f"[{r['cohens_d_ci'][0]:+.3f}, {r['cohens_d_ci'][1]:+.3f}]"
        tbl.add_row(k, str(r["n"]),
                    f"{r['mean_orig']:.5f}",
                    f"{r['mean_clean']:.5f}",
                    f"{r['mean_adv']:.5f}",
                    f"{r['cohens_d']:+.4f}",
                    ci,
                    f"{r['auc']:.3f}")
    console.print(tbl)

    # 5. Save
    PATH_RESULTS.write_text(json.dumps(results, indent=2))
    console.log(f"[green]JSON → {PATH_RESULTS}[/green]")

    # 6. Figure
    make_figure(results, OUT_DIR / f"casper_adversarial_figure_{ATTACK_NAME}")

    # 7. Hand-off file for the vec2vec session
    handoff = OUT_DIR / f"casper_adversarial_triples_for_vec2vec_{ATTACK_NAME}.json"
    handoff.write_text(json.dumps({
        "triples": triples,
        "embeddings_path": str(PATH_EMB),
        "casper_results": {
            "cohens_d": res_casper["cohens_d"],
            "cohens_d_ci": res_casper["cohens_d_ci"],
            "auc": res_casper["auc"],
        },
        "instructions": (
            "Run vec2vec_src2tgt on src_orig/src_clean/src_adv embeddings, "
            "compute Δ_v2v = 1 − cos(v2v(z_s), z_t) for each variant, then "
            "compute Cohen's d and AUC between {orig+clean} and {adv} groups. "
            "Mirror the schema of casper_adversarial_results.json."
        ),
    }, indent=2))
    console.log(f"[cyan]Hand-off for vec2vec session → {handoff.name}[/cyan]")


if __name__ == "__main__":
    main()
