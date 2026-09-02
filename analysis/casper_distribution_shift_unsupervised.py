#!/usr/bin/env python3
"""
CASPER — distributional visualisation of semantic shift across the
asymmetric knowledge window. (UNSUPERVISED-MAPPER VARIANT)

Instead of a UMAP (which collapses a distributional effect into an
uninformative scatter), we plot the actual distributions of CASPER
displacement Δ for each knowledge zone:

  (A) KDE density per zone with means marked
  (B) Empirical CDF per zone — the rightward shift of the asymmetric
      curve demonstrates stochastic dominance of the effect.

Additional annotations report Cohen's d with bootstrap 95% CI and
Mann–Whitney p-value.
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import mannwhitneyu

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from rich.console import Console

console = Console()

# Import UNSUPERVISED CASPER mapper class
from ALIGNMENT_OFFICIAL_diff_dim_linear_unsupervised import StandaloneMapper

# ─── Config ──────────────────────────────────────────────────────
SEED = 42
SRC_KEY = "gtr"           # older (2021)
TGT_KEY = "e5"            # newer (2023)
SRC_ERA = 2021
TGT_ERA = 2023
SRC_HF  = "sentence-transformers/gtr-t5-base"
TGT_HF  = "intfloat/e5-base-v2"

# Puntiamo alla cartella dei mapper unsupervised
MAPPER_PATH = Path("saved_mappers_official_gtr_e5_1M_MacroStr_SimTlm_unsupervised/official_mapper_src2tgt.pt")
N_BOOT = 1000
DEVICE = torch.device("cuda:1" if torch.cuda.device_count() > 1 else
                      "cuda:0" if torch.cuda.is_available() else "cpu")

# Cartella di output separata per non sovrascrivere i run supervisionati
OUT_DIR = Path("temporal_drift_results_unsupervised")
OUT_DIR.mkdir(exist_ok=True)

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "temporal_sentence_cache"
EMB_CACHE_DIR = BASE_DIR / "temporal_emb_cache"


# ─── Mapper Loading (AUTO-DETECT ARCH, UNSUPERVISED) ──────────────
def infer_mapper_arch_from_state(state):
    """Infers StandaloneMapper architecture from its state_dict shapes."""
    blocks_keys = [int(k.split(".")[1]) for k in state if k.startswith("blocks.")]
    n_layers = (1 + max(blocks_keys)) if blocks_keys else 4
    return {
        "input_dim" : state["input_proj.weight"].shape[1],
        "hidden_dim": state["input_proj.weight"].shape[0],
        "output_dim": state["output_proj.weight"].shape[0],
        "n_layers"  : n_layers,
    }


def load_mapper(ckpt_path):
    sd = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    arch = infer_mapper_arch_from_state(sd)
    
    m = StandaloneMapper(
        input_dim=arch["input_dim"],
        output_dim=arch["output_dim"],
        memory_dim=arch["output_dim"], # Passato per parità di firma, ma ignorato internamente
        hidden_dim=arch["hidden_dim"],
        n_layers=arch["n_layers"]
    )
    m.load_state_dict(sd)
    m.to(DEVICE).eval()
    return m


# ─── Data ────────────────────────────────────────────────────────
def load_sentences_and_years():
    candidates = sorted(CACHE_DIR.glob("sentences_*.json"))
    path = candidates[-1]
    console.log(f"Sentence cache: [cyan]{path.name}[/cyan]")
    data = json.loads(path.read_text())
    sents, years = [], []
    for y in sorted(data.keys(), key=int):
        for s in data[y]:
            sents.append(s); years.append(int(y))
    return sents, np.array(years)


def load_embeddings(model_key, sentences):
    wanted = len(sentences)
    for p in sorted(EMB_CACHE_DIR.glob(f"{model_key}_{wanted}_*.pt")):
        emb = torch.load(p, map_location="cpu", weights_only=True)
        if emb.shape[0] == wanted:
            return emb
    raise FileNotFoundError(f"No cache for {model_key} ({wanted} rows)")


def bootstrap_d(pre, post, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n_pre, n_post = len(pre), len(post)
    ds = np.empty(n_boot)
    for i in range(n_boot):
        p = pre[rng.integers(0, n_pre, n_pre)]
        q = post[rng.integers(0, n_post, n_post)]
        pool = np.sqrt((p.std()**2 + q.std()**2) / 2)
        ds[i] = (q.mean() - p.mean()) / pool if pool > 0 else 0.0
    return float(ds.mean()), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


# ─── Main ────────────────────────────────────────────────────────
def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    console.rule(f"[bold blue]Distributional shift (Unsupervised) — {SRC_KEY}→{TGT_KEY}[/bold blue]")

    sentences, years = load_sentences_and_years()
    src_emb = load_embeddings(SRC_KEY, sentences)
    tgt_emb = load_embeddings(TGT_KEY, sentences)

    mapper = load_mapper(MAPPER_PATH)
    with torch.no_grad():
        parts = []
        B = 512
        for i in range(0, len(sentences), B):
            # Casting a .float() per evitare mismatch con pesi o NumPy
            s = src_emb[i:i+B].to(DEVICE).float()
            
            # UNSUPERVISED: Non passiamo più 't' al mapper
            mapped_batch, _ = mapper(s)   
            parts.append(mapped_batch.cpu())
        mapped = torch.cat(parts)

    disp = (1 - F.cosine_similarity(mapped.float(), tgt_emb.float())).numpy()

    era_min = min(SRC_ERA, TGT_ERA)
    era_max = max(SRC_ERA, TGT_ERA)

    both_d = disp[years <= era_min]
    asym_d = disp[(years > era_min) & (years <= era_max)]

    console.log(f"both_know: n={len(both_d)}  Δ̄={both_d.mean():.5f}  Δ̃={np.median(both_d):.5f}")
    console.log(f"asymmetric: n={len(asym_d)}  Δ̄={asym_d.mean():.5f}  Δ̃={np.median(asym_d):.5f}")

    # Stats
    d_mean, d_lo, d_hi = bootstrap_d(both_d, asym_d)
    u, p_mw = mannwhitneyu(both_d, asym_d, alternative="less")
    pct_shift = (asym_d.mean() - both_d.mean()) / both_d.mean() * 100

    # ── Figure ──────────────────────────────────────────────────
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6),
                             gridspec_kw={"wspace": 0.22})

    colors = {"both_know": "#2e64a0", "asymmetric": "#c44e52"}
    labels = {"both_know": f"both_know  (yr ≤ {era_min})  n={len(both_d):,}",
              "asymmetric": f"asymmetric  ({era_min} < yr ≤ {era_max})  n={len(asym_d):,}"}

    # ── Panel A: KDE with means ─────────────────────────────────
    ax = axes[0]
    for zone, data in [("both_know", both_d), ("asymmetric", asym_d)]:
        sns.kdeplot(data, ax=ax, color=colors[zone], lw=2.5,
                    fill=True, alpha=0.25, label=labels[zone],
                    clip=(0, None))
        ax.axvline(data.mean(), color=colors[zone], ls="--", lw=1.8,
                   alpha=0.9)
        # Small text above the mean line
        ylim = ax.get_ylim()
        ax.text(data.mean(), ylim[1] * 0.94,
                f"Δ̄={data.mean():.4f}",
                ha="center", color=colors[zone], fontsize=10,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2",
                          fc="white", ec=colors[zone], alpha=0.9, lw=0.6))

    ax.set_xlabel("CASPER Cosine Displacement  Δ", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("(A)  Density of CASPER displacement per zone",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.95)
    # Focus on where the mass is
    lo = np.percentile(np.concatenate([both_d, asym_d]), 1)
    hi = np.percentile(np.concatenate([both_d, asym_d]), 99)
    ax.set_xlim(lo, hi)

    # ── Panel B: ECDF ────────────────────────────────────────────
    ax2 = axes[1]
    for zone, data in [("both_know", both_d), ("asymmetric", asym_d)]:
        s = np.sort(data)
        y = np.arange(1, len(s) + 1) / len(s)
        ax2.plot(s, y, lw=2.5, color=colors[zone], label=labels[zone])
    ax2.set_xlabel("CASPER Cosine Displacement  Δ", fontsize=12)
    ax2.set_ylabel("Cumulative probability  P(Δ ≤ x)", fontsize=12)
    ax2.set_title("(B)  Empirical CDF — rightward shift = higher displacement",
                  fontsize=13, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=9.5, framealpha=0.95)
    ax2.set_xlim(lo, hi)

    # Statistical summary annotation
    stats_text = (
        f"Cohen's d  =  {d_mean:+.3f}   95% CI [{d_lo:+.3f}, {d_hi:+.3f}]\n"
        f"Mann–Whitney p  =  {p_mw:.2e}\n"
        f"Relative Δ̄ shift  =  {pct_shift:+.2f}%"
    )
    ax2.text(0.04, 0.55, stats_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.5",
                       fc="#fff8e7", ec="#444", lw=0.8))

    fig.suptitle(
        f"CASPER Displacement Distribution by Knowledge Zone (Unsupervised)\n"
        f"{SRC_KEY.upper()} ({SRC_ERA})  →  {TGT_KEY.upper()} ({TGT_ERA})",
        fontsize=14, fontweight="bold", y=1.02,
    )

    for fmt in ("pdf", "png"):
        p = OUT_DIR / f"casper_distribution_shift_{SRC_KEY}_{TGT_KEY}.{fmt}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        console.log(f"[green]Saved {p}[/green]")
    plt.close(fig)

    # Console summary
    console.rule("Summary")
    console.log(f"both_know    Δ̄ = {both_d.mean():.5f}   n = {len(both_d):,}")
    console.log(f"asymmetric   Δ̄ = {asym_d.mean():.5f}   n = {len(asym_d):,}")
    console.log(f"Relative shift: {pct_shift:+.2f}%")
    console.log(f"Cohen's d = {d_mean:+.4f}   95% CI [{d_lo:+.4f}, {d_hi:+.4f}]")
    console.log(f"Mann-Whitney p = {p_mw:.2e}")


if __name__ == "__main__":
    main()