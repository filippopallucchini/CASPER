#!/usr/bin/env python3
"""
CASPER — Minimal UMAP "case study" visualisation of semantic shift.

Instead of plotting hundreds of overlapping points (d ≈ 0.2 effect is too
small to see), we pick a handful of representative sentences per zone and
annotate them:

  - 8 sentences from the asymmetric zone with the largest CASPER residual
    (novel concepts that the older encoder struggles with).
  - 8 sentences from the both_know zone for contrast.

The figure shows, for each selected sentence, an arrow from its target
position (where the newer encoder places it) to its mapped position
(where CASPER projects the older encoder's embedding). Everyone else
is rendered as a faded gray background so the eye can focus.
"""

import json
import random
import textwrap
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import seaborn as sns

import umap
from rich.console import Console

console = Console()

# ─── Config ──────────────────────────────────────────────────────
SEED = 42
SRC_KEY = "gtr"           # older (2021)
TGT_KEY = "e5"            # newer (2023)
SRC_ERA = 2021
TGT_ERA = 2023
SRC_HF  = "gtr-t5-base"
TGT_HF  = "e5-base-v2"

MAPPER_PATH = Path("saved_mappers_official_gtr_e5_1M_MacroStr_SimTlm/official_mapper_src2tgt.pt")

N_EXAMPLES_PER_ZONE = 8   # how many "case studies" to annotate
DEVICE = torch.device("cuda:1" if torch.cuda.device_count() > 1 else
                      "cuda:0" if torch.cuda.is_available() else "cpu")

OUT_DIR = Path("temporal_drift_results")
OUT_DIR.mkdir(exist_ok=True)

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "temporal_sentence_cache"
EMB_CACHE_DIR = BASE_DIR / "temporal_emb_cache"


# ─── Mapper ──────────────────────────────────────────────────────
class TransformerEmbeddingMapper(nn.Module):
    def __init__(self, input_dim, output_dim, memory_dim, hidden_dim,
                 n_layers=4, n_heads=16, dropout=0.1):
        super().__init__()
        self.input_proj  = nn.Linear(input_dim, hidden_dim)
        self.memory_proj = nn.Linear(memory_dim, hidden_dim)
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.ln = nn.LayerNorm(output_dim)

    def forward(self, x, memory):
        h = self.input_proj(x.unsqueeze(1) if x.dim() == 2 else x)
        m = self.memory_proj(memory.unsqueeze(1) if memory.dim() == 2 else memory)
        h = self.transformer(tgt=h, memory=m)
        out = self.ln(self.output_proj(h))
        return out[:, 0, :] if x.dim() == 2 else out


def load_mapper(ckpt_path):
    sd = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    hidden_dim, input_dim = sd["input_proj.weight"].shape
    _, memory_dim         = sd["memory_proj.weight"].shape
    output_dim, _         = sd["output_proj.weight"].shape
    n_layers = sum(1 for k in sd if "self_attn.in_proj_weight" in k)
    m = TransformerEmbeddingMapper(input_dim, output_dim, memory_dim,
                                    hidden_dim, n_layers=n_layers)
    m.load_state_dict(sd); m.to(DEVICE).eval()
    return m


# ─── Data loading ────────────────────────────────────────────────
def load_sentences_and_years():
    candidates = sorted(CACHE_DIR.glob("sentences_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No sentence cache in {CACHE_DIR}")
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
            console.log(f"{model_key} embeddings: {tuple(emb.shape)}")
            return emb
    raise FileNotFoundError(f"No cache for {model_key} ({wanted} rows)")


# ─── Main ────────────────────────────────────────────────────────
def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    console.rule(f"[bold blue]UMAP case study — {SRC_KEY}→{TGT_KEY}[/bold blue]")

    sentences, years = load_sentences_and_years()
    src_emb = load_embeddings(SRC_KEY, sentences)
    tgt_emb = load_embeddings(TGT_KEY, sentences)

    era_min = min(SRC_ERA, TGT_ERA)
    era_max = max(SRC_ERA, TGT_ERA)

    both_mask = years <= era_min
    asym_mask = (years > era_min) & (years <= era_max)

    # Compute CASPER mapped embeddings for all sentences (fast enough)
    mapper = load_mapper(MAPPER_PATH)
    with torch.no_grad():
        mapped = []
        B = 512
        for i in range(0, len(sentences), B):
            s = src_emb[i:i+B].to(DEVICE)
            t = tgt_emb[i:i+B].to(DEVICE)
            mapped.append(mapper(s, memory=t).cpu())
        mapped = torch.cat(mapped)

    # Per-sentence displacement
    disp = (1 - F.cosine_similarity(mapped, tgt_emb)).numpy()

    # Background sample for context cloud
    rng = np.random.default_rng(SEED)
    bg_idx = rng.choice(len(sentences), size=1500, replace=False)

    # Case studies: top-k displacement per zone
    both_idx_all = np.where(both_mask)[0]
    asym_idx_all = np.where(asym_mask)[0]

    def top_k(mask_idx, k):
        sub_disp = disp[mask_idx]
        order = np.argsort(sub_disp)[::-1][:k]
        return mask_idx[order]

    both_case = top_k(both_idx_all, N_EXAMPLES_PER_ZONE)
    asym_case = top_k(asym_idx_all, N_EXAMPLES_PER_ZONE)
    case_idx = np.concatenate([both_case, asym_case])
    case_zone = (["both_know"] * len(both_case) +
                 ["asymmetric"] * len(asym_case))

    # Build the index set we need to project: background + case-study
    proj_idx = np.unique(np.concatenate([bg_idx, case_idx]))
    tgt_proj_input = tgt_emb[proj_idx].numpy()

    # Fit UMAP on target embeddings of the union
    console.log("Fitting UMAP (≈30 s) …")
    reducer = umap.UMAP(n_neighbors=20, min_dist=0.3, metric="cosine",
                        random_state=SEED)
    reducer.fit(tgt_proj_input)

    tgt_all_2d = reducer.transform(tgt_emb.numpy())
    map_all_2d = reducer.transform(mapped.numpy())

    # ── Plot ─────────────────────────────────────────────────────
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(14, 10))

    # Faded background
    ax.scatter(tgt_all_2d[bg_idx, 0], tgt_all_2d[bg_idx, 1],
               s=5, c="lightgray", alpha=0.35, zorder=1)

    # Case-study arrows + markers
    zone_colors = {"both_know": "#2e64a0",
                   "asymmetric": "#c44e52"}

    for idx, zone in zip(case_idx, case_zone):
        c = zone_colors[zone]
        t = tgt_all_2d[idx]
        m = map_all_2d[idx]

        arrow = FancyArrowPatch(tuple(t), tuple(m),
                                 arrowstyle="->", color=c,
                                 lw=1.6, mutation_scale=14, alpha=0.9,
                                 zorder=4)
        ax.add_patch(arrow)
        ax.scatter(*t, s=90, c=c, marker="o",
                   edgecolors="black", linewidths=0.6, zorder=5)
        ax.scatter(*m, s=180, c=c, marker="*",
                   edgecolors="black", linewidths=0.6, zorder=5)

        label = textwrap.shorten(sentences[idx], width=55, placeholder="…")
        ax.annotate(label,
                    xy=tuple(m), xytext=(8, 4),
                    textcoords="offset points",
                    fontsize=7.5, color=c, alpha=0.95,
                    bbox=dict(boxstyle="round,pad=0.25",
                              fc="white", ec=c, alpha=0.75, lw=0.5))

    both_avg = disp[both_case].mean()
    asym_avg = disp[asym_case].mean()

    ax.set_title(
        f"CASPER Semantic Shift — case study ({SRC_KEY}→{TGT_KEY})\n"
        f"top-{N_EXAMPLES_PER_ZONE} highest-drift sentences per zone  "
        f"[both_know avg Δ={both_avg:.3f}   "
        f"asymmetric avg Δ={asym_avg:.3f}]",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")

    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=zone_colors["both_know"], markeredgecolor="black",
               markersize=10, label=f"both_know  target (yr ≤ {era_min})"),
        Line2D([0], [0], marker="*", color="w",
               markerfacecolor=zone_colors["both_know"], markeredgecolor="black",
               markersize=14, label="both_know  CASPER-mapped"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=zone_colors["asymmetric"], markeredgecolor="black",
               markersize=10, label=f"asymmetric  target ({era_min} < yr ≤ {era_max})"),
        Line2D([0], [0], marker="*", color="w",
               markerfacecolor=zone_colors["asymmetric"], markeredgecolor="black",
               markersize=14, label="asymmetric  CASPER-mapped"),
        Line2D([0], [0], color="gray", alpha=0.5, lw=1.5, label="target → mapped shift"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="lightgray",
               markeredgecolor="none", markersize=6,
               label=f"background ({len(bg_idx)} random sentences)"),
    ]
    ax.legend(handles=legend_items, loc="upper left", fontsize=9,
              framealpha=0.95)

    for fmt in ("pdf", "png"):
        p = OUT_DIR / f"casper_umap_shift_{SRC_KEY}_{TGT_KEY}.{fmt}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        console.log(f"[green]Saved {p}[/green]")
    plt.close(fig)


if __name__ == "__main__":
    main()
