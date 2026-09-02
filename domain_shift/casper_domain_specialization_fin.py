#!/usr/bin/env python3
"""
CASPER — Application: Domain Specialization Detection.

Tests whether CASPER's residual reveals when an input lies in a
domain-specialized region of one encoder's manifold but not the other's.
Uses the FinBERT (financial-domain) ↔ GTR (general) pair.

Hypothesis
----------
For sentences with high financial jargon density, the two encoders
produce inconsistent representations (FinBERT encodes domain semantics
that GTR does not), so the trained CASPER mapper—calibrated on a
generic NQ-style corpus—cannot fully reconcile them and leaves an
elevated residual. For non-financial sentences, both encoders are
"generic" and the mapping is tight.

Pipeline
--------
1. Sample candidate sentences from cc_news (and optionally
   stanford-oval/ccnews) — same source for all zones to eliminate
   source-distribution confound.
2. Compute a financial-specificity score per sentence (count of
   finance-lexicon hits, normalized for length).
3. Stratify into three zones:
     non_financial  — score = 0
     mixed          — score in {1, 2}
     financial_jargon — score >= 3
   Sample N per zone.
4. Embed with FinBERT (src) and GTR (tgt).
5. Apply trained CASPER mapper (src2tgt, FinBERT → GTR direction).
6. Compute Δ_CASPER and Δ_raw per sentence.
7. Stats: Cohen's d (non_financial vs financial_jargon), bootstrap
   95% CI, Mann–Whitney U, ROC-AUC.
8. Save JSON + 4-panel figure.
"""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Force TensorFlow to CPU if some downstream import pulls it in
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

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

# ─── Config ──────────────────────────────────────────────────────
SEED = 42
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SRC_KEY = "finbert"
TGT_KEY = "gtr"
SRC_HF  = "ProsusAI/finbert"
TGT_HF  = "sentence-transformers/gtr-t5-base"

BASE_DIR = Path(__file__).resolve().parent
MAPPER_PATH = BASE_DIR / "saved_mappers_official_finbert_gtr_1M_MacroStr_SimTlm_linear/official_mapper_src2tgt.pt"

CACHE_DIR = BASE_DIR / "casper_domain_cache"
CACHE_DIR.mkdir(exist_ok=True)
OUT_DIR = BASE_DIR / "temporal_drift_results"  # match other paper figures
OUT_DIR.mkdir(exist_ok=True)

PATH_SENTENCES = CACHE_DIR / "domain_sentences.json"
PATH_EMB       = CACHE_DIR / "embeddings.pt"
PATH_RESULTS   = OUT_DIR / "casper_domain_results.json"

# Stratification
SAMPLES_PER_ZONE  = 2000
CANDIDATE_TARGET  = 60_000   # candidate sentences to scan from cc_news
SENTENCE_MIN_WORDS = 8
SENTENCE_MAX_WORDS = 30
JARGON_THRESHOLD  = 3        # >=3 finance terms → financial_jargon

# Bootstrap
N_BOOT = 1000


# ─── Financial lexicon ───────────────────────────────────────────
# Curated list of finance-domain terms. Designed to fire on financial
# news/jargon without over-matching common business English. Lower-cased
# match against word boundaries; multi-word phrases also matched.
_FINANCE_LEXICON = sorted({
    # Securities / instruments
    "stock", "stocks", "share", "shares", "shareholder", "shareholders",
    "bond", "bonds", "bondholder", "equity", "equities", "security", "securities",
    "derivative", "derivatives", "futures", "options", "warrant", "warrants",
    "etf", "etfs", "mutual fund", "hedge fund", "hedge funds",
    "convertible", "preferred stock", "common stock",
    # Markets / trading
    "nasdaq", "nyse", "dow jones", "s&p 500", "ftse", "nikkei", "hang seng",
    "bull market", "bear market", "trading", "trader", "traders", "broker",
    "brokerage", "exchange", "ipo", "listing",
    "yield", "yields", "spread", "spreads", "volatility", "liquidity",
    "arbitrage", "short selling", "long position", "short position",
    "bid", "ask", "ticker", "premarket", "aftermarket",
    # Corporate finance
    "earnings", "earnings per share", "eps", "ebitda", "revenue", "revenues",
    "profit", "profits", "loss", "losses", "margin", "margins",
    "dividend", "dividends", "buyback", "buybacks", "payout",
    "balance sheet", "cash flow", "income statement",
    "asset", "assets", "liability", "liabilities", "goodwill",
    "depreciation", "amortization", "accruals", "writedown", "writedowns",
    "p/e ratio", "p/e", "price-to-earnings", "valuation",
    "market cap", "market capitalization", "capitalization",
    "fiscal year", "fiscal", "quarterly", "quarter", "q1", "q2", "q3", "q4",
    "year-over-year", "yoy", "year-on-year",
    # Financial actors
    "investor", "investors", "shareholder", "creditor", "creditors",
    "underwriter", "analyst", "analysts", "fund manager", "portfolio manager",
    "bank", "banks", "banking", "lender", "lenders", "lending", "borrower",
    # Macro / monetary
    "interest rate", "interest rates", "rate hike", "rate cut",
    "federal reserve", "the fed", "ecb", "boe", "central bank", "monetary policy",
    "inflation", "deflation", "recession", "recessionary", "stagflation",
    "gdp", "gross domestic product", "treasury", "treasuries", "sovereign debt",
    "deficit", "surplus", "fiscal stimulus",
    # Deals / events
    "merger", "mergers", "acquisition", "acquisitions", "m&a", "takeover", "takeovers",
    "buyout", "leveraged buyout", "lbo", "spinoff", "spin-off", "divestiture",
    "bankruptcy", "chapter 11", "default", "defaults", "restructuring",
    # Regulation / accounting
    "sec", "securities and exchange commission", "fdic", "regulator", "regulators",
    "compliance", "audit", "auditor", "gaap", "ifrs",
    # Loans / credit
    "mortgage", "mortgages", "credit", "credit rating", "downgrade", "upgrade",
    "junk bond", "investment grade", "subprime",
    # Growth / valuation
    "guidance", "outlook", "forecast", "consensus estimate", "beat estimates",
    "miss estimates", "revised guidance",
    "valuation", "overvalued", "undervalued", "fair value",
    # Crypto-finance
    "cryptocurrency", "bitcoin", "ethereum", "blockchain", "stablecoin",
}, key=len, reverse=True)  # match longest phrases first

# Pre-compile regex: word-bounded, lower-case
def _build_finance_regex(lexicon):
    escaped = [re.escape(term) for term in lexicon]
    pattern = r"(?<![a-z0-9])(" + "|".join(escaped) + r")(?![a-z0-9])"
    return re.compile(pattern, flags=re.IGNORECASE)

_FIN_RE = _build_finance_regex(_FINANCE_LEXICON)
# Also detect ticker-like tokens: 2–5 capital letters, often preceded by '$'
_TICKER_RE = re.compile(r"\$[A-Z]{2,5}\b|\b[A-Z]{3,5}\b")
# Common false positives for the bare-ticker pattern (don't count these)
_TICKER_BLACKLIST = {
    "USA", "UK", "EU", "UN", "NATO", "OPEC", "BBC", "CNN", "FBI", "CIA",
    "AI", "ML", "API", "CEO", "CFO", "COO", "CTO", "VP", "PR", "TV",
    "MR", "MS", "DR", "JR", "SR", "PHD", "MD", "NEW", "OLD", "AGE",
    "ONE", "TWO", "TEN", "GOD", "NOT", "BUT", "AND", "OR", "FOR",
    "ARE", "WAS", "HAS", "HAD", "HIS", "HER", "ITS", "OUR", "OUT",
}


def financial_specificity(sent: str) -> int:
    """Count finance-lexicon hits + ticker-like tokens (modest weight)."""
    hits = len(_FIN_RE.findall(sent))
    # Light ticker bonus: at most +1 per sentence, only if it's not a blacklisted acronym
    tickers = [t for t in _TICKER_RE.findall(sent)
               if t.lstrip("$") not in _TICKER_BLACKLIST]
    if tickers:
        hits += 1
    return hits


# ─── Mapper architecture (residual FFN with linear injection) ────
class ResidualFFNMapper(nn.Module):
    """
    Target-conditioned projection mapper, matching the paper's Eq. (1)–(3):
      h0 = Ws @ z_s + Wt @ z_t + biases
      for each block: h <- h + FFN(LN(h))   where FFN = Linear → GELU → Linear
      h <- final_ln(h)
      out <- output_proj(h)
      out <- out_ln(out)
    """
    def __init__(self, input_dim: int, output_dim: int, memory_dim: int,
                 hidden_dim: int, n_blocks: int = 4, ffn_mult: int = 4):
        super().__init__()
        self.input_proj  = nn.Linear(input_dim, hidden_dim)
        self.memory_proj = nn.Linear(memory_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),                  # blocks.i.0
                nn.Linear(hidden_dim, hidden_dim * ffn_mult),  # blocks.i.1
                nn.GELU(),                                  # blocks.i.2 (no params)
                nn.Dropout(0.0),                            # blocks.i.3 (no params, eval mode)
                nn.Linear(hidden_dim * ffn_mult, hidden_dim),  # blocks.i.4
            )
            for _ in range(n_blocks)
        ])
        self.final_ln    = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.out_ln      = nn.LayerNorm(output_dim)

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        # Linear injection of target as coordinate anchor
        h = self.input_proj(z_s) + self.memory_proj(z_t)
        # Residual FFN refinement
        for block in self.blocks:
            h = h + block(h)
        h = self.final_ln(h)
        out = self.output_proj(h)
        out = self.out_ln(out)
        return out


def load_mapper(ckpt_path: Path, device: torch.device) -> ResidualFFNMapper:
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    hidden_dim, input_dim = sd["input_proj.weight"].shape
    _, memory_dim         = sd["memory_proj.weight"].shape
    output_dim, _         = sd["output_proj.weight"].shape
    # Count blocks via .0.weight (LayerNorm) entries
    n_blocks = len(set(int(k.split(".")[1]) for k in sd if k.startswith("blocks.")))
    mapper = ResidualFFNMapper(input_dim, output_dim, memory_dim, hidden_dim,
                                n_blocks=n_blocks)
    mapper.load_state_dict(sd)
    mapper.to(device).eval()
    console.log(f"  Mapper loaded: in={input_dim} mem={memory_dim} "
                f"out={output_dim} hidden={hidden_dim} blocks={n_blocks}")
    return mapper


# ─── Sentence quality + extraction ───────────────────────────────
def _quality_filter(s: str) -> bool:
    if not s:
        return False
    words = s.split()
    if not (SENTENCE_MIN_WORDS <= len(words) <= SENTENCE_MAX_WORDS):
        return False
    if not s[0].isupper() or s[-1] not in ".!?":
        return False
    if re.search(r"https?://|www\.|\.com|cookie|subscribe|copyright", s, re.IGNORECASE):
        return False
    if sum(1 for c in s if ord(c) > 127) / max(len(s), 1) > 0.05:
        return False
    return True


def _extract_sentences(text: str) -> List[str]:
    if not text or len(text) < 60:
        return []
    text = re.sub(r"\s+", " ", text)
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text[:4000])
    return [s.strip() for s in raw if _quality_filter(s.strip())]


# ─── Build stratified corpus ─────────────────────────────────────
def build_corpus():
    if PATH_SENTENCES.exists():
        console.log(f"[green]Loading cached corpus[/green] → {PATH_SENTENCES.name}")
        with open(PATH_SENTENCES) as f:
            return json.load(f)

    console.rule("[bold]Building domain-stratified corpus from cc_news[/bold]")
    from datasets import load_dataset

    pool: Dict[int, List[str]] = {}
    n_processed = 0
    n_kept = 0

    try:
        ds = iter(load_dataset("cc_news", split="train", streaming=True))
        for article in ds:
            text = article.get("text", "")
            for s in _extract_sentences(text):
                score = financial_specificity(s)
                pool.setdefault(score, []).append(s)
                n_kept += 1
            n_processed += 1
            if n_processed % 5000 == 0:
                counts = {sc: len(v) for sc, v in sorted(pool.items())}
                console.log(f"  processed {n_processed:,} articles, kept {n_kept:,} sentences. "
                            f"Score distribution (truncated): "
                            f"0={counts.get(0,0):,}  1={counts.get(1,0):,}  "
                            f"2={counts.get(2,0):,}  3+={sum(v for k,v in counts.items() if k>=3):,}")
            if n_kept >= CANDIDATE_TARGET and \
               len(pool.get(0, [])) >= SAMPLES_PER_ZONE * 5 and \
               sum(len(pool.get(k, [])) for k in [1, 2]) >= SAMPLES_PER_ZONE * 3 and \
               sum(len(pool.get(k, [])) for k in pool if k >= JARGON_THRESHOLD) >= SAMPLES_PER_ZONE * 2:
                break
    except Exception as e:
        console.log(f"[red]cc_news streaming failed: {e}[/red]")
        raise

    rng = random.Random(SEED)
    non_fin = pool.get(0, [])
    mixed   = [s for k in (1, 2) for s in pool.get(k, [])]
    jargon  = [s for k in pool if k >= JARGON_THRESHOLD for s in pool[k]]

    rng.shuffle(non_fin); rng.shuffle(mixed); rng.shuffle(jargon)

    # Deduplicate within each zone before sampling
    def dedup(lst):
        seen = set(); out = []
        for s in lst:
            if s not in seen:
                seen.add(s); out.append(s)
        return out

    non_fin, mixed, jargon = dedup(non_fin), dedup(mixed), dedup(jargon)

    take = SAMPLES_PER_ZONE
    sentences = {
        "non_financial":     non_fin[:take],
        "mixed":             mixed[:take],
        "financial_jargon":  jargon[:take],
    }

    for zone, sents in sentences.items():
        console.log(f"  {zone:18s}: {len(sents):>5} sentences")

    with open(PATH_SENTENCES, "w") as f:
        json.dump(sentences, f, indent=2)
    console.log(f"[green]Saved corpus → {PATH_SENTENCES}[/green]")
    return sentences


# ─── Embedding ───────────────────────────────────────────────────
@torch.no_grad()
def embed_with_hf(texts: List[str], hf_name: str, device: torch.device,
                  batch_size: int = 32, max_len: int = 128) -> torch.Tensor:
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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.cat(embs)


@torch.no_grad()
def embed_with_st(texts: List[str], hf_name: str, device: torch.device,
                  batch_size: int = 32) -> torch.Tensor:
    from sentence_transformers import SentenceTransformer
    mod = SentenceTransformer(hf_name, device="cpu").to(device)
    return mod.encode(texts, convert_to_tensor=True, device=device,
                      batch_size=batch_size, show_progress_bar=False).cpu()


def get_embeddings(sentences_by_zone: Dict[str, List[str]],
                   device: torch.device) -> Dict[str, torch.Tensor]:
    if PATH_EMB.exists():
        console.log(f"[green]Embeddings cached[/green] → {PATH_EMB.name}")
        return torch.load(PATH_EMB, map_location="cpu", weights_only=True)

    console.rule("[bold]Embedding sentences (FinBERT + GTR)[/bold]")
    out: Dict[str, torch.Tensor] = {}
    for zone, sents in sentences_by_zone.items():
        console.log(f"  {zone}: embedding {len(sents)} sentences with finbert …")
        out[f"src_{zone}"] = embed_with_hf(sents, SRC_HF, device)
        console.log(f"  {zone}: embedding {len(sents)} sentences with gtr …")
        out[f"tgt_{zone}"] = embed_with_st(sents, TGT_HF, device)
    torch.save(out, PATH_EMB)
    console.log(f"[green]Saved → {PATH_EMB.name}[/green]")
    return out


# ─── Displacement ───────────────────────────────────────────────
@torch.no_grad()
def casper_displacement(mapper: nn.Module, src_emb: torch.Tensor,
                        tgt_emb: torch.Tensor, device: torch.device,
                        batch_size: int = 512) -> np.ndarray:
    parts = []
    for i in range(0, len(src_emb), batch_size):
        s = src_emb[i:i+batch_size].to(device)
        t = tgt_emb[i:i+batch_size].to(device)
        mapped = mapper(s, t)
        parts.append((1 - F.cosine_similarity(mapped, t)).cpu())
    return torch.cat(parts).numpy()


def raw_displacement(src_emb: torch.Tensor, tgt_emb: torch.Tensor) -> np.ndarray:
    return (1 - F.cosine_similarity(src_emb, tgt_emb, dim=1)).numpy()


# ─── Stats ──────────────────────────────────────────────────────
def bootstrap_d(a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOT,
                seed: int = SEED) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n_a, n_b = len(a), len(b)
    ds = np.empty(n_boot)
    for i in range(n_boot):
        sa = a[rng.integers(0, n_a, n_a)]
        sb = b[rng.integers(0, n_b, n_b)]
        pool = np.sqrt((sa.std()**2 + sb.std()**2) / 2)
        ds[i] = (sb.mean() - sa.mean()) / pool if pool > 0 else 0.0
    return float(ds.mean()), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


def analyze_method(disp_by_zone: Dict[str, np.ndarray], method_name: str) -> dict:
    non_fin = disp_by_zone["non_financial"]
    mixed   = disp_by_zone["mixed"]
    jargon  = disp_by_zone["financial_jargon"]

    # Headline: non_financial vs financial_jargon
    d_mean, d_lo, d_hi = bootstrap_d(non_fin, jargon)
    u, p_mw = mannwhitneyu(non_fin, jargon, alternative="less")

    y_true  = np.concatenate([np.zeros(len(non_fin)), np.ones(len(jargon))])
    y_score = np.concatenate([non_fin, jargon])
    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)

    return {
        "method": method_name,
        "per_zone": {
            zone: {
                "n":      int(len(d)),
                "mean":   float(d.mean()),
                "median": float(np.median(d)),
                "std":    float(d.std()),
            }
            for zone, d in disp_by_zone.items()
        },
        "cohens_d_nf_vs_fj":      d_mean,
        "cohens_d_nf_vs_fj_ci":   [d_lo, d_hi],
        "mannwhitney_p_nf_vs_fj": float(p_mw),
        "auc_nf_vs_fj":           float(auc),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "disp_by_zone": {z: d.tolist() for z, d in disp_by_zone.items()},
    }


# ─── Figure ─────────────────────────────────────────────────────
def make_figure(results: Dict[str, dict], out_path: Path):
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.36, wspace=0.26)

    casper = results["CASPER"]
    raw    = results["raw_cosine"]
    method_colors = {"CASPER": "#c44e52", "raw_cosine": "#2e64a0"}
    zone_colors = {"non_financial": "#4c72b0",
                   "mixed":         "#dd8452",
                   "financial_jargon": "#c44e52"}
    zone_labels = {"non_financial":   "non-financial\n(score = 0)",
                   "mixed":           "mixed\n(score 1–2)",
                   "financial_jargon":"financial jargon\n(score ≥ 3)"}
    zone_order = ["non_financial", "mixed", "financial_jargon"]

    import pandas as pd

    # ── (A) CASPER violins by zone ───────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    rows = []
    for zone in zone_order:
        for d in casper["disp_by_zone"][zone]:
            rows.append({"zone": zone, "Δ": d})
    df = pd.DataFrame(rows)
    sns.violinplot(data=df, x="zone", y="Δ", ax=ax, order=zone_order,
                   palette=[zone_colors[z] for z in zone_order],
                   inner="quartile", cut=0)
    ax.set_xticklabels([zone_labels[z] for z in zone_order])
    ax.set_xlabel(""); ax.set_ylabel("CASPER displacement  Δ", fontsize=11)
    ax.set_title("(A)  Δ_CASPER by financial-specificity zone",
                 fontsize=13, fontweight="bold", loc="left")
    d_lo, d_hi = casper["cohens_d_nf_vs_fj_ci"]
    ax.text(0.02, 0.97,
            f"non-fin vs jargon:  d = {casper['cohens_d_nf_vs_fj']:+.3f}\n"
            f"95% CI [{d_lo:+.3f}, {d_hi:+.3f}]\n"
            f"AUC = {casper['auc_nf_vs_fj']:.3f}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", fc="#fff8e7", ec="#444"))

    # ── (B) Raw-cosine violins by zone (baseline) ────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    rows = []
    for zone in zone_order:
        for d in raw["disp_by_zone"][zone]:
            rows.append({"zone": zone, "Δ": d})
    df = pd.DataFrame(rows)
    sns.violinplot(data=df, x="zone", y="Δ", ax=ax2, order=zone_order,
                   palette=[zone_colors[z] for z in zone_order],
                   inner="quartile", cut=0)
    ax2.set_xticklabels([zone_labels[z] for z in zone_order])
    ax2.set_xlabel(""); ax2.set_ylabel("Raw-cosine displacement  Δ", fontsize=11)
    ax2.set_title("(B)  Same, with raw-cosine baseline (no alignment)",
                  fontsize=13, fontweight="bold", loc="left")
    d_lo, d_hi = raw["cohens_d_nf_vs_fj_ci"]
    ax2.text(0.02, 0.97,
             f"non-fin vs jargon:  d = {raw['cohens_d_nf_vs_fj']:+.3f}\n"
             f"95% CI [{d_lo:+.3f}, {d_hi:+.3f}]\n"
             f"AUC = {raw['auc_nf_vs_fj']:.3f}",
             transform=ax2.transAxes, va="top", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.4", fc="#fff8e7", ec="#444"))

    # ── (C) ECDF of CASPER Δ per zone ─────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    for zone in zone_order:
        data = np.sort(np.array(casper["disp_by_zone"][zone]))
        y = np.arange(1, len(data) + 1) / len(data)
        ax3.plot(data, y, lw=2.5, color=zone_colors[zone],
                 label=f"{zone_labels[zone].splitlines()[0]}  n={len(data)}")
    ax3.set_xlabel("CASPER displacement  Δ", fontsize=11)
    ax3.set_ylabel("Cumulative probability  P(Δ ≤ x)", fontsize=11)
    ax3.set_title("(C)  ECDF of Δ_CASPER per zone",
                  fontsize=13, fontweight="bold", loc="left")
    ax3.legend(loc="lower right", fontsize=9.5)

    # ── (D) ROC: discriminating jargon from non-financial ────────
    ax4 = fig.add_subplot(gs[1, 1])
    for method, info in results.items():
        ax4.plot(info["fpr"], info["tpr"], lw=2.5,
                 color=method_colors.get(method, "gray"),
                 label=f"{method}  (AUC = {info['auc_nf_vs_fj']:.3f})")
    ax4.plot([0, 1], [0, 1], ls="--", color="gray", lw=1, label="chance")
    ax4.set_xlabel("False positive rate", fontsize=11)
    ax4.set_ylabel("True positive rate", fontsize=11)
    ax4.set_title("(D)  ROC: jargon vs non-financial",
                  fontsize=13, fontweight="bold", loc="left")
    ax4.legend(loc="lower right", fontsize=10)

    fig.suptitle(
        "CASPER — Domain Specialization Detection\n"
        f"FinBERT (financial) → GTR (general)  ·  "
        f"N = {casper['per_zone']['non_financial']['n']:,} per zone",
        fontsize=14, fontweight="bold", y=0.995,
    )

    for fmt in ("pdf", "png"):
        p = out_path.with_suffix(f".{fmt}")
        fig.savefig(p, dpi=300, bbox_inches="tight")
        console.log(f"[green]Saved {p.name}[/green]")
    plt.close(fig)


# ─── Main ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    if args.device:
        global DEVICE
        DEVICE = torch.device(args.device)

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    console.rule("[bold blue]CASPER — Domain Specialization Application[/bold blue]")
    console.log(f"Device: {DEVICE}")
    console.log(f"src = {SRC_HF} (FinBERT, financial)")
    console.log(f"tgt = {TGT_HF} (GTR, general)")
    console.log(f"Mapper: {MAPPER_PATH.name}")

    # 1. Corpus
    sentences_by_zone = build_corpus()
    if any(len(v) < SAMPLES_PER_ZONE for v in sentences_by_zone.values()):
        console.log("[yellow]Some zone is below SAMPLES_PER_ZONE — proceeding with what's available.[/yellow]")

    # 2. Embeddings
    embeddings = get_embeddings(sentences_by_zone, DEVICE)

    # 3. Mapper
    console.rule("[bold]Computing displacements[/bold]")
    mapper = load_mapper(MAPPER_PATH, DEVICE)

    casper_disp = {}
    raw_disp = {}
    for zone in ["non_financial", "mixed", "financial_jargon"]:
        s = embeddings[f"src_{zone}"].float()
        t = embeddings[f"tgt_{zone}"].float()
        casper_disp[zone] = casper_displacement(mapper, s, t, DEVICE)
        raw_disp[zone]    = raw_displacement(s, t)
        console.log(f"  {zone:18s}: Δ_CASPER mean = {casper_disp[zone].mean():.5f}   "
                    f"Δ_raw mean = {raw_disp[zone].mean():.5f}")

    del mapper
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 4. Stats
    console.rule("[bold]Analyzing[/bold]")
    res_casper = analyze_method(casper_disp, "CASPER")
    res_raw    = analyze_method(raw_disp,    "raw_cosine")
    results = {"CASPER": res_casper, "raw_cosine": res_raw}

    # 5. Print summary
    tbl = Table(title="Domain specialization discrimination summary")
    tbl.add_column("Method", style="bold")
    tbl.add_column("Δ̄ non-fin", justify="right")
    tbl.add_column("Δ̄ mixed", justify="right")
    tbl.add_column("Δ̄ jargon", justify="right")
    tbl.add_column("Cohen's d", justify="right")
    tbl.add_column("95% CI", justify="center")
    tbl.add_column("AUC", justify="right")
    for k in ("CASPER", "raw_cosine"):
        r = results[k]
        ci = f"[{r['cohens_d_nf_vs_fj_ci'][0]:+.3f}, {r['cohens_d_nf_vs_fj_ci'][1]:+.3f}]"
        tbl.add_row(k,
                    f"{r['per_zone']['non_financial']['mean']:.5f}",
                    f"{r['per_zone']['mixed']['mean']:.5f}",
                    f"{r['per_zone']['financial_jargon']['mean']:.5f}",
                    f"{r['cohens_d_nf_vs_fj']:+.4f}",
                    ci,
                    f"{r['auc_nf_vs_fj']:.3f}")
    console.print(tbl)

    # 6. Save
    PATH_RESULTS.write_text(json.dumps(results, indent=2))
    console.log(f"[green]JSON → {PATH_RESULTS}[/green]")

    # 7. Figure
    make_figure(results, OUT_DIR / "casper_domain_figure")


if __name__ == "__main__":
    main()
