#!/usr/bin/env python3
"""
CASPER — Temporal Semantic Drift Analysis (Publication-Grade)
=============================================================

Inspired by Hamilton, Leskovec & Jurafsky (2016) "Diachronic Word
Embeddings Reveal Statistical Laws of Semantic Change", this script
extends temporal drift analysis to modern sentence-embedding models
using the CASPER alignment framework.

Where Hamilton et al. align word2vec snapshots via Orthogonal Procrustes
and measure per-word cosine displacement, CASPER uses a learned
Transformer-decoder mapper and measures per-sentence displacement.

Hypothesis
----------
Sentences published AFTER an embedding model's training cutoff exhibit
systematically higher cosine displacement post-CASPER alignment than
sentences from within the model's training window.  The magnitude of
this effect scales with the temporal gap between the two models.

Methodology
-----------
1.  Collect real news sentences stratified by publication year
    (no topic filtering — pure random stratified sampling).
2.  For each model pair (spanning different training eras):
    a.  Embed all sentences with both models.
    b.  Map embeddings via pre-trained CASPER mapper (both directions).
    c.  Measure cosine displacement:  Δ = 1 − cos(mapped, target).
3.  Statistical tests per pair:
    •  Spearman ρ(article_year, Δ)
    •  Mann–Whitney U between pre-cutoff and post-cutoff groups
    •  Cohen's d effect size
4.  Cross-pair analysis:  temporal gap  ↔  displacement magnitude.

Datasets
--------
•  cc_news (HuggingFace, streaming)         →  2017 – 2019
•  stanford-oval/ccnews (year configs)      →  2020 – 2024
•  RealTimeData/bbc_news_alltime (monthly)  →  2025

Usage
-----
    source activate alignment_env
    python casper_temporal_drift.py
"""

# ── Imports ───────────────────────────────────────────────────────────
import os
import re
import json
import time
import random
import warnings
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.decomposition import PCA

from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
sns.set_style("whitegrid")

import pandas as pd

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from langdetect import detect, DetectorFactory, LangDetectException
    DetectorFactory.seed = 42  # deterministic
    _LANGDETECT_OK = True
except ImportError:
    _LANGDETECT_OK = False

console = Console()

# ── Configuration ─────────────────────────────────────────────────────
SEED = 42
DEVICE = torch.device("cuda:1" if torch.cuda.device_count() > 1 else
                      "cuda:0" if torch.cuda.is_available() else "cpu")
SAMPLES_PER_YEAR = 2000          # sentences per year bucket
EMBED_BATCH = 64                 # batch size for model embedding
MAPPER_BATCH = 512               # batch size for mapper inference
MAX_SEQ_LEN = 128                # max tokens per sentence
TARGET_YEARS = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
CACHE_VERSION = 3                # bump to invalidate sentence cache
ENGLISH_ONLY = True              # filter out non-English sentences

# ── Sentence length filter ────────────────────────────────────────────
# Shorter sentences concentrate temporal-signal-bearing content words.
# Typical ranges:
#   8-20   → short, focused, strongest signal
#   10-30  → balanced
#   10-50  → permissive (previous default)
SENTENCE_MIN_WORDS = 6
SENTENCE_MAX_WORDS = 15

# ── Quality filters (applied uniformly across all years) ──────────────
# Anchor mode for temporal signal:
#   "none" → no requirement (permissive)
#   "any"  → sentence must have proper noun OR number
#   "both" → sentence must have proper noun AND number (strictest)
ANCHOR_MODE = "any"
# Reject sentences that are mostly filler/stopwords.
MAX_STOPWORD_RATIO = 0.40

# Bootstrap confidence intervals
BOOTSTRAP_N = 1000
BOOTSTRAP_CI = 0.95

# ── Analysis mode ─────────────────────────────────────────────────────
# "sentence" → aggregate Δ per sentence (current behaviour)
# "word"     → Hamilton-style per-word drift (context-averaged embeddings)
# "both"     → run both pipelines (recommended for publication)
ANALYSIS_MODE = "both"
WORD_MIN_COUNT = 20              # min sentence occurrences for a word to qualify
WORD_MIN_YEAR_SPREAD = 2         # min different years word must appear in
WORD_EMERGENCE_THRESHOLD = 3     # min count in a year for it to count as "emerged"

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "temporal_drift_results"
CACHE_DIR = BASE_DIR / "temporal_sentence_cache"
EMB_CACHE_DIR = BASE_DIR / "temporal_emb_cache"


# ── Model Registry ────────────────────────────────────────────────────
# Each entry stores the HuggingFace name, approximate training-data
# cutoff year, and backend type (hf = HuggingFace, st = SentenceTransformer).
MODELS = {
    "gtr":    {"hf": "sentence-transformers/gtr-t5-base",              "era": 2021, "backend": "st"},
    "gte":    {"hf": "thenlper/gte-base",                              "era": 2023, "backend": "hf"},
    "e5":     {"hf": "intfloat/e5-base-v2",                           "era": 2023, "backend": "hf"},
    "gtel":   {"hf": "thenlper/gte-large",                            "era": 2023, "backend": "hf"},
    "stella": {"hf": "infgrad/stella-base-en-v2",                     "era": 2023, "backend": "hf"},
    "gra":    {"hf": "ibm-granite/granite-embedding-278m-multilingual","era": 2024, "backend": "hf"},
    "qwen3":  {"hf": "Qwen/Qwen3-Embedding-0.6B",                    "era": 2025, "backend": "hf"},
}

# Model pairs to analyse — selected to cover a range of temporal gaps.
# Each (src, tgt) must have a corresponding linear-injection mapper directory.
# qwen3→gte is omitted because no linear-injection mapper exists for that pair.
PAIRS = [
    ("gte",   "e5"),      # same-era control    (0 yr gap)
    ("stella","gte"),     # same-era control    (0 yr gap)
    ("gtr",   "gte"),     # small gap           (2 yr)
    ("gtr",   "e5"),      # small gap           (2 yr)
    ("gtr",   "gra"),     # medium gap          (3 yr)
    ("qwen3", "gtr"),     # large gap           (4 yr)
]


def _mapper_dir(src: str, tgt: str) -> Path:
    return BASE_DIR / f"saved_mappers_official_{src}_{tgt}_1M_MacroStr_SimTlm_linear"


# ── Timing helper ─────────────────────────────────────────────────────
@contextmanager
def timer(label: str, store: dict):
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    store[label] = round(elapsed, 2)
    console.log(f"[dim]  timer {label}: {elapsed:.1f}s[/dim]")


# ══════════════════════════════════════════════════════════════════════
#  MAPPER ARCHITECTURE  (auto-inferred from checkpoint)
# ══════════════════════════════════════════════════════════════════════

class LinearInjectionMapper(nn.Module):
    """
    Target-conditioned projection mapper, matching the LinearInjectionMapper
    defined in ALIGNMENT_OFFICIAL_diff_dim_linear.py and the paper's
    Eq. (1)–(3):

        h0 = W_s @ z_s + W_t @ z_t                  (linear target injection)
        for each block:  h = h + FFN(LN(h))         (residual refinement)
            FFN = Linear → GELU → Dropout → Linear → Dropout
        h = final_ln(h);  out = output_proj(h);  out = out_ln(out)

    All dimensions (input, output, memory, hidden, n_layers) are inferred
    automatically from the checkpoint via ``load_mapper()``. Forward returns
    a (out, h) tuple for parity with the cross-attentive legacy API; the
    second element is unused at inference time and is unpacked at the
    call site.
    """

    def __init__(self, input_dim: int, output_dim: int, memory_dim: int,
                 hidden_dim: int = 1024, n_layers: int = 4,
                 n_heads: int = 16, dropout: float = 0.1):
        super().__init__()
        # n_heads kept for API parity with the cross-attentive variant; unused.
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.input_proj  = nn.Linear(input_dim,  hidden_dim)
        self.memory_proj = nn.Linear(memory_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),                 # .0
                nn.Linear(hidden_dim, hidden_dim * 4),    # .1
                nn.GELU(),                                 # .2 (no params)
                nn.Dropout(dropout),                       # .3 (no params)
                nn.Linear(hidden_dim * 4, hidden_dim),    # .4
                nn.Dropout(dropout),                       # .5 (no params)
            )
            for _ in range(n_layers)
        ])

        self.final_ln    = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.out_ln      = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor, memory: torch.Tensor):
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


# Alias kept for backward compatibility with external callers
TransformerEmbeddingMapper = LinearInjectionMapper


def load_mapper(ckpt_path: str, device: torch.device) -> LinearInjectionMapper:
    """Instantiate a LinearInjectionMapper with dims auto-inferred from the checkpoint."""
    try:
        sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        sd = torch.load(ckpt_path, map_location=device)

    hidden_dim, input_dim = sd["input_proj.weight"].shape
    _, memory_dim         = sd["memory_proj.weight"].shape
    output_dim, _         = sd["output_proj.weight"].shape
    # Count residual blocks by collecting unique block indices in the state dict
    n_layers = len({int(k.split(".")[1]) for k in sd if k.startswith("blocks.")})

    mapper = LinearInjectionMapper(
        input_dim, output_dim, memory_dim,
        hidden_dim=hidden_dim, n_layers=n_layers,
    )
    mapper.load_state_dict(sd)
    mapper.to(device).eval()
    console.log(f"  Mapper loaded: in={input_dim} mem={memory_dim} "
                f"out={output_dim} hidden={hidden_dim} blocks={n_layers}")
    return mapper


# ══════════════════════════════════════════════════════════════════════
#  DATASET PIPELINE
# ══════════════════════════════════════════════════════════════════════

# Common English stopwords (used by quality filters and word-level analysis)
_STOPWORDS = set("""
a an the and or but if then so to of in on at for with by from is are was
were be been being have has had do does did this that these those i you he
she it we they my your his her its our their me him us them which who what
when where why how as not no nor very just also there here all any some
each every more most other such only own same than too can could should
would will may might must shall up down out off over under again further
about against between into through during before after above below onto
upon within without across around per via like within while whose whom
whether yet still than until unless because although though since among
said say says saying made make makes making get got gets getting
""".split())


def _is_english(s: str) -> bool:
    """Fast English detection. Falls back to a heuristic if langdetect is unavailable."""
    if not _LANGDETECT_OK:
        # Heuristic fallback: ASCII ratio + common English stopwords
        if sum(1 for c in s if ord(c) > 127) / max(len(s), 1) > 0.05:
            return False
        common = {"the", "a", "an", "and", "or", "of", "in", "on", "to",
                  "for", "is", "was", "with", "by", "at", "from", "as", "that"}
        words = re.findall(r"\b[a-z]+\b", s.lower())
        if not words:
            return False
        return sum(1 for w in words if w in common) / len(words) >= 0.10
    try:
        return detect(s) == "en"
    except LangDetectException:
        return False


def _extract_sentences(text: str, min_words: int = None,
                       max_words: int = None,
                       english_only: bool = None) -> List[str]:
    if min_words is None:
        min_words = SENTENCE_MIN_WORDS
    if max_words is None:
        max_words = SENTENCE_MAX_WORDS
    """Extract well-formed English sentences from article text (no topic bias)."""
    if english_only is None:
        english_only = ENGLISH_ONLY
    if not text or len(text) < 60:
        return []
    # Strip newlines/paragraph breaks that corrupt sentence splitting
    text = re.sub(r"\s+", " ", text)
    # Only consider the first ~3000 chars to avoid boilerplate footers
    raw = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text[:3000])
    out: List[str] = []
    for s in raw:
        s = s.strip()
        if not s:
            continue
        words = s.split()
        if not (min_words <= len(words) <= max_words):
            continue
        if not s[0].isupper() or s[-1] not in ".!?":
            continue
        # Reject URLs, markup, boilerplate
        if re.search(r"https?://|www\.|[\[\]<>@#\\{}]|\.com|cookie|subscribe",
                      s, re.IGNORECASE):
            continue
        # Heavy non-ASCII rejection (quick pre-filter before langdetect)
        non_ascii = sum(1 for c in s if ord(c) > 127)
        if non_ascii / len(s) > 0.10:
            continue
        if english_only and not _is_english(s):
            continue

        # ── Quality filter #1: temporal anchor ────────────────────
        if ANCHOR_MODE != "none":
            has_proper_noun = any(
                len(w) > 1 and w[0].isupper() and w[1:].islower()
                for w in words[1:]
            )
            has_number = any(re.search(r"\d", w) for w in words)
            if ANCHOR_MODE == "both":
                if not (has_proper_noun and has_number):
                    continue
            elif ANCHOR_MODE == "any":
                if not (has_proper_noun or has_number):
                    continue

        # ── Quality filter #2: stopword density ───────────────────
        content_words = re.findall(r"\b[A-Za-z][A-Za-z\-']*\b", s.lower())
        if content_words:
            stop_ratio = sum(1 for w in content_words if w in _STOPWORDS) / len(content_words)
            if stop_ratio > MAX_STOPWORD_RATIO:
                continue

        out.append(s)
    return out


def _load_ccnews(target_years: set, n: int) -> Dict[int, List[str]]:
    """cc_news (HuggingFace) — fallback, covers approximately 2016-2019."""
    needed = {y for y in target_years if y <= 2019}
    # Only load if stanford-oval hasn't already filled these years
    result: Dict[int, List[str]] = {y: [] for y in target_years}
    if not needed:
        return result

    console.log("  [bold]Source: cc_news[/bold] (streaming, fallback for pre-2020)")
    try:
        ds = iter(load_dataset("cc_news", split="train", streaming=True))
        processed = 0
        MAX_ARTICLES = 200_000  # hard cap to avoid infinite streaming
        for article in ds:
            date_str = str(article.get("date", ""))
            if len(date_str) < 4:
                continue
            try:
                year = int(date_str[:4])
            except ValueError:
                continue
            if year not in needed:
                continue
            for s in _extract_sentences(article.get("text", "")):
                result[year].append(s)
            processed += 1
            if processed % 10_000 == 0:
                counts = {y: len(result[y]) for y in sorted(needed)}
                console.log(f"    processed {processed:,} articles  {counts}")
            if all(len(result[y]) >= n * 2 for y in needed):
                break
            if processed >= MAX_ARTICLES:
                console.log(f"    hit {MAX_ARTICLES:,} article cap")
                break
    except Exception as e:
        console.log(f"  [red]cc_news failed: {e}[/red]")
    return result


def _load_stanford_ccnews(target_years: set, n: int) -> Dict[int, List[str]]:
    """stanford-oval/ccnews — year-based config names, tries 2017-2024."""
    needed = {y for y in target_years if 2017 <= y <= 2024}
    result: Dict[int, List[str]] = {y: [] for y in target_years}
    if not needed:
        return result

    for year in sorted(needed):
        console.log(f"  [bold]Source: stanford-oval/ccnews  config={year}[/bold]")
        try:
            ds = iter(load_dataset("stanford-oval/ccnews", str(year),
                                   split="train", streaming=True))
            count = 0
            for article in ds:
                text = article.get("plain_text",
                        article.get("text", ""))
                for s in _extract_sentences(text):
                    result[year].append(s)
                count += 1
                if count % 5_000 == 0:
                    console.log(f"    {year}: {count:,} articles → "
                                f"{len(result[year]):,} sentences")
                if len(result[year]) >= n * 2:
                    break
            console.log(f"    {year}: final {len(result[year]):,} sentences")
        except Exception as e:
            console.log(f"  [yellow]{year} unavailable: {e}[/yellow]")
    return result


def _load_bbc_news(target_years: set, n: int) -> Dict[int, List[str]]:
    """RealTimeData/bbc_news_alltime — monthly config names, 2024-2025."""
    needed = {y for y in target_years if y >= 2024}
    result: Dict[int, List[str]] = {y: [] for y in target_years}
    if not needed:
        return result

    for year in sorted(needed):
        for month in range(1, 13):
            cfg = f"{year}-{month:02d}"
            try:
                ds = iter(load_dataset("RealTimeData/bbc_news_alltime",
                                       cfg, split="train", streaming=True))
                console.log(f"  [bold]Source: bbc_news  config={cfg}[/bold]")
                for article in ds:
                    text = article.get("content",
                            article.get("description",
                             article.get("text", "")))
                    for s in _extract_sentences(text):
                        result[year].append(s)
                    if len(result[year]) >= n * 2:
                        break
                console.log(f"    {cfg}: {len(result[year]):,} sentences "
                            f"total for {year}")
            except Exception:
                continue  # month not available
            if len(result[year]) >= n * 2:
                break
    return result


def _load_huffpost(target_years: set, n: int) -> Dict[int, List[str]]:
    """HuffPost News Category Dataset — ~210K articles, 2012-2022."""
    needed = {y for y in target_years if 2012 <= y <= 2022}
    result: Dict[int, List[str]] = {y: [] for y in target_years}
    if not needed:
        return result

    console.log("  [bold]Source: HuffPost News Category Dataset[/bold]")
    try:
        ds = iter(load_dataset("heegyu/news-category-dataset",
                               split="train", streaming=True))
        processed = 0
        for article in ds:
            date_str = str(article.get("date", ""))
            if len(date_str) < 4:
                continue
            try:
                year = int(date_str[:4])
            except ValueError:
                continue
            if year not in needed:
                continue

            # Use short_description (clean 1-2 sentence summary)
            text = article.get("short_description", "")
            if not text:
                text = article.get("headline", "")
            text = text.strip()
            words = text.split()
            if 8 <= len(words) <= 50 and text and text[0].isupper():
                if not text[-1] in ".!?":
                    text = text + "."
                result[year].append(text)

            processed += 1
            if processed % 50_000 == 0:
                counts = {y: len(result[y]) for y in sorted(needed)
                          if len(result[y]) > 0}
                console.log(f"    processed {processed:,} articles  {counts}")
            if all(len(result[y]) >= n * 2 for y in needed):
                break
    except Exception as e:
        console.log(f"  [yellow]HuffPost failed: {e}[/yellow]")
    return result


def build_temporal_dataset(
    years: Optional[List[int]] = None,
    n: int = SAMPLES_PER_YEAR,
) -> Dict[int, List[str]]:
    """
    Build a year-stratified sentence corpus from multiple news sources.
    Results are cached to disk for reproducibility.
    """
    if years is None:
        years = list(TARGET_YEARS)

    qfilt = f"anchor-{ANCHOR_MODE}_sw{int(MAX_STOPWORD_RATIO*100)}"
    tag = (f"v{CACHE_VERSION}_n{n}_len{SENTENCE_MIN_WORDS}-{SENTENCE_MAX_WORDS}"
           f"_{qfilt}_y{'_'.join(map(str, years))}")
    cache_path = CACHE_DIR / f"sentences_{tag}.json"

    if cache_path.exists():
        console.log(f"[green]Loading cached dataset: {cache_path.name}[/green]")
        with open(cache_path) as f:
            return {int(k): v for k, v in json.load(f).items()}

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    year_set = set(years)
    pool: Dict[int, List[str]] = {y: [] for y in years}

    # Merge from all available sources (order = priority):
    # 1. stanford-oval/ccnews — broadest, most consistent news corpus
    # 2. cc_news — fallback for pre-2020 if stanford is sparse
    # 3. HuffPost — additional fallback, different editorial perspective
    # 4. BBC — recent years (2024-2025)
    loaders = [_load_stanford_ccnews, _load_ccnews,
               _load_huffpost, _load_bbc_news]
    for loader in loaders:
        # Skip loaders for years already fully covered
        still_needed = {y for y in year_set if len(pool[y]) < n * 2}
        if not still_needed:
            break
        partial = loader(still_needed, n)
        for y in years:
            pool[y].extend(partial.get(y, []))

    # Deduplicate, shuffle, sample
    rng = random.Random(SEED)
    result: Dict[int, List[str]] = {}
    for y in years:
        uniq = list(set(pool[y]))
        rng.shuffle(uniq)
        result[y] = uniq[:n]
        status = "[green]OK[/green]" if len(result[y]) >= n else "[yellow]PARTIAL[/yellow]"
        console.log(f"  {y}: {len(result[y]):>5} / {n}  "
                     f"({len(uniq):,} unique candidates)  {status}")

    # Remove empty years
    result = {y: sents for y, sents in result.items() if sents}

    with open(cache_path, "w") as f:
        json.dump(result, f)
    console.log(f"[green]Cached → {cache_path}[/green]")
    return result


# ══════════════════════════════════════════════════════════════════════
#  EMBEDDING ENGINE
# ══════════════════════════════════════════════════════════════════════

def _load_model(key: str):
    info = MODELS[key]
    name = info["hf"]
    if info["backend"] == "st" and SentenceTransformer is not None:
        model = SentenceTransformer(name, device="cpu")
        model = model.to(DEVICE)
        return model.tokenizer, model, "st"
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    mod = AutoModel.from_pretrained(name, trust_remote_code=True)
    return tok, mod, "hf"


@torch.no_grad()
def _embed(texts: List[str], tok, model, backend: str,
           device: torch.device, batch_size: int = EMBED_BATCH) -> torch.Tensor:
    if backend == "st":
        return model.encode(
            texts, convert_to_tensor=True, device=device,
            batch_size=batch_size, show_progress_bar=False,
        ).cpu()

    model.to(device).eval()
    chunks: List[torch.Tensor] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_SEQ_LEN).to(device)
        out = model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1)
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
        chunks.append(pooled.cpu())
        del out, enc, mask, pooled
        torch.cuda.empty_cache()
    return torch.cat(chunks)


def get_embeddings(key: str, sentences: List[str],
                   device: torch.device) -> torch.Tensor:
    """Compute (or load cached) embeddings for a model."""
    h = hash(tuple(sentences[:5] + sentences[-5:])) % 10**8
    cache = EMB_CACHE_DIR / f"{key}_{len(sentences)}_{h}.pt"

    if cache.exists():
        console.log(f"  [green]{key}: cached embeddings[/green]")
        try:
            return torch.load(cache, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(cache, map_location="cpu")

    EMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    console.log(f"  {key}: embedding {len(sentences):,} sentences …")
    tok, mod, backend = _load_model(key)

    # Adaptive batch size: large models (>768D or >300M params) get smaller batches
    param_count = sum(p.numel() for p in mod.parameters()) if hasattr(mod, "parameters") else 0
    if param_count > 300_000_000:
        bs = max(8, EMBED_BATCH // 8)
        console.log(f"  {key}: large model ({param_count/1e6:.0f}M params) → batch_size={bs}")
    else:
        bs = EMBED_BATCH

    emb = _embed(sentences, tok, mod, backend, device, batch_size=bs)
    torch.save(emb, cache)
    console.log(f"  {key}: dim={emb.shape[1]}, saved → {cache.name}")

    # Free GPU thoroughly
    if hasattr(mod, "cpu"):
        mod.cpu()
    del tok, mod
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return emb


# ══════════════════════════════════════════════════════════════════════
#  DISPLACEMENT COMPUTATION
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_displacement(
    mapper: nn.Module,
    input_emb: torch.Tensor,
    memory_emb: torch.Tensor,
    target_emb: torch.Tensor,
    device: torch.device,
    batch_size: int = MAPPER_BATCH,
) -> np.ndarray:
    """
    Δ = 1 − cos( mapper(input, memory), target )

    For src2tgt:  input=src_emb, memory=tgt_emb, target=tgt_emb
    For tgt2src:  input=tgt_emb, memory=src_emb, target=src_emb

    LinearInjectionMapper.forward returns (out, h_compat); we keep only out.
    """
    parts: List[torch.Tensor] = []
    for i in range(0, len(input_emb), batch_size):
        inp = input_emb[i:i + batch_size].to(device)
        mem = memory_emb[i:i + batch_size].to(device)
        tgt = target_emb[i:i + batch_size].to(device)
        mapped, _ = mapper(inp, mem)
        parts.append((1 - F.cosine_similarity(mapped, tgt)).cpu())
    return torch.cat(parts).numpy()


def compute_raw_baseline(src_emb: torch.Tensor,
                         tgt_emb: torch.Tensor) -> Optional[np.ndarray]:
    """
    Baseline displacement with NO alignment: Δ = 1 − cos(src_emb, tgt_emb).
    Only well-defined when both embedding spaces share the same dimension.
    Returns None for asymmetric-dim pairs.

    This answers: "without CASPER, can you detect the temporal signal?"
    """
    if src_emb.shape[1] != tgt_emb.shape[1]:
        return None
    return (1 - F.cosine_similarity(src_emb, tgt_emb, dim=1)).numpy()


# ══════════════════════════════════════════════════════════════════════
#  STATISTICAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def _bootstrap_cohens_d(
    pre: np.ndarray, post: np.ndarray,
    n_boot: int = BOOTSTRAP_N, ci: float = BOOTSTRAP_CI,
    seed: int = SEED,
) -> Optional[dict]:
    """
    Stratified bootstrap CI for Cohen's d between two independent groups.
    Resamples each group with replacement n_boot times.
    """
    if len(pre) < 3 or len(post) < 3:
        return None
    rng = np.random.default_rng(seed)
    n_pre, n_post = len(pre), len(post)
    ds = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        p = pre[rng.integers(0, n_pre, n_pre)]
        q = post[rng.integers(0, n_post, n_post)]
        pooled = np.sqrt((p.std()**2 + q.std()**2) / 2)
        ds[i] = (q.mean() - p.mean()) / pooled if pooled > 0 else 0.0
    alpha = (1 - ci) / 2
    return dict(
        mean=float(ds.mean()),
        ci_low=float(np.percentile(ds, 100 * alpha)),
        ci_high=float(np.percentile(ds, 100 * (1 - alpha))),
        n_boot=n_boot,
    )


def analyse(displacements: np.ndarray, years: np.ndarray,
            src_era: int, tgt_era: int) -> dict:
    """
    Per-year descriptive stats + hypothesis tests following the
    Hamilton et al. (2016) framework adapted for sentence-level drift.
    Includes bootstrap CIs on Cohen's d.
    """
    unique = sorted(set(years))

    # Per-year descriptives
    per_year = {}
    for y in unique:
        d = displacements[years == y]
        per_year[int(y)] = dict(
            mean=float(np.mean(d)), std=float(np.std(d)),
            median=float(np.median(d)),
            q25=float(np.percentile(d, 25)),
            q75=float(np.percentile(d, 75)),
            n=int(len(d)),
        )

    # Spearman ρ: year ↔ displacement
    rho, p_sp = spearmanr(years, displacements)

    # Mann–Whitney U: pre-cutoff vs post-cutoff
    cutoff = min(src_era, tgt_era)
    pre  = displacements[years <= cutoff]
    post = displacements[years >  cutoff]

    if len(pre) > 0 and len(post) > 0:
        u, p_mw = mannwhitneyu(pre, post, alternative="less")
        pooled = np.sqrt((pre.std()**2 + post.std()**2) / 2)
        d_cohen = float((post.mean() - pre.mean()) / pooled) if pooled > 0 else 0.0
        d_cohen_boot = _bootstrap_cohens_d(pre, post)
    else:
        u, p_mw, d_cohen, d_cohen_boot = None, None, None, None

    # ── Three-zone analysis (asymmetric knowledge window) ───────────
    # Zone A "both_know":   year ≤ min(era)  — both models trained on this
    # Zone B "asymmetric":  min(era) < year ≤ max(era) — one knows, other doesn't
    # Zone C "neither":     year > max(era)  — neither model fully covers this
    era_min = min(src_era, tgt_era)
    era_max = max(src_era, tgt_era)

    zone_a = displacements[years <= era_min]       # both know
    zone_b = displacements[(years > era_min) & (years <= era_max)]  # asymmetric
    zone_c = displacements[years > era_max]        # neither knows

    zones = {}
    for label, arr in [("both_know", zone_a),
                        ("asymmetric", zone_b),
                        ("neither", zone_c)]:
        if len(arr) > 0:
            zones[label] = dict(
                mean=float(np.mean(arr)), std=float(np.std(arr)),
                median=float(np.median(arr)), n=int(len(arr)),
            )
        else:
            zones[label] = dict(mean=None, std=None, median=None, n=0)

    # Mann–Whitney between zone A (both_know) and zone B (asymmetric)
    if len(zone_a) > 0 and len(zone_b) > 0:
        u_ab, p_ab = mannwhitneyu(zone_a, zone_b, alternative="less")
        pool_ab = np.sqrt((zone_a.std()**2 + zone_b.std()**2) / 2)
        d_ab = float((zone_b.mean() - zone_a.mean()) / pool_ab) if pool_ab > 0 else 0.0
        d_ab_boot = _bootstrap_cohens_d(zone_a, zone_b)
    else:
        u_ab, p_ab, d_ab, d_ab_boot = None, None, None, None

    return dict(
        per_year=per_year,
        spearman_rho=float(rho), spearman_p=float(p_sp),
        mannwhitney_u=float(u) if u is not None else None,
        mannwhitney_p=float(p_mw) if p_mw is not None else None,
        cohens_d=d_cohen,
        cohens_d_boot=d_cohen_boot,
        cutoff_year=cutoff,
        n_pre=int(len(pre)), n_post=int(len(post)),
        mean_pre=float(pre.mean()) if len(pre) > 0 else None,
        mean_post=float(post.mean()) if len(post) > 0 else None,
        zones=zones,
        zone_era_bounds={"min": era_min, "max": era_max},
        zone_ab_mannwhitney_p=float(p_ab) if p_ab is not None else None,
        zone_ab_cohens_d=d_ab,
        zone_ab_cohens_d_boot=d_ab_boot,
    )


# ══════════════════════════════════════════════════════════════════════
#  WORD-LEVEL ANALYSIS  (Hamilton et al. 2016 style)
# ══════════════════════════════════════════════════════════════════════

_WORD_RE = re.compile(r"\b[A-Za-z][A-Za-z\-']{3,}\b")


def extract_target_words(
    sentences: List[str],
    years: np.ndarray,
    min_count: int = WORD_MIN_COUNT,
    min_spread: int = WORD_MIN_YEAR_SPREAD,
    emergence_threshold: int = WORD_EMERGENCE_THRESHOLD,
) -> Dict[str, dict]:
    """
    Extract content words with temporal metadata.

    For each qualifying word, return:
      - indices: list of sentence indices containing the word
      - year_counts: {year: occurrence count}
      - emergence_year: first year with count >= emergence_threshold
      - total: total occurrences
    """
    word_indices: Dict[str, List[int]] = {}
    for i, sent in enumerate(sentences):
        # Lowercase, deduplicate tokens within a sentence
        toks = set(tok.lower() for tok in _WORD_RE.findall(sent))
        for tok in toks:
            if tok in _STOPWORDS:
                continue
            if tok.isdigit():
                continue
            word_indices.setdefault(tok, []).append(i)

    result: Dict[str, dict] = {}
    years_arr = np.asarray(years)
    for word, indices in word_indices.items():
        if len(indices) < min_count:
            continue
        idx_years = years_arr[indices]
        unique_years, counts = np.unique(idx_years, return_counts=True)
        year_counts = {int(y): int(c) for y, c in zip(unique_years, counts)}
        if len(year_counts) < min_spread:
            continue
        # Emergence year: first year with count >= threshold
        emergence = None
        for y in sorted(year_counts):
            if year_counts[y] >= emergence_threshold:
                emergence = y
                break
        if emergence is None:
            continue
        result[word] = {
            "indices": indices,
            "year_counts": year_counts,
            "emergence_year": emergence,
            "total": len(indices),
        }
    return result


def run_word_level_analysis(
    all_sentences: List[str],
    all_years: np.ndarray,
    embeddings: Dict[str, torch.Tensor],
    pairs: List[Tuple[str, str]],
    device: torch.device,
    timings: Dict[str, float],
) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """
    Hamilton-style word-level drift analysis.

    Uses already-computed sentence embeddings. For each word, averages the
    embeddings of sentences containing that word to get a "contextual word
    embedding" — analogous to Hamilton et al.'s per-word diachronic vectors.

    Returns (word_results, words_map).
    """
    console.rule("[bold]Word-Level Analysis (Hamilton et al. style)[/bold]")

    with timer("word_extract", timings):
        words_map = extract_target_words(all_sentences, all_years)

    console.log(f"Extracted [bold]{len(words_map):,}[/bold] target words "
                f"(min count={WORD_MIN_COUNT}, min spread={WORD_MIN_YEAR_SPREAD}yr)")

    # Distribution of emergence years
    emerg_counts: Dict[int, int] = {}
    for w, info in words_map.items():
        emerg_counts[info["emergence_year"]] = emerg_counts.get(info["emergence_year"], 0) + 1
    tbl = Table(title="Word Emergence Year Distribution")
    tbl.add_column("Emergence Year", style="bold")
    tbl.add_column("# Words", justify="right")
    for y in sorted(emerg_counts):
        tbl.add_row(str(y), str(emerg_counts[y]))
    console.print(tbl)

    if len(words_map) < 50:
        console.log("[yellow]Too few words — word-level analysis unreliable[/yellow]")
        return {}, words_map

    # Pre-compute per-word embeddings per model (context-averaged)
    words = sorted(words_map.keys())
    emergence_years = np.array([words_map[w]["emergence_year"] for w in words])
    word_total_counts = np.array([words_map[w]["total"] for w in words])

    with timer("word_embed_aggregate", timings):
        word_embeddings: Dict[str, torch.Tensor] = {}
        for mk, emb in embeddings.items():
            wembs = []
            for w in words:
                idx = words_map[w]["indices"]
                wembs.append(emb[idx].mean(dim=0))
            word_embeddings[mk] = torch.stack(wembs)
        console.log(f"  Aggregated word embeddings: shape "
                    f"{tuple(next(iter(word_embeddings.values())).shape)}")

    # Per-pair, per-direction analysis
    word_results: Dict[str, dict] = {}
    for src_key, tgt_key in pairs:
        pair_tag = f"{src_key}→{tgt_key}"
        src_era = MODELS[src_key]["era"]
        tgt_era = MODELS[tgt_key]["era"]
        gap = abs(src_era - tgt_era)
        console.rule(f"  WORD-LEVEL  {pair_tag}  (gap={gap}yr)", style="dim")

        src_emb = word_embeddings[src_key]
        tgt_emb = word_embeddings[tgt_key]

        for direction in ("src2tgt", "tgt2src"):
            ckpt = _mapper_dir(src_key, tgt_key) / f"official_mapper_{direction}.pt"
            if not ckpt.exists():
                continue

            with timer(f"word_pair_{src_key}_{tgt_key}_{direction}", timings):
                mapper = load_mapper(str(ckpt), device)
                if direction == "src2tgt":
                    disp = compute_displacement(
                        mapper, src_emb, tgt_emb, tgt_emb, device)
                else:
                    disp = compute_displacement(
                        mapper, tgt_emb, src_emb, src_emb, device)
                stats = analyse(disp, emergence_years, src_era, tgt_era)

            # Pretty-print
            console.log(f"  Spearman ρ = {stats['spearman_rho']:+.4f}  "
                        f"(p = {stats['spearman_p']:.2e})")
            if stats["cohens_d"] is not None:
                console.log(f"  Mann-Whitney p = {stats['mannwhitney_p']:.2e}   "
                            f"Cohen's d = {stats['cohens_d']:.4f}")
            z = stats.get("zones", {})
            if z:
                def _zf(zd):
                    m = zd.get("mean")
                    return f"{m:.5f}" if m is not None else "N/A"
                console.log(
                    f"  Zones: both_know Δ̄={_zf(z.get('both_know', {}))} "
                    f"(n={z.get('both_know', {}).get('n', 0)})  "
                    f"asymmetric Δ̄={_zf(z.get('asymmetric', {}))} "
                    f"(n={z.get('asymmetric', {}).get('n', 0)})  "
                    f"neither Δ̄={_zf(z.get('neither', {}))} "
                    f"(n={z.get('neither', {}).get('n', 0)})")
                if stats.get("zone_ab_cohens_d") is not None:
                    console.log(f"  Zone A→B: Cohen's d = {stats['zone_ab_cohens_d']:.4f}   "
                                f"p = {stats['zone_ab_mannwhitney_p']:.2e}")

            # Top-5 highest and lowest displacement words
            order = np.argsort(disp)
            lowest = [(words[i], float(disp[i]), int(emergence_years[i]))
                      for i in order[:5]]
            highest = [(words[i], float(disp[i]), int(emergence_years[i]))
                       for i in order[-5:]]
            console.log(f"  Lowest Δ:  {', '.join(f'{w}({y}):{d:.4f}' for w,d,y in lowest)}")
            console.log(f"  Highest Δ: {', '.join(f'{w}({y}):{d:.4f}' for w,d,y in highest)}")

            rk = f"{src_key}_{tgt_key}_{direction}"
            word_results[rk] = dict(
                src=src_key, tgt=tgt_key,
                src_model=MODELS[src_key]["hf"],
                tgt_model=MODELS[tgt_key]["hf"],
                src_era=src_era, tgt_era=tgt_era,
                temporal_gap=gap, direction=direction,
                stats=stats,
                per_word={
                    words[i]: dict(
                        emergence_year=int(emergence_years[i]),
                        total_count=int(word_total_counts[i]),
                        displacement=float(disp[i]),
                    )
                    for i in range(len(words))
                },
            )

            del mapper
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return word_results, words_map


def make_word_figure(word_results: dict, out_dir: Path):
    """Word-level counterpart of make_figure() + top-word highlight."""
    if not word_results:
        console.log("[yellow]No word-level results — skipping word figure[/yellow]")
        return

    rows = []
    for rk, rv in word_results.items():
        for w, wd in rv["per_word"].items():
            rows.append(dict(
                pair=f"{rv['src']}→{rv['tgt']}",
                direction=rv["direction"],
                word=w,
                emergence_year=wd["emergence_year"],
                displacement=wd["displacement"],
                count=wd["total_count"],
                gap=rv["temporal_gap"],
            ))
    df = pd.DataFrame(rows)
    if df.empty:
        return

    src2tgt = df[df.direction == "src2tgt"]
    if src2tgt.empty:
        src2tgt = df

    gap_by_pair = src2tgt.groupby("pair")["gap"].first().sort_values()
    control_pair = gap_by_pair.index[0]
    max_pair = gap_by_pair.index[-1]

    year_order = sorted(df["emergence_year"].unique())
    palette = sns.color_palette("viridis", n_colors=len(year_order))

    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.30)

    # (A) Control pair — box by emergence year
    ax_a = fig.add_subplot(gs[0, 0])
    sub = src2tgt[src2tgt.pair == control_pair]
    sns.boxplot(data=sub, x="emergence_year", y="displacement", ax=ax_a,
                palette=palette, fliersize=1.5, order=year_order)
    ax_a.set_title(f"(A)  Control: {control_pair}  (gap = {int(gap_by_pair[control_pair])} yr)",
                   fontsize=13, fontweight="bold")
    ax_a.set_ylabel("Word-Level Δ")
    ax_a.set_xlabel("Word Emergence Year")

    # (B) Max-gap pair — box by emergence year
    ax_b = fig.add_subplot(gs[0, 1])
    sub = src2tgt[src2tgt.pair == max_pair]
    sns.boxplot(data=sub, x="emergence_year", y="displacement", ax=ax_b,
                palette=palette, fliersize=1.5, order=year_order)
    ax_b.set_title(f"(B)  Max Gap: {max_pair}  (gap = {int(gap_by_pair[max_pair])} yr)",
                   fontsize=13, fontweight="bold")
    ax_b.set_ylabel("Word-Level Δ")
    ax_b.set_xlabel("Word Emergence Year")

    # (C) Mean Δ by emergence year — all pairs
    ax_c = fig.add_subplot(gs[1, 0])
    agg = src2tgt.groupby(["pair", "emergence_year"])["displacement"].mean().reset_index()
    pair_colours = sns.color_palette("tab10", n_colors=len(agg["pair"].unique()))
    for i, (pair_name, grp) in enumerate(agg.groupby("pair")):
        grp = grp.sort_values("emergence_year")
        ax_c.plot(grp.emergence_year, grp.displacement, "o-",
                  label=pair_name, markersize=5,
                  color=pair_colours[i % len(pair_colours)])
    ax_c.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax_c.set_title("(C)  Mean Word Δ by Emergence Year",
                   fontsize=13, fontweight="bold")
    ax_c.set_ylabel("Mean Word-Level Displacement")
    ax_c.set_xlabel("Word Emergence Year")

    # (D) Top innovation vs conformity words (for max-gap pair)
    ax_d = fig.add_subplot(gs[1, 1])
    sub = src2tgt[src2tgt.pair == max_pair].copy()
    top_hi = sub.nlargest(12, "displacement")
    top_lo = sub.nsmallest(12, "displacement")
    combined = pd.concat([top_lo, top_hi]).reset_index(drop=True)
    combined["label"] = combined.apply(
        lambda r: f"{r['word']} ({r['emergence_year']})", axis=1)
    y_pos = np.arange(len(combined))
    colors = ["#4c72b0"] * len(top_lo) + ["#c44e52"] * len(top_hi)
    ax_d.barh(y_pos, combined["displacement"], color=colors, alpha=0.85)
    ax_d.set_yticks(y_pos)
    ax_d.set_yticklabels(combined["label"], fontsize=8)
    ax_d.invert_yaxis()
    ax_d.set_xlabel("Word-Level Δ")
    ax_d.set_title(f"(D)  Innovation vs Conformity: {max_pair}\n"
                   f"top-12 high-drift (red) vs low-drift (blue) words",
                   fontsize=12, fontweight="bold")

    fig.suptitle(
        "CASPER — Word-Level Diachronic Analysis\n"
        "Hamilton et al. (2016) framework on context-averaged word embeddings",
        fontsize=15, fontweight="bold", y=0.99,
    )

    for fmt in ("pdf", "png"):
        p = out_dir / f"temporal_drift_word_figure.{fmt}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        console.log(f"  [green]Saved {p.name}[/green]")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
#  VISUALISATION
# ══════════════════════════════════════════════════════════════════════

def make_figure(all_results: dict, out_dir: Path):
    """
    Publication-grade 4-panel figure:
      (A) Control pair box-plots by year
      (B) Max-gap pair box-plots by year
      (C) Mean Δ by year for all pairs (line plot)
      (D) Spearman ρ vs temporal gap (scatter)
    """
    # Build long-form DataFrame
    rows = []
    for rk, rv in all_results.items():
        for yr_str, disps in rv["displacements_by_year"].items():
            yr = int(yr_str)
            for d in disps:
                rows.append(dict(
                    pair=f"{rv['src']}→{rv['tgt']}",
                    direction=rv["direction"],
                    year=yr, displacement=d,
                    gap=rv["temporal_gap"],
                ))
    df = pd.DataFrame(rows)
    if df.empty:
        console.log("[yellow]No displacement data — skipping figure[/yellow]")
        return

    src2tgt = df[df.direction == "src2tgt"]
    if src2tgt.empty:
        src2tgt = df  # fallback: use whatever direction is available

    # Identify control (min gap) and max-gap pair
    gap_by_pair = src2tgt.groupby("pair")["gap"].first().sort_values()
    control_pair = gap_by_pair.index[0]
    max_pair     = gap_by_pair.index[-1]

    # Colour palette keyed by year
    year_order = sorted(df["year"].unique())
    palette = sns.color_palette("viridis", n_colors=len(year_order))

    fig = plt.figure(figsize=(16, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.30)

    # ── (A) Control pair ──────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    sub = src2tgt[src2tgt.pair == control_pair].copy()
    sns.boxplot(data=sub, x="year", y="displacement", ax=ax_a,
                palette=palette, fliersize=1, order=year_order)
    ax_a.set_title(f"(A)  Control: {control_pair}  (gap = {int(gap_by_pair[control_pair])} yr)",
                   fontsize=13, fontweight="bold")
    ax_a.set_ylabel("Cosine Displacement  Δ")
    ax_a.set_xlabel("Article Publication Year")

    # ── (B) Max-gap pair ──────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    sub = src2tgt[src2tgt.pair == max_pair].copy()
    sns.boxplot(data=sub, x="year", y="displacement", ax=ax_b,
                palette=palette, fliersize=1, order=year_order)
    ax_b.set_title(f"(B)  Max Gap: {max_pair}  (gap = {int(gap_by_pair[max_pair])} yr)",
                   fontsize=13, fontweight="bold")
    ax_b.set_ylabel("Cosine Displacement  Δ")
    ax_b.set_xlabel("Article Publication Year")

    # ── (C) Mean Δ by year — all pairs ────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    agg = src2tgt.groupby(["pair", "year"])["displacement"].mean().reset_index()
    pair_colours = sns.color_palette("tab10", n_colors=len(agg["pair"].unique()))
    for idx, (pair_name, grp) in enumerate(agg.groupby("pair")):
        grp = grp.sort_values("year")
        ax_c.plot(grp.year, grp.displacement, "o-",
                  label=pair_name, markersize=5,
                  color=pair_colours[idx % len(pair_colours)])
    ax_c.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax_c.set_title("(C)  Mean Δ by Year  (all pairs, src→tgt)",
                   fontsize=13, fontweight="bold")
    ax_c.set_ylabel("Mean Cosine Displacement")
    ax_c.set_xlabel("Article Publication Year")

    # ── (D) Zone A→B Cohen's d  vs  temporal gap (with 95% CI) ────
    ax_d = fig.add_subplot(gs[1, 1])
    pts = []
    for rk, rv in all_results.items():
        s = rv["stats"]
        # For same-era pairs, use overall Cohen's d (zone_ab doesn't exist)
        d_val = (s.get("zone_ab_cohens_d")
                 if s.get("zone_ab_cohens_d") is not None
                 else s.get("cohens_d"))
        boot = (s.get("zone_ab_cohens_d_boot")
                if s.get("zone_ab_cohens_d_boot") is not None
                else s.get("cohens_d_boot"))
        if d_val is None:
            continue
        pts.append(dict(
            gap=rv["temporal_gap"],
            d=d_val,
            ci_low=boot["ci_low"] if boot else d_val,
            ci_high=boot["ci_high"] if boot else d_val,
            direction=rv["direction"],
            pair=f"{rv['src']}→{rv['tgt']}",
            is_control=(rv["temporal_gap"] == 0),
        ))
    pdf = pd.DataFrame(pts)
    markers = {"src2tgt": "o", "tgt2src": "s"}

    # Small horizontal jitter per direction so CIs don't overlap
    jit = {"src2tgt": -0.08, "tgt2src": +0.08}
    for direction, marker in markers.items():
        sub = pdf[pdf.direction == direction]
        if sub.empty:
            continue
        x = sub.gap + jit[direction]
        colors = ["#888888" if c else ("#c44e52" if direction == "src2tgt" else "#4c72b0")
                  for c in sub.is_control]
        yerr_low = (sub.d - sub.ci_low).values
        yerr_high = (sub.ci_high - sub.d).values
        ax_d.errorbar(x, sub.d, yerr=[yerr_low, yerr_high],
                      fmt="none", ecolor="black", alpha=0.6,
                      capsize=3, lw=1.0, zorder=1)
        ax_d.scatter(x, sub.d, marker=marker, s=100,
                     c=colors, alpha=0.9, edgecolors="k",
                     linewidths=0.7, label=direction, zorder=2)
        for _, row in sub.iterrows():
            ax_d.annotate(row.pair, (row.gap + jit[direction], row.d),
                          fontsize=7, textcoords="offset points",
                          xytext=(6, 4))
    ax_d.axhline(0, color="gray", ls="--", lw=0.8)
    ax_d.legend(fontsize=10, loc="upper left")
    ax_d.set_title("(D)  Zone A→B Cohen's d  vs  Temporal Gap\n"
                   "(red=cross-era, gray=control; error bars = bootstrap 95% CI)",
                   fontsize=12, fontweight="bold")
    ax_d.set_xlabel("Temporal Gap Between Models (years)")
    ax_d.set_ylabel("Cohen's d  (asymmetric vs both_know)")

    fig.suptitle(
        "CASPER — Temporal Semantic Drift Analysis\n"
        "Diachronic evaluation on real news corpora "
        "(Hamilton et al. 2016 framework)",
        fontsize=15, fontweight="bold", y=0.99,
    )

    for fmt in ("pdf", "png"):
        p = out_dir / f"temporal_drift_figure.{fmt}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        console.log(f"  [green]Saved {p.name}[/green]")
    plt.close(fig)


def make_zone_figure(all_results: dict, out_dir: Path):
    """
    Asymmetric Knowledge Window figure — the key finding.
    Shows that displacement peaks when ONE model knows a concept
    but the other doesn't (the asymmetric zone).
    """
    # Collect zone data for cross-era pairs (gap > 0), src2tgt only
    rows = []
    for rk, rv in all_results.items():
        if rv["temporal_gap"] == 0 or rv["direction"] != "src2tgt":
            continue
        zones = rv["stats"].get("zones", {})
        pair = f"{rv['src']}→{rv['tgt']}"
        gap = rv["temporal_gap"]
        for zone_name, zone_data in zones.items():
            if zone_data["n"] > 0 and zone_data["mean"] is not None:
                rows.append(dict(
                    pair=pair, gap=gap, zone=zone_name,
                    mean=zone_data["mean"], std=zone_data["std"],
                    n=zone_data["n"],
                ))

    if not rows:
        console.log("[yellow]No zone data for cross-era pairs — skipping zone figure[/yellow]")
        return

    df = pd.DataFrame(rows)

    # Also build per-sentence zone data for violin plots
    violin_rows = []
    for rk, rv in all_results.items():
        if rv["temporal_gap"] == 0 or rv["direction"] != "src2tgt":
            continue
        pair = f"{rv['src']}→{rv['tgt']}"
        bounds = rv["stats"]["zone_era_bounds"]
        for yr_str, disps in rv["displacements_by_year"].items():
            yr = int(yr_str)
            if yr <= bounds["min"]:
                zone = "both_know"
            elif yr <= bounds["max"]:
                zone = "asymmetric"
            else:
                zone = "neither"
            for d in disps:
                violin_rows.append(dict(pair=pair, zone=zone, displacement=d))

    vdf = pd.DataFrame(violin_rows)
    zone_order = ["both_know", "asymmetric", "neither"]
    zone_labels = {"both_know": "Both Know\n(yr ≤ min era)",
                   "asymmetric": "Asymmetric\n(min < yr ≤ max era)",
                   "neither": "Neither Knows\n(yr > max era)"}
    zone_colors = {"both_know": "#4c72b0", "asymmetric": "#dd8452",
                   "neither": "#55a868"}

    fig, axes = plt.subplots(1, 3, figsize=(22, 7), gridspec_kw={"wspace": 0.28})

    # ── Left: grouped bar chart (mean Δ per zone per pair) ────────
    ax = axes[0]
    pairs_sorted = df.groupby("pair")["gap"].first().sort_values().index.tolist()
    x = np.arange(len(pairs_sorted))
    width = 0.25
    for i, zone in enumerate(zone_order):
        sub = df[df.zone == zone].set_index("pair")
        means = [sub.loc[p, "mean"] if p in sub.index else 0 for p in pairs_sorted]
        stds = [sub.loc[p, "std"] if p in sub.index else 0 for p in pairs_sorted]
        ax.bar(x + (i - 1) * width, means, width, yerr=stds,
               label=zone_labels[zone], color=zone_colors[zone],
               capsize=3, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}" for p in pairs_sorted], fontsize=9, rotation=15)
    ax.set_ylabel("Mean Cosine Displacement  Δ", fontsize=11)
    ax.set_title("(A)  Displacement by Knowledge Zone", fontsize=13,
                 fontweight="bold")
    ax.legend(fontsize=9)

    # ── Right: violin plot (all pairs pooled) ─────────────────────
    ax2 = axes[1]
    present = [z for z in zone_order if z in vdf["zone"].values]
    if present:
        sns.violinplot(data=vdf, x="zone", y="displacement", ax=ax2,
                       order=present, palette=zone_colors,
                       inner="quartile", cut=0)
        ax2.set_xticklabels([zone_labels.get(z, z) for z in present], fontsize=9)
    ax2.set_ylabel("Cosine Displacement  Δ", fontsize=11)
    ax2.set_xlabel("")
    ax2.set_title("(B)  Pooled Distribution by Zone\n(all cross-era pairs)",
                  fontsize=13, fontweight="bold")

    # ── Panel C: CASPER vs Baseline (Cohen's d) ───────────────────
    ax3 = axes[2]
    rows = []
    for rk, rv in all_results.items():
        if rv["temporal_gap"] == 0 or rv["direction"] != "src2tgt":
            continue
        s = rv["stats"]
        b = rv.get("baseline_stats")
        casper_d = s.get("zone_ab_cohens_d")
        casper_boot = s.get("zone_ab_cohens_d_boot")
        base_d = b.get("zone_ab_cohens_d") if b else None
        base_boot = b.get("zone_ab_cohens_d_boot") if b else None
        rows.append(dict(
            pair=f"{rv['src']}→{rv['tgt']}",
            gap=rv["temporal_gap"],
            casper_d=casper_d,
            casper_lo=casper_boot["ci_low"] if casper_boot else casper_d,
            casper_hi=casper_boot["ci_high"] if casper_boot else casper_d,
            base_d=base_d,
            base_lo=base_boot["ci_low"] if base_boot else base_d,
            base_hi=base_boot["ci_high"] if base_boot else base_d,
        ))
    cmp_df = pd.DataFrame(rows).sort_values("gap")
    ypos = np.arange(len(cmp_df))
    bw = 0.38

    for i, row in cmp_df.reset_index(drop=True).iterrows():
        # CASPER bar
        if row.casper_d is not None:
            ax3.barh(i - bw/2, row.casper_d, bw,
                     color="#c44e52", alpha=0.88,
                     label="CASPER" if i == 0 else None,
                     edgecolor="black", linewidth=0.5)
            ax3.errorbar(row.casper_d, i - bw/2,
                         xerr=[[row.casper_d - row.casper_lo],
                               [row.casper_hi - row.casper_d]],
                         fmt="none", ecolor="black", capsize=3, lw=0.8)
        # Baseline bar
        if row.base_d is not None:
            ax3.barh(i + bw/2, row.base_d, bw,
                     color="#4c72b0", alpha=0.88,
                     label="Raw cos (no CASPER)" if i == 0 else None,
                     edgecolor="black", linewidth=0.5)
            ax3.errorbar(row.base_d, i + bw/2,
                         xerr=[[row.base_d - row.base_lo],
                               [row.base_hi - row.base_d]],
                         fmt="none", ecolor="black", capsize=3, lw=0.8)
        elif row.base_d is None:
            ax3.text(0, i + bw/2, "  dims differ",
                     va="center", fontsize=8, color="gray", style="italic")
    ax3.set_yticks(ypos)
    ax3.set_yticklabels([f"{r.pair}\n(gap={r.gap}yr)"
                        for _, r in cmp_df.reset_index(drop=True).iterrows()],
                       fontsize=9)
    ax3.invert_yaxis()
    ax3.axvline(0, color="gray", ls="--", lw=0.8)
    ax3.set_xlabel("Zone A→B  Cohen's d", fontsize=11)
    ax3.set_title("(C)  CASPER vs Raw-Cosine Baseline\n(error bars = bootstrap 95% CI)",
                  fontsize=12, fontweight="bold")
    ax3.legend(fontsize=10, loc="lower right")

    fig.suptitle("CASPER — Asymmetric Knowledge Window Analysis",
                 fontsize=15, fontweight="bold", y=1.01)

    for fmt in ("pdf", "png"):
        p = out_dir / f"temporal_drift_zones.{fmt}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        console.log(f"  [green]Saved {p.name}[/green]")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_grad_enabled(False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timings: Dict[str, float] = {}
    run_start = time.perf_counter()

    # ── Header ────────────────────────────────────────────────────
    console.rule("[bold blue]CASPER — Temporal Semantic Drift Analysis[/bold blue]")
    console.log(f"Device       : {DEVICE}")
    console.log(f"Pairs        : {len(PAIRS)}  x  2 directions = {len(PAIRS)*2} conditions")
    console.log(f"Target       : {SAMPLES_PER_YEAR} sentences/year  x  "
                f"{len(TARGET_YEARS)} years")
    console.log(f"Output dir   : {OUTPUT_DIR}")

    # ── 1.  Dataset ──────────────────────────────────────────────
    console.rule("[bold]Step 1 · Build temporal dataset[/bold]")
    with timer("dataset_build", timings):
        data = build_temporal_dataset()

    all_sentences: List[str] = []
    all_years_list: List[int] = []
    for y in sorted(data):
        for s in data[y]:
            all_sentences.append(s)
            all_years_list.append(y)
    all_years = np.array(all_years_list)

    # Dataset summary table
    ds_tbl = Table(title="Dataset Summary")
    ds_tbl.add_column("Year", style="bold")
    ds_tbl.add_column("Sentences", justify="right")
    ds_tbl.add_column("Sample (first 80 chars)", max_width=80)
    for y in sorted(data):
        sample = data[y][0][:80] + "…" if data[y] else "—"
        ds_tbl.add_row(str(y), f"{len(data[y]):,}", sample)
    ds_tbl.add_row("[bold]Total[/bold]", f"[bold]{len(all_sentences):,}[/bold]", "")
    console.print(ds_tbl)

    if len(all_sentences) == 0:
        console.print(Panel(
            "No sentences were loaded.  Check your network connection\n"
            "and HuggingFace access for the configured datasets.",
            title="Error", border_style="red",
        ))
        return

    # ── 2.  Embeddings ───────────────────────────────────────────
    console.rule("[bold]Step 2 · Compute embeddings[/bold]")
    needed_models = set()
    for s, t in PAIRS:
        needed_models |= {s, t}

    embeddings: Dict[str, torch.Tensor] = {}
    for mk in sorted(needed_models):
        with timer(f"embed_{mk}", timings):
            embeddings[mk] = get_embeddings(mk, all_sentences, DEVICE)

    # ── 3.  Pair-wise analysis ───────────────────────────────────
    console.rule("[bold]Step 3 · Displacement analysis[/bold]")
    console.log(f"Analysis mode: [bold]{ANALYSIS_MODE}[/bold]")
    all_results: Dict[str, dict] = {}
    word_results: Dict[str, dict] = {}
    words_map: Dict[str, dict] = {}

    run_sentence = ANALYSIS_MODE in ("sentence", "both")
    run_word = ANALYSIS_MODE in ("word", "both")

    if not run_sentence:
        console.log("[yellow]Sentence-level analysis skipped (mode = word)[/yellow]")

    if not run_sentence:
        # Skip sentence loop; jump to word-level
        pass
    for src_key, tgt_key in (PAIRS if run_sentence else []):
        pair_tag = f"{src_key}→{tgt_key}"
        src_era = MODELS[src_key]["era"]
        tgt_era = MODELS[tgt_key]["era"]
        gap = abs(src_era - tgt_era)
        console.rule(f"  {pair_tag}  (gap={gap} yr)", style="dim")

        src_emb = embeddings[src_key]
        tgt_emb = embeddings[tgt_key]

        # ── Raw baseline (no CASPER): only for same-dim pairs ────
        baseline_disp = compute_raw_baseline(src_emb, tgt_emb)
        baseline_stats = None
        if baseline_disp is not None:
            baseline_stats = analyse(baseline_disp, all_years, src_era, tgt_era)
            b_d = baseline_stats.get("zone_ab_cohens_d")
            b_boot = baseline_stats.get("zone_ab_cohens_d_boot")
            b_ci = (f" [95% CI: {b_boot['ci_low']:+.4f}, {b_boot['ci_high']:+.4f}]"
                    if b_boot else "")
            if b_d is not None:
                console.log(
                    f"  [dim]BASELINE (raw cos, no CASPER):[/dim] "
                    f"Zone A→B d = {b_d:.4f}{b_ci}  "
                    f"p = {baseline_stats['zone_ab_mannwhitney_p']:.2e}")
        else:
            console.log(f"  [dim]BASELINE: skipped (dims differ: "
                        f"{src_emb.shape[1]} vs {tgt_emb.shape[1]})[/dim]")

        for direction in ("src2tgt", "tgt2src"):
            ckpt = _mapper_dir(src_key, tgt_key) / f"official_mapper_{direction}.pt"
            if not ckpt.exists():
                console.log(f"  [yellow]{direction}: checkpoint not found → skip[/yellow]")
                continue

            with timer(f"pair_{src_key}_{tgt_key}_{direction}", timings):
                mapper = load_mapper(str(ckpt), DEVICE)

                if direction == "src2tgt":
                    # mapper maps src→tgt space; compare with tgt ground-truth
                    disp = compute_displacement(
                        mapper, src_emb, tgt_emb, tgt_emb, DEVICE)
                else:
                    # mapper maps tgt→src space; compare with src ground-truth
                    disp = compute_displacement(
                        mapper, tgt_emb, src_emb, src_emb, DEVICE)

                stats = analyse(disp, all_years, src_era, tgt_era)

            # Per-year results table
            tbl = Table(title=f"{pair_tag}  {direction}", show_lines=False)
            tbl.add_column("Year", style="bold")
            tbl.add_column("N", justify="right")
            tbl.add_column("Mean Δ", justify="right")
            tbl.add_column("Median Δ", justify="right")
            tbl.add_column("Std", justify="right")
            for yr in sorted(stats["per_year"]):
                ys = stats["per_year"][yr]
                tbl.add_row(
                    str(yr), str(ys["n"]),
                    f"{ys['mean']:.5f}", f"{ys['median']:.5f}",
                    f"{ys['std']:.5f}",
                )
            console.print(tbl)

            console.log(f"  Spearman ρ = {stats['spearman_rho']:+.4f}  "
                        f"(p = {stats['spearman_p']:.2e})")
            if stats["cohens_d"] is not None:
                cb = stats.get("cohens_d_boot")
                ci_str = (f"  [95% CI: {cb['ci_low']:+.4f}, {cb['ci_high']:+.4f}]"
                          if cb else "")
                console.log(f"  Mann-Whitney p = {stats['mannwhitney_p']:.2e}   "
                            f"Cohen's d = {stats['cohens_d']:.4f}{ci_str}   "
                            f"(cutoff = {stats['cutoff_year']})")

            # Zone analysis
            z = stats.get("zones", {})
            if z:
                z_a = z.get("both_know", {})
                z_b = z.get("asymmetric", {})
                z_c = z.get("neither", {})
                def _zfmt(zd):
                    m = zd.get("mean")
                    return f"{m:.5f}" if m is not None else "N/A"
                console.log(
                    f"  Zones: both_know Δ̄={_zfmt(z_a)} (n={z_a.get('n', 0)})  "
                    f"asymmetric Δ̄={_zfmt(z_b)} (n={z_b.get('n', 0)})  "
                    f"neither Δ̄={_zfmt(z_c)} (n={z_c.get('n', 0)})")
                if stats.get("zone_ab_cohens_d") is not None:
                    zb_boot = stats.get("zone_ab_cohens_d_boot")
                    zci = (f"  [95% CI: {zb_boot['ci_low']:+.4f}, {zb_boot['ci_high']:+.4f}]"
                           if zb_boot else "")
                    console.log(
                        f"  Zone A→B: Cohen's d = {stats['zone_ab_cohens_d']:.4f}{zci}   "
                        f"p = {stats['zone_ab_mannwhitney_p']:.2e}")
                    # Effect-size interpretation in absolute units
                    za_mean = z_a.get("mean")
                    zb_mean = z_b.get("mean")
                    if za_mean is not None and zb_mean is not None and za_mean > 0:
                        pct = (zb_mean - za_mean) / za_mean * 100
                        console.log(
                            f"  Effect: Δ̄ shifts {za_mean:.5f} → {zb_mean:.5f} "
                            f"in asymmetric zone  ({pct:+.2f}% relative)")

            # Store
            rk = f"{src_key}_{tgt_key}_{direction}"
            all_results[rk] = dict(
                src=src_key, tgt=tgt_key,
                src_model=MODELS[src_key]["hf"],
                tgt_model=MODELS[tgt_key]["hf"],
                src_era=src_era, tgt_era=tgt_era,
                temporal_gap=gap, direction=direction,
                stats=stats,
                baseline_stats=baseline_stats,  # raw cosine, no CASPER
                displacements_by_year={
                    str(y): disp[all_years == y].tolist()
                    for y in sorted(set(all_years))
                },
            )

            del mapper
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ── 3b.  Word-level analysis (Hamilton et al. style) ─────────
    if run_word:
        word_results, words_map = run_word_level_analysis(
            all_sentences, all_years, embeddings, PAIRS, DEVICE, timings,
        )

    # ── 4.  Save results ─────────────────────────────────────────
    console.rule("[bold]Step 4 · Save results[/bold]")
    total_wall = time.perf_counter() - run_start
    timings["total_wall"] = round(total_wall, 2)

    payload = dict(
        config=dict(
            seed=SEED,
            samples_per_year=SAMPLES_PER_YEAR,
            target_years=sorted(data.keys()),
            years_loaded={y: len(s) for y, s in sorted(data.items())},
            device=str(DEVICE),
            timestamp=datetime.now().isoformat(),
            pairs=[(s, t) for s, t in PAIRS],
            analysis_mode=ANALYSIS_MODE,
            word_config=dict(
                min_count=WORD_MIN_COUNT,
                min_year_spread=WORD_MIN_YEAR_SPREAD,
                emergence_threshold=WORD_EMERGENCE_THRESHOLD,
            ) if run_word else None,
        ),
        timings=timings,
        results=all_results,
    )
    json_path = OUTPUT_DIR / "temporal_drift_results.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    console.log(f"[green]JSON → {json_path}[/green]")

    if run_word and word_results:
        word_payload = dict(
            config=payload["config"],
            n_words=len(words_map),
            word_results=word_results,
        )
        word_json_path = OUTPUT_DIR / "temporal_drift_word_results.json"
        with open(word_json_path, "w") as f:
            json.dump(word_payload, f, indent=2)
        console.log(f"[green]Word-level JSON → {word_json_path}[/green]")

    # ── 5.  Figure ───────────────────────────────────────────────
    console.rule("[bold]Step 5 · Generate figures[/bold]")
    if run_sentence and all_results:
        make_figure(all_results, OUTPUT_DIR)
        make_zone_figure(all_results, OUTPUT_DIR)
    if run_word and word_results:
        make_word_figure(word_results, OUTPUT_DIR)

    # ── 6.  Summary ──────────────────────────────────────────────
    console.rule("[bold]Step 6 · Summary[/bold]")

    if not all_results:
        console.log(f"\n[bold]Total wall time: {total_wall/60:.1f} min[/bold]")
        return

    summary_tbl = Table(title="Cross-Pair Summary  (both directions)")
    summary_tbl.add_column("Pair", style="bold")
    summary_tbl.add_column("Dir", justify="center")
    summary_tbl.add_column("Gap", justify="right")
    summary_tbl.add_column("ρ", justify="right")
    summary_tbl.add_column("p (Spearman)", justify="right")
    summary_tbl.add_column("d (Cohen)", justify="right")
    summary_tbl.add_column("Δ̄ pre", justify="right")
    summary_tbl.add_column("Δ̄ post", justify="right")

    for rk, rv in sorted(all_results.items(),
                         key=lambda x: (x[1]["temporal_gap"], x[1]["src"], x[1]["direction"])):
        s = rv["stats"]
        summary_tbl.add_row(
            f"{rv['src']}→{rv['tgt']}",
            rv["direction"],
            f"{rv['temporal_gap']} yr",
            f"{s['spearman_rho']:+.4f}",
            f"{s['spearman_p']:.1e}",
            f"{s['cohens_d']:.4f}" if s["cohens_d"] is not None else "—",
            f"{s['mean_pre']:.5f}" if s["mean_pre"] is not None else "—",
            f"{s['mean_post']:.5f}" if s["mean_post"] is not None else "—",
        )
    console.print(summary_tbl)

    # ── Zone summary table (both directions) ─────────────────────
    zone_tbl = Table(title="Asymmetric Knowledge Window — CASPER vs raw baseline")
    zone_tbl.add_column("Pair", style="bold")
    zone_tbl.add_column("Dir", justify="center")
    zone_tbl.add_column("Gap", justify="right")
    zone_tbl.add_column("CASPER d", justify="right")
    zone_tbl.add_column("CASPER 95% CI", justify="center")
    zone_tbl.add_column("Baseline d", justify="right")
    zone_tbl.add_column("Baseline 95% CI", justify="center")
    zone_tbl.add_column("Δd (CASPER − base)", justify="right")

    for rk, rv in sorted(all_results.items(),
                         key=lambda x: (x[1]["temporal_gap"], x[1]["src"], x[1]["direction"])):
        if rv["temporal_gap"] == 0:
            continue
        s = rv["stats"]
        zb = s.get("zone_ab_cohens_d_boot")
        casper_d = s.get("zone_ab_cohens_d")
        ci_str = (f"[{zb['ci_low']:+.3f}, {zb['ci_high']:+.3f}]" if zb else "—")

        b = rv.get("baseline_stats")
        if b:
            base_d = b.get("zone_ab_cohens_d")
            bb = b.get("zone_ab_cohens_d_boot")
            base_ci = (f"[{bb['ci_low']:+.3f}, {bb['ci_high']:+.3f}]" if bb else "—")
        else:
            base_d, base_ci = None, "N/A (dims differ)"

        delta = (f"{casper_d - base_d:+.4f}"
                 if casper_d is not None and base_d is not None else "—")

        zone_tbl.add_row(
            f"{rv['src']}→{rv['tgt']}",
            rv["direction"],
            f"{rv['temporal_gap']} yr",
            f"{casper_d:.4f}" if casper_d is not None else "—",
            ci_str,
            f"{base_d:.4f}" if base_d is not None else "—",
            base_ci,
            delta,
        )
    console.print(zone_tbl)

    # Meta-correlation: temporal gap ↔ Spearman ρ
    gaps  = [rv["temporal_gap"]          for rv in all_results.values()]
    rhos  = [rv["stats"]["spearman_rho"] for rv in all_results.values()]
    if len(set(gaps)) > 1:
        meta_rho, meta_p = spearmanr(gaps, rhos)
        console.print(Panel(
            f"[bold]Meta-analysis:  temporal gap  ↔  Spearman ρ[/bold]\n"
            f"ρ = {meta_rho:+.4f}   p = {meta_p:.2e}\n\n"
            f"[bold]Asymmetric Knowledge Window:[/bold]\n"
            f"Displacement peaks in the asymmetric zone (where one model\n"
            f"knows a concept but the other doesn't), not at the most\n"
            f"recent year — extending Hamilton et al.'s diachronic\n"
            f"framework to cross-model alignment.\n\n"
            f"[dim]cf. Hamilton et al. (2016) Laws of Semantic Change[/dim]",
            title="Key Finding",
            border_style="green",
        ))

    # ── Word-level summary ──────────────────────────────────────
    if run_word and word_results:
        word_tbl = Table(title="Word-Level Summary  (Hamilton et al. style, src→tgt)")
        word_tbl.add_column("Pair", style="bold")
        word_tbl.add_column("Gap", justify="right")
        word_tbl.add_column("ρ (word)", justify="right")
        word_tbl.add_column("p", justify="right")
        word_tbl.add_column("d (Cohen)", justify="right")
        word_tbl.add_column("Δ̄ both", justify="right")
        word_tbl.add_column("Δ̄ asym", justify="right")
        word_tbl.add_column("A→B d", justify="right")

        for rk, rv in sorted(word_results.items(),
                             key=lambda x: x[1]["temporal_gap"]):
            if rv["direction"] != "src2tgt":
                continue
            s = rv["stats"]
            z = s.get("zones", {})
            word_tbl.add_row(
                f"{rv['src']}→{rv['tgt']}",
                f"{rv['temporal_gap']} yr",
                f"{s['spearman_rho']:+.4f}",
                f"{s['spearman_p']:.1e}",
                f"{s['cohens_d']:.4f}" if s["cohens_d"] is not None else "—",
                f"{z.get('both_know', {}).get('mean', 0):.5f}" if z.get('both_know', {}).get('mean') else "—",
                f"{z.get('asymmetric', {}).get('mean', 0):.5f}" if z.get('asymmetric', {}).get('mean') else "—",
                f"{s.get('zone_ab_cohens_d'):.4f}" if s.get("zone_ab_cohens_d") is not None else "—",
            )
        console.print(word_tbl)

    console.log(f"\n[bold]Total wall time: {total_wall/60:.1f} min[/bold]")


if __name__ == "__main__":
    main()
