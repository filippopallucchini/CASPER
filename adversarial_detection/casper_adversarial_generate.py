#!/usr/bin/env python3
"""
CASPER — Adversarial Application: Step 1, Generate (orig, clean, adversarial) triples.

Pipeline:
  1. Sample N test sentences from IMDB (sentiment-labelled, balanced).
  2. Run TextFooler against a BERT-base IMDB classifier → adversarial paraphrase.
  3. Generate a benign paraphrase via Pegasus.
  4. Save aligned triples (orig, clean, adv) for the analysis stage.

Outputs:
  casper_adversarial_cache/imdb_subset.json
  casper_adversarial_cache/attack_outputs.json
  casper_adversarial_cache/clean_paraphrases.json
  casper_adversarial_cache/triples.json   ← consumed by the analyzer

Usage:
  python casper_adversarial_generate.py --n-samples 1500 --device cuda:0

Resumable: if a cache file exists, the corresponding step is skipped.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import List, Dict, Tuple

# ── Force TensorFlow to CPU (TextFooler's USE constraint uses TF) ──
# This avoids XLA JIT failures from GPU-memory contention with PyTorch.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # PyTorch sees GPU 0
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")  # TF is CPU-only

import numpy as np
import torch

from rich.console import Console
from rich.progress import track

console = Console()

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "casper_adversarial_cache"
CACHE_DIR.mkdir(exist_ok=True)

# --- File paths (attack-tagged so different recipes don't overwrite) ---
ATTACK_NAME = os.environ.get("CASPER_ADV_ATTACK", "deepwordbug")
PATH_SUBSET    = CACHE_DIR / "imdb_subset.json"
PATH_ATTACK    = CACHE_DIR / f"attack_outputs_{ATTACK_NAME}.json"
PATH_PARAPH    = CACHE_DIR / f"clean_paraphrases_{ATTACK_NAME}.json"
PATH_TRIPLES   = CACHE_DIR / f"triples_{ATTACK_NAME}.json"

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


# ─── Step 1: sample IMDB sentences ───────────────────────────────
def sample_imdb(n_samples: int) -> List[Dict]:
    if PATH_SUBSET.exists():
        console.log(f"[green]Step 1 cached:[/green] {PATH_SUBSET.name}")
        return json.loads(PATH_SUBSET.read_text())

    from datasets import load_dataset
    console.rule("[bold]Step 1 · Sample IMDB[/bold]")
    ds = load_dataset("imdb", split="test", streaming=False)
    rng = random.Random(SEED)

    # Balanced sample, keep only short reviews to bound TextAttack runtime
    pos = [(i, r) for i, r in enumerate(ds) if r["label"] == 1 and 50 < len(r["text"]) < 600]
    neg = [(i, r) for i, r in enumerate(ds) if r["label"] == 0 and 50 < len(r["text"]) < 600]
    rng.shuffle(pos); rng.shuffle(neg)
    n_each = n_samples // 2
    selected = pos[:n_each] + neg[:n_each]
    rng.shuffle(selected)

    out = [{"id": i, "text": r["text"], "label": r["label"]} for i, r in selected]
    PATH_SUBSET.write_text(json.dumps(out, indent=2))
    console.log(f"  Saved {len(out)} sentences → {PATH_SUBSET.name}")
    return out


# ─── Step 2: TextFooler attacks ──────────────────────────────────
def run_attacks(samples: List[Dict], device: str) -> List[Dict]:
    if PATH_ATTACK.exists():
        existing = json.loads(PATH_ATTACK.read_text())
        if len(existing) > 0:
            console.log(f"[green]Step 2 cached:[/green] {PATH_ATTACK.name} ({len(existing)} attacks)")
            return existing
        console.log(f"[yellow]Step 2 cache empty, re-running[/yellow]")

    console.rule(f"[bold]Step 2 · Run {ATTACK_NAME} attacks[/bold]")
    from textattack.models.wrappers import HuggingFaceModelWrapper
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    VICTIM = "textattack/bert-base-uncased-imdb"
    console.log(f"Loading victim {VICTIM} on {device}…")
    tok = AutoTokenizer.from_pretrained(VICTIM)
    mod = AutoModelForSequenceClassification.from_pretrained(VICTIM).to(device).eval()
    wrapper = HuggingFaceModelWrapper(mod, tok)

    # Recipe selection: character-level attacks produce stronger encoder shifts
    # than word-substitution attacks, because they break tokenization differently
    # across encoders that use different tokenizers (e.g. BERT WordPiece vs T5 SentencePiece).
    if ATTACK_NAME == "textfooler":
        from textattack.attack_recipes import TextFoolerJin2019
        attack = TextFoolerJin2019.build(wrapper)
    elif ATTACK_NAME == "deepwordbug":
        from textattack.attack_recipes import DeepWordBugGao2018
        attack = DeepWordBugGao2018.build(wrapper)
    elif ATTACK_NAME == "pwws":
        from textattack.attack_recipes import PWWSRen2019
        attack = PWWSRen2019.build(wrapper)
    elif ATTACK_NAME == "textbugger":
        from textattack.attack_recipes import TextBuggerLi2018
        attack = TextBuggerLi2018.build(wrapper)
    elif ATTACK_NAME == "bert-attack":
        from textattack.attack_recipes import BERTAttackLi2020
        attack = BERTAttackLi2020.build(wrapper)
    else:
        raise ValueError(f"Unknown attack: {ATTACK_NAME}")
    console.log(f"Attack recipe: {ATTACK_NAME}")

    results = []
    failures = 0
    skipped = 0
    t0 = time.perf_counter()
    for i, s in enumerate(samples):
        try:
            atk = attack.attack(s["text"], s["label"])
            cls = type(atk).__name__
            if cls.startswith("Successful"):
                results.append({
                    "id": s["id"],
                    "label": s["label"],
                    "orig": s["text"],
                    "adv":  atk.perturbed_result.attacked_text.text,
                })
            elif cls.startswith("Failed"):
                failures += 1
            else:
                skipped += 1
        except Exception as e:
            console.log(f"  [yellow]error on id={s['id']}: {e}[/yellow]")
            skipped += 1

        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            eta = (len(samples) - i - 1) / rate
            console.log(f"  [{i+1}/{len(samples)}] succ={len(results)} fail={failures} skip={skipped}   "
                        f"rate={rate:.2f}/s   ETA={eta/60:.1f} min")
            # Periodic checkpoint
            PATH_ATTACK.write_text(json.dumps(results, indent=2))

    PATH_ATTACK.write_text(json.dumps(results, indent=2))
    console.log(f"  Final: {len(results)} successful attacks "
                f"({failures} failed, {skipped} skipped) → {PATH_ATTACK.name}")
    del mod, wrapper, attack
    torch.cuda.empty_cache()
    return results


# ─── Step 3: clean Pegasus paraphrases ───────────────────────────
def make_paraphrases(attack_rows: List[Dict], device: str) -> Dict[int, str]:
    if PATH_PARAPH.exists():
        existing = json.loads(PATH_PARAPH.read_text())
        if len(existing) > 0:
            console.log(f"[green]Step 3 cached:[/green] {PATH_PARAPH.name} ({len(existing)} paraphrases)")
            return existing
        console.log(f"[yellow]Step 3 cache empty, re-running[/yellow]")

    console.rule("[bold]Step 3 · Pegasus paraphrasing[/bold]")
    from transformers import PegasusForConditionalGeneration, PegasusTokenizer

    PARAPHRASER = "tuner007/pegasus_paraphrase"
    console.log(f"Loading {PARAPHRASER} on {device}…")
    tok = PegasusTokenizer.from_pretrained(PARAPHRASER)
    mod = PegasusForConditionalGeneration.from_pretrained(PARAPHRASER).to(device).eval()

    out = {}
    BATCH = 8
    t0 = time.perf_counter()
    texts = [r["orig"] for r in attack_rows]
    ids = [r["id"] for r in attack_rows]

    with torch.no_grad():
        for i in range(0, len(texts), BATCH):
            batch_texts = texts[i:i+BATCH]
            batch_ids = ids[i:i+BATCH]
            # truncate inputs to Pegasus max
            enc = tok(batch_texts, return_tensors="pt", padding=True,
                      truncation=True, max_length=60).to(device)
            generated = mod.generate(**enc, max_length=60, num_beams=4,
                                      num_return_sequences=1, temperature=1.0)
            decoded = tok.batch_decode(generated, skip_special_tokens=True)
            for idx, paraphrase in zip(batch_ids, decoded):
                out[idx] = paraphrase

            if (i // BATCH + 1) % 20 == 0:
                elapsed = time.perf_counter() - t0
                done = min(i + BATCH, len(texts))
                rate = done / elapsed
                eta = (len(texts) - done) / rate
                console.log(f"  [{done}/{len(texts)}] rate={rate:.2f}/s ETA={eta/60:.1f} min")
                PATH_PARAPH.write_text(json.dumps(out, indent=2))

    PATH_PARAPH.write_text(json.dumps(out, indent=2))
    console.log(f"  Saved {len(out)} paraphrases → {PATH_PARAPH.name}")
    del mod
    torch.cuda.empty_cache()
    return out


# ─── Step 4: build triples ───────────────────────────────────────
def build_triples(attack_rows: List[Dict], paraphrases: Dict) -> List[Dict]:
    if PATH_TRIPLES.exists():
        existing = json.loads(PATH_TRIPLES.read_text())
        if len(existing) > 0:
            console.log(f"[green]Step 4 cached:[/green] {PATH_TRIPLES.name} ({len(existing)} triples)")
            return existing
        console.log(f"[yellow]Step 4 cache empty, re-running[/yellow]")

    console.rule("[bold]Step 4 · Build triples[/bold]")
    triples = []
    # paraphrase keys may be str if loaded from JSON
    paraphrases = {str(k): v for k, v in paraphrases.items()}

    for r in attack_rows:
        sid = str(r["id"])
        if sid not in paraphrases:
            continue
        clean = paraphrases[sid].strip()
        if len(clean) < 5 or clean.lower() == r["orig"].lower():
            continue
        triples.append({
            "id": r["id"],
            "label": r["label"],
            "orig":  r["orig"],
            "clean": clean,
            "adv":   r["adv"],
        })

    PATH_TRIPLES.write_text(json.dumps(triples, indent=2))
    console.log(f"  Saved {len(triples)} triples → {PATH_TRIPLES.name}")
    return triples


# ─── Main ────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=1500)
    p.add_argument("--device",    type=str, default="cuda:0")
    p.add_argument("--step", choices=["sample", "attack", "paraphrase", "triples", "all"], default="all")
    args = p.parse_args()

    console.rule("[bold blue]CASPER — Adversarial dataset generation[/bold blue]")
    console.log(f"Device: {args.device}")
    console.log(f"Target samples: {args.n_samples}")

    if args.step in ("sample", "all"):
        samples = sample_imdb(args.n_samples)
    else:
        samples = json.loads(PATH_SUBSET.read_text()) if PATH_SUBSET.exists() else []

    if args.step in ("attack", "all"):
        attack_rows = run_attacks(samples, args.device)
    else:
        attack_rows = json.loads(PATH_ATTACK.read_text()) if PATH_ATTACK.exists() else []

    if args.step in ("paraphrase", "all"):
        paraphrases = make_paraphrases(attack_rows, args.device)
    else:
        paraphrases = json.loads(PATH_PARAPH.read_text()) if PATH_PARAPH.exists() else {}

    if args.step in ("triples", "all"):
        triples = build_triples(attack_rows, paraphrases)
    else:
        triples = json.loads(PATH_TRIPLES.read_text()) if PATH_TRIPLES.exists() else []

    console.rule("Summary")
    console.log(f"Sampled:           {len(samples):>5}")
    console.log(f"Successful attacks:{len(attack_rows):>5}")
    console.log(f"Paraphrases:       {len(paraphrases):>5}")
    console.log(f"Final triples:     {len(triples):>5}")


if __name__ == "__main__":
    main()
