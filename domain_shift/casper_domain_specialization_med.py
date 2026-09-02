#!/usr/bin/env python3
"""
CASPER — Domain Specialization Detection (MEDICAL) — V4.

Adapted from the financial-domain script. Methodology, stats, and
figure layout are the same; the *sentence sources* differ because
cc_news alone does not contain enough genuinely-medical content to
populate a jargon-dense zone.

Design (this is the deliberate methodological choice for medical)
-----------------------------------------------------------------
- non_medical  : cc_news, score=0  in medical lexicon → general
                 English news, no medical content.
- mixed        : cc_news, score 1–2 in medical lexicon → news with
                 light health/medical mentions.
- medical_jargon: PubMed abstracts (pubmed_qa.pqa_artificial),
                 score ≥ 3 in medical lexicon → dense biomedical text.

This introduces a known source-domain confound between
medical_jargon (PubMed) and the other zones (cc_news). It is the
right design here because:
  (a) the experimental hypothesis is precisely that BioBERT encodes
      biomedical-domain semantics that GTR misses,
  (b) PubMed is the corpus BioBERT was pretrained on, so it is the
      *intended* domain of specialization,
  (c) cc_news at any plausible lexicon threshold yields too few
      jargon-dense sentences to support a 2k-per-zone test.

V4 also keeps:
- Lexicon-content hash in cache filenames → automatic invalidation
  on lexicon edits.
- Sanity-print of 3 example sentences per zone after building.
- Robust BioBERT loader (use_fast→slow→BertTokenizer / AutoModel→
  AutoModelForSequenceClassification→BertModel) — though it is no
  longer required for dmis-lab/biobert-v1.1, it is kept for safety
  if you swap the encoder.
"""

import argparse
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple, Iterator

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

SRC_KEY = "biobert"
TGT_KEY = "gtr"
SRC_HF  = "dmis-lab/biobert-v1.1"          # ← modern config, no model_type issue
TGT_HF  = "sentence-transformers/gtr-t5-base"

BASE_DIR = Path(__file__).resolve().parent
MAPPER_PATH = BASE_DIR / "saved_mappers_official_biobert_gtr_1M_MacroStr_SimTlm_linear/official_mapper_src2tgt.pt"

CACHE_DIR = BASE_DIR / "casper_medical_cache"
CACHE_DIR.mkdir(exist_ok=True)
OUT_DIR = BASE_DIR / "temporal_drift_results"
OUT_DIR.mkdir(exist_ok=True)

# Stratification — identical to the financial run for the non_med/mixed side.
SAMPLES_PER_ZONE   = 2000
SENTENCE_MIN_WORDS = 8
SENTENCE_MAX_WORDS = 30
JARGON_THRESHOLD   = 3
# Safety caps — how many examples to scan before giving up
MAX_CCNEWS_ARTICLES = 700_000
MAX_PUBMED_SENTS    = 1_000_000

N_BOOT = 1000


# ─── Medical lexicon ─────────────────────────────────────────────
# Mirrors the financial-lexicon philosophy: broad-but-medical terms
# (patient, treatment, drug, disease, hospital, …). The score≥3
# threshold does the discrimination.
_MEDICAL_LEXICON_RAW = {
    # Broad-but-medical
    "patient", "patients", "doctor", "doctors", "physician", "physicians",
    "nurse", "nurses", "nursing",
    "hospital", "hospitals", "hospitalization", "hospitalisation",
    "hospitalized", "hospitalised",
    "clinic", "clinics", "clinical", "clinically",
    "medicine", "medications", "medication",
    "treatment", "treatments", "treated", "treating", "untreated",
    "therapy", "therapies", "therapeutic",
    "drug", "drugs",
    "disease", "diseases", "disorder", "disorders",
    "illness", "illnesses", "syndrome", "syndromes",
    "infection", "infections", "infectious", "infected",
    "inflammation", "inflammatory",
    "symptom", "symptoms", "symptomatic", "asymptomatic",
    "diagnosis", "diagnoses", "diagnosed", "diagnostic", "diagnostics",
    "prognosis", "prognostic",
    "vaccine", "vaccines", "vaccination", "vaccinations",
    "vaccinate", "vaccinated", "unvaccinated",
    "immunization", "immunisation",
    "surgery", "surgical", "surgeon", "surgeons", "operated",
    "prescription", "prescribed", "prescribing",
    "dose", "doses", "dosage", "dosing",
    "side effect", "side effects", "adverse effect", "adverse effects", "adverse reaction",
    "healthcare", "health care",
    "specialist", "specialists",
    "biomedical", "biotechnology",

    # Diseases & conditions
    "cancer", "cancers", "carcinoma", "tumor", "tumors", "tumour", "tumours",
    "malignancy", "malignant",
    "lymphoma", "leukemia", "leukaemia", "melanoma", "sarcoma",
    "neoplasm", "neoplasms", "neoplastic",
    "metastasis", "metastases", "metastatic", "metastasized",
    "diabetes", "diabetic", "type 1 diabetes", "type 2 diabetes",
    "hypertension", "hypertensive", "hypotension",
    "hyperlipidemia", "hypercholesterolemia",
    "hyperglycemia", "hypoglycemia",
    "asthma", "asthmatic",
    "copd", "chronic obstructive pulmonary disease",
    "pneumonia", "bronchitis", "tuberculosis", "emphysema",
    "stroke", "ischemic stroke", "hemorrhagic stroke",
    "myocardial infarction", "heart attack", "cardiac arrest",
    "heart failure", "heart disease",
    "arrhythmia", "atrial fibrillation", "tachycardia", "bradycardia",
    "alzheimer", "parkinson",
    "dementia", "epilepsy", "epileptic",
    "multiple sclerosis", "als", "amyotrophic lateral sclerosis",
    "hepatitis", "cirrhosis", "fatty liver disease", "jaundice",
    "hiv", "aids", "influenza", "flu",
    "covid-19", "covid", "sars-cov-2", "coronavirus", "long covid",
    "ebola", "malaria", "sepsis", "septic shock",
    "arthritis", "rheumatoid arthritis", "osteoarthritis",
    "osteoporosis",
    "anemia", "anaemia", "thrombosis", "embolism", "pulmonary embolism",
    "deep vein thrombosis", "dvt",
    "schizophrenia", "bipolar disorder", "ptsd", "depression", "anxiety disorder",
    "lupus", "ulcerative colitis", "fibromyalgia",
    "cystic fibrosis", "sickle cell",
    "chronic disease", "autoimmune disease", "autoimmune",
    "mental health", "mental illness",

    # Anatomy / physiology
    "cardiovascular", "pulmonary", "renal", "hepatic", "gastric",
    "intestinal", "cerebral", "cranial", "thoracic", "abdominal",
    "cutaneous", "subcutaneous",
    "coronary artery", "aorta", "aortic", "ventricle", "ventricular",
    "atrium", "atrial", "alveoli", "alveolar", "bronchi", "bronchial",
    "thyroid", "adrenal", "pancreas", "pancreatic",
    "neuron", "neuronal", "neurological", "neurodegenerative",
    "synapse", "synaptic", "axon",
    "cortex", "cortical", "hippocampus", "amygdala", "cerebellum",
    "myocardium", "myocardial",
    "immune system", "immune response", "immunity", "immune",
    "blood cell", "white blood cell", "red blood cell",
    "blood vessel", "blood vessels",
    "lymph node", "lymph nodes", "lymphatic",
    "bone marrow",

    # Pharmacology
    "antibiotic", "antibiotics", "antiviral", "antivirals",
    "antifungal", "antimicrobial", "antibacterial", "antiretroviral",
    "chemotherapy", "chemotherapeutic", "immunotherapy", "radiotherapy",
    "analgesic", "analgesics",
    "anesthetic", "anaesthetic", "anesthesia", "anaesthesia",
    "sedative", "sedation",
    "antibody", "antibodies", "monoclonal antibody",
    "antigen", "antigens",
    "antidepressant", "antidepressants",
    "anticoagulant", "anticoagulants",
    "antihypertensive", "statin", "statins",
    "beta-blocker", "beta-blockers", "ace inhibitor",
    "insulin", "corticosteroid", "corticosteroids", "steroid", "steroids",
    "opioid", "opioids",
    "nsaid", "nsaids", "ibuprofen", "acetaminophen", "paracetamol", "aspirin",
    "placebo", "pharmacokinetics", "pharmacodynamics",
    "intravenous", "intramuscular",
    "drug interaction", "drug interactions", "drug trial", "drug trials",
    "booster shot", "booster", "boosters",

    # Procedures
    "biopsy", "endoscopy", "colonoscopy", "mammography", "mammogram",
    "mri", "magnetic resonance imaging", "ct scan", "x-ray", "ultrasound",
    "transplant", "transplantation", "organ transplant",
    "dialysis", "intubation", "resuscitation", "cpr", "defibrillation",
    "catheter", "stent", "angioplasty", "bypass surgery",
    "lumbar puncture", "ventilator", "ventilators",
    "icu admission", "icu admissions",

    # Diagnostics / labs
    "biomarker", "biomarkers",
    "blood pressure", "blood test", "blood tests", "blood sample",
    "ecg", "ekg", "electrocardiogram", "eeg", "electroencephalogram",
    "pathology", "pathological", "histology", "histological", "cytology",
    "pathogen", "pathogens", "pathogenic",
    "virus", "viruses", "viral", "bacteria", "bacterial", "fungal",
    "genome", "genomic", "gene expression", "genetic mutation", "allele",
    "in vivo", "in vitro", "ex vivo",
    "laboratory test", "lab test", "lab tests",

    # Clinical concepts
    "lesion", "lesions",
    "chronic", "acute", "terminal", "benign",
    "comorbidity", "comorbidities",
    "etiology", "aetiology", "pathogenesis",
    "prevalence", "incidence", "mortality", "morbidity", "fatality rate",
    "clinical trial", "clinical trials",
    "randomized controlled trial", "randomised controlled trial",
    "double-blind", "placebo-controlled",
    "patient outcome", "patient outcomes",
    "remission", "relapse", "recurrence",
    "screening", "cancer screening",

    # Specialties
    "cardiology", "cardiologist", "oncology", "oncologist",
    "neurology", "neurologist",
    "pediatrics", "paediatrics", "pediatrician", "paediatrician", "pediatric",
    "geriatrics", "geriatric", "obstetrics",
    "gynecology", "gynaecology", "gynecologist",
    "dermatology", "dermatologist",
    "ophthalmology", "ophthalmologist",
    "radiology", "radiologist", "psychiatry", "psychiatrist", "psychiatric",
    "immunology", "immunological",
    "endocrinology", "endocrinologist",
    "pulmonology", "nephrology", "nephrologist",
    "gastroenterology", "gastroenterologist",
    "hematology", "haematology",
    "urology", "urologist",
    "epidemiology", "epidemiologist", "epidemiological",
    "rheumatology", "rheumatologist",

    # Healthcare practice
    "icu", "intensive care unit", "emergency room", "outpatient", "inpatient",
    "fda", "fda-approved", "fda approval", "ema", "cdc", "nih",
    "healthcare worker", "healthcare workers", "medical care",

    # Public health
    "epidemic", "pandemic", "outbreak", "outbreaks",
    "herd immunity", "case fatality rate",
    "infectious disease", "communicable disease",
    "vaccination rate", "transmission rate",
    "public health", "quarantine", "lockdown",
}

_MEDICAL_LEXICON = sorted(_MEDICAL_LEXICON_RAW, key=len, reverse=True)

# Hash includes sources so swapping cc_news↔pubmed also invalidates cache.
_CACHE_KEY = hashlib.sha1(
    ("|".join(sorted(_MEDICAL_LEXICON_RAW))
     + "||sources=ccnews+pubmedqa"
     + f"||thresh={JARGON_THRESHOLD}"
     + f"||N={SAMPLES_PER_ZONE}").encode("utf-8")
).hexdigest()[:10]

PATH_SENTENCES = CACHE_DIR / f"domain_sentences_v4_{_CACHE_KEY}.json"
PATH_EMB       = CACHE_DIR / f"embeddings_v4_{_CACHE_KEY}.pt"
PATH_RESULTS   = OUT_DIR / "casper_medical_results.json"


def _build_medical_regex(lexicon):
    escaped = [re.escape(term) for term in lexicon]
    pattern = r"(?<![a-z0-9])(" + "|".join(escaped) + r")(?![a-z0-9])"
    return re.compile(pattern, flags=re.IGNORECASE)


_MED_RE = _build_medical_regex(_MEDICAL_LEXICON)

_MED_UNIT_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|µg|ml|mmol|µmol|iu)\b"
    r"|\bmm\s?hg\b|\bmg/dl\b|\bmmol/l\b|\bng/ml\b|\bbpm\b",
    flags=re.IGNORECASE,
)


def medical_specificity(sent: str) -> int:
    sent_norm = sent.replace("\u2019", "'").replace("\u2018", "'")
    hits = len(_MED_RE.findall(sent_norm))
    if _MED_UNIT_RE.search(sent_norm):
        hits += 1
    return hits


# ─── Mapper architecture ─────────────────────────────────────────
class ResidualFFNMapper(nn.Module):
    def __init__(self, input_dim, output_dim, memory_dim,
                 hidden_dim, n_blocks=4, ffn_mult=4):
        super().__init__()
        self.input_proj  = nn.Linear(input_dim, hidden_dim)
        self.memory_proj = nn.Linear(memory_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * ffn_mult),
                nn.GELU(),
                nn.Dropout(0.0),
                nn.Linear(hidden_dim * ffn_mult, hidden_dim),
            )
            for _ in range(n_blocks)
        ])
        self.final_ln    = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.out_ln      = nn.LayerNorm(output_dim)

    def forward(self, z_s, z_t):
        h = self.input_proj(z_s) + self.memory_proj(z_t)
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
    n_blocks = len(set(int(k.split(".")[1]) for k in sd if k.startswith("blocks.")))
    mapper = ResidualFFNMapper(input_dim, output_dim, memory_dim, hidden_dim, n_blocks=n_blocks)
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


# ─── Source streams ──────────────────────────────────────────────
def _stream_ccnews_sentences() -> Iterator[Tuple[str, int]]:
    """Yield (sentence, medical_score) from cc_news."""
    from datasets import load_dataset
    ds = load_dataset("cc_news", split="train", streaming=True)
    n_articles = 0
    for art in ds:
        text = art.get("text", "")
        for s in _extract_sentences(text):
            yield s, medical_specificity(s)
        n_articles += 1
        if n_articles >= MAX_CCNEWS_ARTICLES:
            return


def _stream_pubmed_sentences() -> Iterator[Tuple[str, int]]:
    """
    Yield (sentence, medical_score) from PubMed abstracts.
    pubmed_qa.pqa_artificial has ~211k examples; the `context.contexts`
    field is a pre-split list of sentence-like strings from the abstract.
    """
    from datasets import load_dataset
    try:
        ds = load_dataset("pubmed_qa", "pqa_artificial", split="train", streaming=True)
    except Exception as e:
        console.log(f"[yellow]pubmed_qa load failed ({type(e).__name__}: {e}); "
                    f"falling back to scientific_papers/pubmed[/yellow]")
        ds = load_dataset("scientific_papers", "pubmed", split="train", streaming=True)
        for ex in ds:
            for s in _extract_sentences(ex.get("abstract", "")):
                yield s, medical_specificity(s)
        return

    n_yielded = 0
    for ex in ds:
        ctx = ex.get("context", None)
        sents: List[str] = []
        if isinstance(ctx, dict):
            sents = list(ctx.get("contexts", []) or [])
        elif isinstance(ctx, str):
            sents = _extract_sentences(ctx)
        for s in sents:
            s = s.strip()
            if _quality_filter(s):
                yield s, medical_specificity(s)
                n_yielded += 1
                if n_yielded >= MAX_PUBMED_SENTS:
                    return


# ─── Build stratified corpus (dual source) ───────────────────────
def build_corpus():
    if PATH_SENTENCES.exists():
        console.log(f"[green]Loading cached corpus[/green] → {PATH_SENTENCES.name} "
                    f"(key {_CACHE_KEY})")
        with open(PATH_SENTENCES) as f:
            return json.load(f)

    console.rule(f"[bold]Building dual-source corpus  (key {_CACHE_KEY}, |L|={len(_MEDICAL_LEXICON)})[/bold]")

    # ── Stage 1: cc_news for non_medical + mixed ─────────────────
    console.log("[bold cyan]Stage 1[/bold cyan]: cc_news → non_medical (score=0) + mixed (score=1,2)")
    cc_pool: Dict[int, List[str]] = {0: [], 1: [], 2: []}
    n_seen = 0
    t0 = time.time()
    for s, score in _stream_ccnews_sentences():
        n_seen += 1
        if score in cc_pool:
            cc_pool[score].append(s)
        if n_seen % 250_000 == 0:
            console.log(f"  [{time.time()-t0:5.0f}s] cc_news sentences scanned={n_seen:,}  "
                        f"zone0={len(cc_pool[0]):,}  zone1={len(cc_pool[1]):,}  "
                        f"zone2={len(cc_pool[2]):,}")
        if (len(cc_pool[0]) >= SAMPLES_PER_ZONE * 5
                and len(cc_pool[1]) + len(cc_pool[2]) >= SAMPLES_PER_ZONE * 3):
            console.log(f"[green]  cc_news quotas met after {n_seen:,} sentences.[/green]")
            break

    # ── Stage 2: pubmed for medical_jargon ───────────────────────
    console.log("[bold cyan]Stage 2[/bold cyan]: PubMed (pubmed_qa) → medical_jargon (score≥3)")
    pubmed_jargon: List[str] = []
    n_seen_pm = 0
    n_kept_pm = 0
    t1 = time.time()
    for s, score in _stream_pubmed_sentences():
        n_seen_pm += 1
        if score >= JARGON_THRESHOLD:
            pubmed_jargon.append(s)
            n_kept_pm += 1
        if n_seen_pm % 100_000 == 0:
            console.log(f"  [{time.time()-t1:5.0f}s] PubMed sentences scanned={n_seen_pm:,}  "
                        f"jargon (score≥{JARGON_THRESHOLD})={n_kept_pm:,}")
        if n_kept_pm >= SAMPLES_PER_ZONE * 3:
            console.log(f"[green]  PubMed jargon quota met after {n_seen_pm:,} sentences.[/green]")
            break

    # ── Stage 3: shuffle, dedup, take ────────────────────────────
    rng = random.Random(SEED)
    non_med = cc_pool[0]
    mixed   = cc_pool[1] + cc_pool[2]
    jargon  = pubmed_jargon

    rng.shuffle(non_med); rng.shuffle(mixed); rng.shuffle(jargon)

    def dedup(lst):
        seen, out = set(), []
        for s in lst:
            if s not in seen:
                seen.add(s); out.append(s)
        return out

    non_med, mixed, jargon = dedup(non_med), dedup(mixed), dedup(jargon)
    take = SAMPLES_PER_ZONE
    sentences = {
        "non_medical":    non_med[:take],
        "mixed":          mixed[:take],
        "medical_jargon": jargon[:take],
    }

    console.log(f"[bold]Final zone sizes[/bold]:")
    for zone, sents in sentences.items():
        marker = "" if len(sents) >= take else "  [yellow]← below target[/yellow]"
        console.log(f"  {zone:18s}: {len(sents):>5} sentences{marker}")

    # ── Stage 4: sanity-print examples ───────────────────────────
    console.rule("[bold]Zone examples (sanity check)[/bold]")
    for zone in ("non_medical", "mixed", "medical_jargon"):
        console.log(f"[bold cyan]{zone}[/bold cyan]:")
        for ex in sentences[zone][:3]:
            score = medical_specificity(ex)
            console.log(f"  [dim](score={score})[/dim] {ex[:200]}")

    min_zone_size = max(200, take // 10)
    too_small = [z for z, s in sentences.items() if len(s) < min_zone_size]
    if too_small:
        raise RuntimeError(
            f"Zones below minimum useful size ({min_zone_size}): {too_small}. "
            f"Check the source-streaming output above.")

    with open(PATH_SENTENCES, "w") as f:
        json.dump(sentences, f, indent=2)
    console.log(f"[green]Saved corpus → {PATH_SENTENCES}[/green]")
    return sentences


# ─── Robust HF loader (kept for safety) ──────────────────────────
def _robust_load_hf_encoder(hf_name: str):
    from transformers import (
        AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
        BertTokenizer, BertModel, BertConfig,
    )
    tok = None
    for use_fast in (True, False):
        try:
            tok = AutoTokenizer.from_pretrained(hf_name, use_fast=use_fast,
                                                 trust_remote_code=True)
            break
        except Exception as e:
            console.log(f"[yellow]Tokenizer (use_fast={use_fast}) failed: "
                        f"{type(e).__name__}[/yellow]")
    if tok is None:
        tok = BertTokenizer.from_pretrained(hf_name)

    model = None
    for ctor, name in [(AutoModel, "AutoModel"),
                       (AutoModelForSequenceClassification, "AutoModelForSeqCls")]:
        try:
            model = ctor.from_pretrained(hf_name, trust_remote_code=True)
            break
        except Exception as e:
            console.log(f"[yellow]{name} failed: {type(e).__name__}[/yellow]")
    if model is None:
        cfg = BertConfig.from_pretrained(hf_name)
        model = BertModel.from_pretrained(hf_name, config=cfg)
    return tok, model


# ─── Embedding ───────────────────────────────────────────────────
@torch.no_grad()
def embed_with_hf(texts, hf_name, device, batch_size=32, max_len=128):
    tok, mod = _robust_load_hf_encoder(hf_name)
    mod = mod.to(device).eval()
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
def embed_with_st(texts, hf_name, device, batch_size=32):
    from sentence_transformers import SentenceTransformer
    mod = SentenceTransformer(hf_name, device="cpu").to(device)
    return mod.encode(texts, convert_to_tensor=True, device=device,
                      batch_size=batch_size, show_progress_bar=False).cpu()


def get_embeddings(sentences_by_zone, device):
    if PATH_EMB.exists():
        console.log(f"[green]Embeddings cached[/green] → {PATH_EMB.name}")
        return torch.load(PATH_EMB, map_location="cpu", weights_only=True)
    console.rule("[bold]Embedding sentences (BioBERT + GTR)[/bold]")
    out = {}
    for zone, sents in sentences_by_zone.items():
        console.log(f"  {zone}: embedding {len(sents)} sentences with biobert …")
        out[f"src_{zone}"] = embed_with_hf(sents, SRC_HF, device)
        console.log(f"  {zone}: embedding {len(sents)} sentences with gtr …")
        out[f"tgt_{zone}"] = embed_with_st(sents, TGT_HF, device)
    torch.save(out, PATH_EMB)
    console.log(f"[green]Saved → {PATH_EMB.name}[/green]")
    return out


# ─── Displacement ───────────────────────────────────────────────
@torch.no_grad()
def casper_displacement(mapper, src_emb, tgt_emb, device, batch_size=512):
    parts = []
    for i in range(0, len(src_emb), batch_size):
        s = src_emb[i:i+batch_size].to(device)
        t = tgt_emb[i:i+batch_size].to(device)
        mapped = mapper(s, t)
        parts.append((1 - F.cosine_similarity(mapped, t)).cpu())
    return torch.cat(parts).numpy()


def raw_displacement(src_emb, tgt_emb):
    return (1 - F.cosine_similarity(src_emb, tgt_emb, dim=1)).numpy()


# ─── Stats ──────────────────────────────────────────────────────
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


def analyze_method(disp_by_zone, method_name):
    non_med = disp_by_zone["non_medical"]
    mixed   = disp_by_zone["mixed"]
    jargon  = disp_by_zone["medical_jargon"]

    d_mean, d_lo, d_hi = bootstrap_d(non_med, jargon)
    u, p_mw = mannwhitneyu(non_med, jargon, alternative="less")

    y_true  = np.concatenate([np.zeros(len(non_med)), np.ones(len(jargon))])
    y_score = np.concatenate([non_med, jargon])
    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)

    return {
        "method": method_name,
        "per_zone": {
            zone: {"n": int(len(d)), "mean": float(d.mean()),
                   "median": float(np.median(d)), "std": float(d.std())}
            for zone, d in disp_by_zone.items()
        },
        "cohens_d_nm_vs_mj":      d_mean,
        "cohens_d_nm_vs_mj_ci":   [d_lo, d_hi],
        "mannwhitney_p_nm_vs_mj": float(p_mw),
        "auc_nm_vs_mj":           float(auc),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "disp_by_zone": {z: d.tolist() for z, d in disp_by_zone.items()},
    }


# ─── Figure ─────────────────────────────────────────────────────
def make_figure(results, out_path: Path):
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.36, wspace=0.26)

    casper = results["CASPER"]
    raw    = results["raw_cosine"]
    method_colors = {"CASPER": "#c44e52", "raw_cosine": "#2e64a0"}
    zone_colors = {"non_medical":    "#4c72b0",
                   "mixed":          "#dd8452",
                   "medical_jargon": "#c44e52"}
    zone_labels = {"non_medical":    "non-medical\n(cc_news, score=0)",
                   "mixed":          "mixed\n(cc_news, score 1–2)",
                   "medical_jargon": "medical jargon\n(PubMed, score ≥ 3)"}
    zone_order = ["non_medical", "mixed", "medical_jargon"]

    import pandas as pd

    ax = fig.add_subplot(gs[0, 0])
    rows = [{"zone": z, "Δ": d} for z in zone_order for d in casper["disp_by_zone"][z]]
    df = pd.DataFrame(rows)
    sns.violinplot(data=df, x="zone", y="Δ", ax=ax, order=zone_order,
                   palette=[zone_colors[z] for z in zone_order],
                   inner="quartile", cut=0)
    ax.set_xticklabels([zone_labels[z] for z in zone_order])
    ax.set_xlabel(""); ax.set_ylabel("CASPER displacement  Δ", fontsize=11)
    ax.set_title("(A)  Δ_CASPER by medical-specificity zone",
                 fontsize=13, fontweight="bold", loc="left")
    d_lo, d_hi = casper["cohens_d_nm_vs_mj_ci"]
    ax.text(0.02, 0.97,
            f"non-med vs jargon:  d = {casper['cohens_d_nm_vs_mj']:+.3f}\n"
            f"95% CI [{d_lo:+.3f}, {d_hi:+.3f}]\n"
            f"AUC = {casper['auc_nm_vs_mj']:.3f}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", fc="#fff8e7", ec="#444"))

    ax2 = fig.add_subplot(gs[0, 1])
    rows = [{"zone": z, "Δ": d} for z in zone_order for d in raw["disp_by_zone"][z]]
    df = pd.DataFrame(rows)
    sns.violinplot(data=df, x="zone", y="Δ", ax=ax2, order=zone_order,
                   palette=[zone_colors[z] for z in zone_order],
                   inner="quartile", cut=0)
    ax2.set_xticklabels([zone_labels[z] for z in zone_order])
    ax2.set_xlabel(""); ax2.set_ylabel("Raw-cosine displacement  Δ", fontsize=11)
    ax2.set_title("(B)  Same, with raw-cosine baseline (no alignment)",
                  fontsize=13, fontweight="bold", loc="left")
    d_lo, d_hi = raw["cohens_d_nm_vs_mj_ci"]
    ax2.text(0.02, 0.97,
             f"non-med vs jargon:  d = {raw['cohens_d_nm_vs_mj']:+.3f}\n"
             f"95% CI [{d_lo:+.3f}, {d_hi:+.3f}]\n"
             f"AUC = {raw['auc_nm_vs_mj']:.3f}",
             transform=ax2.transAxes, va="top", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.4", fc="#fff8e7", ec="#444"))

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

    ax4 = fig.add_subplot(gs[1, 1])
    for method, info in results.items():
        ax4.plot(info["fpr"], info["tpr"], lw=2.5,
                 color=method_colors.get(method, "gray"),
                 label=f"{method}  (AUC = {info['auc_nm_vs_mj']:.3f})")
    ax4.plot([0, 1], [0, 1], ls="--", color="gray", lw=1, label="chance")
    ax4.set_xlabel("False positive rate", fontsize=11)
    ax4.set_ylabel("True positive rate", fontsize=11)
    ax4.set_title("(D)  ROC: jargon vs non-medical",
                  fontsize=13, fontweight="bold", loc="left")
    ax4.legend(loc="lower right", fontsize=10)

    fig.suptitle(
        "CASPER — Domain Specialization Detection (Medical)\n"
        f"BioBERT (biomedical) → GTR (general)  ·  "
        f"sources: cc_news + PubMed  ·  "
        f"N = {casper['per_zone']['non_medical']['n']:,} per zone",
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
    parser.add_argument("--clear-cache", action="store_true",
                        help="Delete ALL cache files before running.")
    args = parser.parse_args()
    if args.device:
        global DEVICE
        DEVICE = torch.device(args.device)
    if args.clear_cache:
        n = 0
        for p in CACHE_DIR.iterdir():
            if p.is_file():
                p.unlink(); n += 1
        console.log(f"[yellow]Cleared {n} cache files.[/yellow]")

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    console.rule("[bold blue]CASPER — Domain Specialization (Medical, V4 / dual-source)[/bold blue]")
    console.log(f"Device: {DEVICE}")
    console.log(f"src = {SRC_HF}")
    console.log(f"tgt = {TGT_HF}")
    console.log(f"Mapper: {MAPPER_PATH.name}")
    console.log(f"Lexicon: {len(_MEDICAL_LEXICON)} terms  ·  cache key: {_CACHE_KEY}")
    console.log(f"Sources: non_medical/mixed←cc_news  ·  medical_jargon←pubmed_qa")
    console.log(f"Cache files: {PATH_SENTENCES.name}, {PATH_EMB.name}")

    sentences_by_zone = build_corpus()
    embeddings = get_embeddings(sentences_by_zone, DEVICE)

    console.rule("[bold]Computing displacements[/bold]")
    mapper = load_mapper(MAPPER_PATH, DEVICE)

    casper_disp, raw_disp = {}, {}
    for zone in ["non_medical", "mixed", "medical_jargon"]:
        s = embeddings[f"src_{zone}"].float()
        t = embeddings[f"tgt_{zone}"].float()
        casper_disp[zone] = casper_displacement(mapper, s, t, DEVICE)
        raw_disp[zone]    = raw_displacement(s, t)
        console.log(f"  {zone:18s}: Δ_CASPER mean = {casper_disp[zone].mean():.5f}   "
                    f"Δ_raw mean = {raw_disp[zone].mean():.5f}")

    del mapper
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    console.rule("[bold]Analyzing[/bold]")
    res_casper = analyze_method(casper_disp, "CASPER")
    res_raw    = analyze_method(raw_disp,    "raw_cosine")
    results = {"CASPER": res_casper, "raw_cosine": res_raw}

    tbl = Table(title="Medical domain specialization discrimination summary")
    tbl.add_column("Method", style="bold")
    tbl.add_column("Δ̄ non-med", justify="right")
    tbl.add_column("Δ̄ mixed", justify="right")
    tbl.add_column("Δ̄ jargon", justify="right")
    tbl.add_column("Cohen's d", justify="right")
    tbl.add_column("95% CI", justify="center")
    tbl.add_column("AUC", justify="right")
    for k in ("CASPER", "raw_cosine"):
        r = results[k]
        ci = f"[{r['cohens_d_nm_vs_mj_ci'][0]:+.3f}, {r['cohens_d_nm_vs_mj_ci'][1]:+.3f}]"
        tbl.add_row(k,
                    f"{r['per_zone']['non_medical']['mean']:.5f}",
                    f"{r['per_zone']['mixed']['mean']:.5f}",
                    f"{r['per_zone']['medical_jargon']['mean']:.5f}",
                    f"{r['cohens_d_nm_vs_mj']:+.4f}",
                    ci,
                    f"{r['auc_nm_vs_mj']:.3f}")
    console.print(tbl)

    PATH_RESULTS.write_text(json.dumps(results, indent=2))
    console.log(f"[green]JSON → {PATH_RESULTS}[/green]")

    make_figure(results, OUT_DIR / "casper_medical_figure")


if __name__ == "__main__":
    main()