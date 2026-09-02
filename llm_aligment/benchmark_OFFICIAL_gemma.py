#!/usr/bin/env python3
"""
Neural Bridge benchmark — validates that translating the source's residual
stream into the target's space makes the TARGET speak like the SOURCE.

Adapted to: Llama-3.2-1B (source) -> Gemma-4-E4B-it (target).

Key changes vs. the original benchmark.py
-----------------------------------------
1. Architecture: LinearInjectionMapper, same as the training script.
2. bridge_gen unchanged in semantics (final norm before lm_head, no source
   KV-cache, granular token-by-token decode).
3. Source-fidelity metrics:
     - fidelity_gain     = sim_src_st - sim_tgt_st  (per-row)
     - hijack_win_rate   = P(sim_src_st > sim_tgt_st)
     - first_token_match = stitched first token == source first token
4. Bootstrap CIs on fidelity gain.

Gemma-4 compatibility fixes
---------------------------
A. tokenizer_config.json patch (extra_special_tokens: list -> dict on
   transformers v4); safe no-op on other tokenizers.
B. pad_token = eos_token fallback (causal-LM tokenizers usually have none).
C. _get_hidden_size(config) — reads from .hidden_size or .text_config
   .hidden_size (Gemma4Config / Gemma3nConfig).
D. _get_final_norm / _get_lm_head — walk known attribute paths so we find
   model.model.norm (Llama) or model.model.language_model.norm (Gemma 4
   multimodal wrapper). Same for lm_head.
E. _decode_logits — casts h to the model's parameter dtype (bf16) before
   norm and lm_head, fixing the fp32 mapper output / bf16 model mismatch.
F. .float() casts on hidden states before division by fp32 scales.
"""

import argparse
import json
import math
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

# --- 1. CONFIGURATION ---
CONSOLE      = Console()
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Mapper produced by ALIGNMENT_OFFICIAL_diff_dim_linear_DECODER_llama_gemma4.py
MAPPER_PATH  = "official_residual_mapper_llama321B_gemma4E4B_100e_2M_5l.pt"
SRC_ID       = "meta-llama/Llama-3.2-1B"
TGT_ID       = "google/gemma-4-E4B-it"
SIM_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# These must match the values used in the training script
MAPPER_LAYERS = 5
MAPPER_HEADS  = 32


# -----------------------------------------------------------------------------
# Gemma-4 / multimodal-wrapper compatibility helpers (same as the eval script)
# -----------------------------------------------------------------------------
def _patch_gemma_tokenizer_config_if_needed(name):
    """
    Workaround for the known transformers v4 incompatibility with the
    google/gemma-4-* tokenizers (extra_special_tokens stored as list).
    See https://huggingface.co/google/gemma-4-E4B-it/discussions/17
    Safe no-op for other tokenizers.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return
    try:
        config_path = hf_hub_download(name, "tokenizer_config.json")
    except Exception:
        return
    try:
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        if isinstance(cfg.get("extra_special_tokens"), list):
            cfg["extra_special_tokens"] = {}
            with open(config_path, 'w') as f:
                json.dump(cfg, f)
            CONSOLE.print(f"[yellow][*] Patched tokenizer_config.json for {name} "
                          f"(extra_special_tokens: list -> dict)[/yellow]")
    except Exception as e:
        CONSOLE.print(f"[red][!] Could not patch tokenizer_config.json for {name}: {e}[/red]")


def _get_hidden_size(config):
    """Read hidden_size from top-level or from .text_config (Gemma4Config)."""
    if hasattr(config, 'hidden_size') and config.hidden_size is not None:
        return config.hidden_size
    if hasattr(config, 'text_config') and config.text_config is not None:
        tc = config.text_config
        if hasattr(tc, 'hidden_size') and tc.hidden_size is not None:
            return tc.hidden_size
    raise AttributeError(
        f"Could not find hidden_size on {type(config).__name__}. "
        f"Tried .hidden_size and .text_config.hidden_size."
    )


def _walk_path(root, path):
    obj = root
    for attr in path:
        obj = getattr(obj, attr, None)
        if obj is None:
            return None
    return obj


def _get_final_norm(model):
    """Locate the final RMSNorm before lm_head across Llama / Gemma-4 paths."""
    for path in [
        ('model', 'norm'),
        ('model', 'language_model', 'norm'),
        ('language_model', 'model', 'norm'),
        ('language_model', 'norm'),
        ('text_model', 'norm'),
        ('text_model', 'model', 'norm'),
    ]:
        leaf = _walk_path(model, path)
        if isinstance(leaf, nn.Module):
            return leaf
    return None


def _get_lm_head(model):
    """Locate the lm_head across Llama / Gemma-4 paths."""
    for path in [
        ('lm_head',),
        ('language_model', 'lm_head'),
        ('text_model', 'lm_head'),
        ('model', 'lm_head'),
    ]:
        leaf = _walk_path(model, path)
        if isinstance(leaf, nn.Module):
            return leaf
    return None


# --- 2. ARCHITECTURE ---
class LinearInjectionMapper(nn.Module):
    """Same architecture as in the training script."""
    def __init__(self, input_dim, output_dim, memory_dim,
                 hidden_dim=1024, n_layers=4, n_heads=16, dropout=0.1):
        super().__init__()
        self.input_proj  = nn.Linear(input_dim,  hidden_dim)
        self.memory_proj = nn.Linear(memory_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout),
            )
            for _ in range(n_layers)
        ])
        self.final_ln    = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.out_ln      = nn.LayerNorm(output_dim)

    def forward(self, x, memory):
        if x.dim() == 3:      x      = x.squeeze(1)
        if memory.dim() == 3: memory = memory.squeeze(1)
        h = self.input_proj(x) + self.memory_proj(memory)
        for block in self.blocks:
            h = h + block(h)
        h   = self.final_ln(h)
        out = self.output_proj(h)
        out = self.out_ln(out)
        return out, h

# --- 3. THE TRIPLE-METRIC VALIDATOR ---
class NeuralBridgeValidator:
    def __init__(self):
        self.load_systems()
        self.sim_model = SentenceTransformer(SIM_MODEL_ID).to(DEVICE)

    def load_systems(self):
        CONSOLE.print("[bold cyan][*] Loading models & scales...[/bold cyan]")

        # Defensive: patch gemma-4-* tokenizer_config.json before loading.
        _patch_gemma_tokenizer_config_if_needed(SRC_ID)
        _patch_gemma_tokenizer_config_if_needed(TGT_ID)

        self.s_tok = AutoTokenizer.from_pretrained(SRC_ID, trust_remote_code=True)
        self.t_tok = AutoTokenizer.from_pretrained(TGT_ID, trust_remote_code=True)
        if self.s_tok.pad_token is None:
            self.s_tok.pad_token = self.s_tok.eos_token
        if self.t_tok.pad_token is None:
            self.t_tok.pad_token = self.t_tok.eos_token

        self.s_mod = AutoModelForCausalLM.from_pretrained(SRC_ID, trust_remote_code=True).to(DEVICE).eval()
        self.t_mod = AutoModelForCausalLM.from_pretrained(TGT_ID, trust_remote_code=True).to(DEVICE).eval()

        cp = torch.load(MAPPER_PATH, map_location=DEVICE)
        self.s_scale = cp['ss']
        self.t_scale = cp['ts']

        # Hidden sizes via the multimodal-aware helper.
        D_SRC = _get_hidden_size(self.s_mod.config)
        D_TGT = _get_hidden_size(self.t_mod.config)
        CONSOLE.print(f"[green][+] Detected D_SRC={D_SRC} (Llama), D_TGT={D_TGT} (Gemma 4)[/green]")

        cfg   = cp.get('config', {})
        hidden_dim = cfg.get('hidden_dim', max(D_SRC, D_TGT))
        n_layers   = cfg.get('n_layers',   MAPPER_LAYERS)
        n_heads    = cfg.get('n_heads',    MAPPER_HEADS)

        self.m_s2t = LinearInjectionMapper(
            input_dim=D_SRC, output_dim=D_TGT, memory_dim=D_TGT,
            hidden_dim=hidden_dim, n_layers=n_layers, n_heads=n_heads
        ).to(DEVICE)
        self.m_s2t.load_state_dict(cp['s2t'])
        self.m_s2t.eval()

    # ------------------------------------------------------------
    # Dtype-safe, path-aware norm + lm_head application.
    # Mapper output is fp32 while Llama / Gemma-4 weights are bf16, so we
    # cast h to the model's parameter dtype before calling norm/head.
    # apply_norm=False is used for source-side advancement in bridge_gen
    # (matches the original code, which skipped the source's final norm).
    # ------------------------------------------------------------
    def _decode_logits(self, h, model=None, apply_norm=True):
        if model is None:
            model = self.t_mod
        head = _get_lm_head(model)
        if head is None:
            raise AttributeError(
                f"Could not locate lm_head on {type(model).__name__}; "
                f"stitched generation is not possible for this backbone."
            )
        target_dtype = next(model.parameters()).dtype
        h_d = h.to(target_dtype)
        if apply_norm:
            norm = _get_final_norm(model)
            h_d = norm(h_d) if norm is not None else h_d
        return head(h_d)

    # ------------------------------------------------------------
    # Embedding similarity helper
    # ------------------------------------------------------------
    def get_sim(self, a, b):
        if not a or not b:
            return 0.0
        embs = self.sim_model.encode([a, b], convert_to_tensor=True)
        return util.cos_sim(embs[0], embs[1]).item()

    # ------------------------------------------------------------
    # bridge_gen
    # ------------------------------------------------------------
    @torch.no_grad()
    def bridge_gen(self, prompt, max_tokens=20):
        """
        Reference logic with final-norm integration:
          - granular decoding (token-by-token string concatenation),
          - no KV-cache on source (full-sequence recompute),
          - target's final RMSNorm applied before lm_head (via _decode_logits).
        """
        curr_src = self.s_tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
        out_tokens = []

        for _ in range(max_tokens):
            # 1. Source hidden state (last token)
            s_out = self.s_mod(curr_src, output_hidden_states=True)
            h_src = s_out.hidden_states[-1][:, -1, :]

            # 2. Target context (re-tokenized from source's decoded text)
            txt = self.s_tok.decode(curr_src[0], skip_special_tokens=True)
            t_in = self.t_tok(txt, return_tensors="pt").input_ids.to(DEVICE)
            h_tgt_ctx = self.t_mod(t_in, output_hidden_states=True).hidden_states[-1][:, -1, :]

            # 3. Neural Bridge Projection — .float() bridges bf16 model output
            #    with fp32 scales and mapper.
            p_v, _ = self.m_s2t(h_src.float() / self.s_scale,
                                memory=h_tgt_ctx.float() / self.t_scale)
            mapped = p_v * self.t_scale   # fp32

            # 4. Final norm + lm_head (dtype-cast inside _decode_logits)
            logits_t = self._decode_logits(mapped, model=self.t_mod, apply_norm=True)
            next_tgt = torch.argmax(logits_t, dim=-1)

            # 5. Append token to running output
            word = self.t_tok.decode(next_tgt)
            out_tokens.append(word)

            # 6. Advance source for next step (apply_norm=False mirrors original)
            src_logits = self._decode_logits(h_src, model=self.s_mod, apply_norm=False)
            next_src = torch.argmax(src_logits, dim=-1)
            curr_src = torch.cat([curr_src, next_src.unsqueeze(0)], dim=1)

            if next_src.item() == self.s_tok.eos_token_id:
                break

        return "".join(out_tokens).strip()

    # ------------------------------------------------------------
    # First-token comparison helper
    # ------------------------------------------------------------
    @staticmethod
    def first_word(text):
        if not text:
            return ""
        return text.strip().split()[0].lower() if text.strip() else ""

    # ------------------------------------------------------------
    # Per-dataset evaluation
    # ------------------------------------------------------------
    def run_eval(self, name, subset, split, limit=200):
        ds = load_dataset(name, subset, split=split, streaming=True)
        rows = []
        for item in tqdm(ds, total=limit, desc=f"Evaluating {name}"):
            if len(rows) >= limit:
                break
            q = item.get('question', item.get('query', ''))
            if not q:
                continue
            prompt = f"Question: {q}\nAnswer:"

            with torch.no_grad():
                # 1. Source original
                s_in  = self.s_tok(prompt, return_tensors="pt").to(DEVICE)
                s_out = self.s_mod.generate(**s_in, max_new_tokens=30, do_sample=False)
                ans_s = self.s_tok.decode(s_out[0][s_in.input_ids.shape[1]:],
                                          skip_special_tokens=True).strip()

                # 2. Target original
                t_in  = self.t_tok(prompt, return_tensors="pt").to(DEVICE)
                t_out = self.t_mod.generate(**t_in, max_new_tokens=30, do_sample=False)
                ans_t = self.t_tok.decode(t_out[0][t_in.input_ids.shape[1]:],
                                          skip_special_tokens=True).strip()

                # 3. Stitched (bridged)
                ans_st = self.bridge_gen(prompt, max_tokens=30)

            sim_src_st  = self.get_sim(ans_s, ans_st)
            sim_src_tgt = self.get_sim(ans_s, ans_t)
            sim_tgt_st  = self.get_sim(ans_t, ans_st)

            rows.append({
                "dataset":           name,
                "question":          q,
                "ans_source":        ans_s,
                "ans_target":        ans_t,
                "ans_stitched":      ans_st,
                "sim_src_st":        sim_src_st,
                "sim_src_tgt":       sim_src_tgt,
                "sim_tgt_st":        sim_tgt_st,
                "fidelity_gain":     sim_src_st - sim_tgt_st,
                "first_token_match": int(self.first_word(ans_st) == self.first_word(ans_s)
                                         and self.first_word(ans_s) != ""),
            })
        return pd.DataFrame(rows)

# --- 4. REPORTING ---
def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=0):
    """Simple paired bootstrap mean CI."""
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0 or np.all(np.isnan(arr)):
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        sample   = rng.choice(arr, size=len(arr), replace=True)
        means[i] = np.nanmean(sample)
    lo = np.percentile(means, 100 * (alpha / 2))
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return float(lo), float(hi)

def finalize(results_dict):
    full_df = pd.concat(results_dict.values(), ignore_index=True)
    full_df.to_csv("comprehensive_bridge_analysis.csv", index=False)
    CONSOLE.print("\n[bold green][+] Full CSV saved: comprehensive_bridge_analysis.csv[/bold green]")

    table = Table(title="Neural Bridge — Source-Fidelity Validation (Llama -> Gemma-4)")
    table.add_column("Dataset",                style="cyan")
    table.add_column("sim(src, st)\nFIDELITY", justify="right", style="green")
    table.add_column("sim(tgt, st)\nDISPLACE", justify="right", style="yellow")
    table.add_column("sim(src, tgt)\nBASELINE",justify="right", style="dim")
    table.add_column("Fidelity Gain\n(st-tgt)",justify="right", style="bold green")
    table.add_column("Hijack Win-rate",        justify="right", style="bold magenta")
    table.add_column("First-token Match",      justify="right", style="bold blue")

    for name, df in results_dict.items():
        gains    = df['fidelity_gain'].values
        gain_lo, gain_hi = bootstrap_ci(gains)
        win_rate = float((df['sim_src_st'] > df['sim_tgt_st']).mean())
        ftm      = float(df['first_token_match'].mean())

        table.add_row(
            name,
            f"{df['sim_src_st'].mean():.4f}",
            f"{df['sim_tgt_st'].mean():.4f}",
            f"{df['sim_src_tgt'].mean():.4f}",
            f"{gains.mean():+.4f}\n[{gain_lo:+.3f}, {gain_hi:+.3f}]",
            f"{win_rate*100:5.1f}%",
            f"{ftm*100:5.1f}%",
        )
    CONSOLE.print(table)

    overall_gain = full_df['fidelity_gain'].mean()
    overall_win  = float((full_df['sim_src_st'] > full_df['sim_tgt_st']).mean())
    CONSOLE.print(
        f"\n[bold]OVERALL:[/bold] mean fidelity gain "
        f"= [bold green]{overall_gain:+.4f}[/bold green]   "
        f"hijack win-rate = [bold magenta]{overall_win*100:.1f}%[/bold magenta]   "
        f"(N = {len(full_df)})\n"
        "-> A positive gain & win-rate > 50% means: when Llama's residual stream\n"
        "   is injected via the bridge, Gemma-4's voice answers more like Llama\n"
        "   than like itself."
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200, help="Examples per dataset.")
    parser.add_argument("--mapper", type=str, default=None, help="Override mapper checkpoint path.")
    args = parser.parse_args()
    if args.mapper:
        MAPPER_PATH = args.mapper

    v = NeuralBridgeValidator()
    results = {
        "TruthfulQA": v.run_eval("truthful_qa", "generation", "validation", limit=args.limit),
        "MMLU":       v.run_eval("cais/mmlu",   "high_school_world_history", "test", limit=args.limit),
        "ARC":        v.run_eval("ai2_arc",     "ARC-Challenge",             "test", limit=args.limit),
    }
    finalize(results)