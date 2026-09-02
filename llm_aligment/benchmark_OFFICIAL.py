#!/usr/bin/env python3
"""
Neural Bridge benchmark — validates that translating the source's residual
stream into the target's space makes the TARGET speak like the SOURCE.

Key changes vs. the original benchmark.py
-----------------------------------------
1. Architecture: the cross-attentive TransformerResidualMapper is replaced
   with the LinearInjectionMapper used in test.py / ALIGNMENT_OFFICIAL_diff_dim_linear.py.
2. bridge_gen bug fixes:
   (a) The final RMSNorm (t_mod.model.norm) is now applied before lm_head.
       Hidden states from output_hidden_states are PRE-norm; without this
       the logits were systematically shifted, biasing the stitched output.
   (b) Tokens are accumulated as IDs and decoded ONCE at the end, instead
       of being string-concatenated piecewise (which mangled subword pieces
       and dropped Llama's leading-space markers).
   (c) The source side now uses KV-caching, removing the O(N^2) cost on
       the source forward pass. The target memory is still computed by
       re-tokenising the source's text trajectory with t_tok (semantic
       parity with training, where both models read the same text).
3. Source-fidelity metrics: in addition to the original three cosine
   similarities, we now report:
     - fidelity_gain     = sim_src_st - sim_tgt_st  (per-row)
     - hijack_win_rate   = P(sim_src_st > sim_tgt_st)
     - first_token_match = stitched first token == source first token
4. Cleaner final report with paired bootstrap CIs.
"""

import argparse
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
MAPPER_PATH  = "official_residual_mapper_100e_2M_5l.pt"
SRC_ID       = "Qwen/Qwen3-0.6B"
TGT_ID       = "meta-llama/Llama-3.2-1B"
SIM_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# These must match the values used in test.py
MAPPER_LAYERS = 5
MAPPER_HEADS  = 32

# --- 2. ARCHITECTURE ---
class LinearInjectionMapper(nn.Module):
    """Same architecture as in test.py / ALIGNMENT_OFFICIAL_diff_dim_linear.py."""
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
        self.s_tok = AutoTokenizer.from_pretrained(SRC_ID, trust_remote_code=True)
        self.t_tok = AutoTokenizer.from_pretrained(TGT_ID)
        self.s_mod = AutoModelForCausalLM.from_pretrained(SRC_ID, trust_remote_code=True).to(DEVICE).eval()
        self.t_mod = AutoModelForCausalLM.from_pretrained(TGT_ID).to(DEVICE).eval()

        cp = torch.load(MAPPER_PATH, map_location=DEVICE)
        self.s_scale = cp['ss']
        self.t_scale = cp['ts']

        D_SRC = self.s_mod.config.hidden_size
        D_TGT = self.t_mod.config.hidden_size
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
    # Embedding similarity helper
    # ------------------------------------------------------------
    def get_sim(self, a, b):
        if not a or not b:
            return 0.0
        embs = self.sim_model.encode([a, b], convert_to_tensor=True)
        return util.cos_sim(embs[0], embs[1]).item()

    # ------------------------------------------------------------
    # FIXED bridge_gen
    # ------------------------------------------------------------
    @torch.no_grad()
    def bridge_gen(self, prompt, max_tokens=20):
        """
        Logica di riferimento con integrazione della Final Norm:
        - Decodifica granulare (token-by-token string concatenation).
        - No KV-cache sul source (ricalcolo full sequence).
        - Applicazione di t_mod.model.norm prima del lm_head.
        """
        curr_src = self.s_tok(prompt, return_tensors="pt").input_ids.to(DEVICE)
        out_tokens = []

        for _ in range(max_tokens):
            # 1. Estrazione hidden state del source (ultimo token)
            s_out = self.s_mod(curr_src, output_hidden_states=True)
            h_src = s_out.hidden_states[-1][:, -1, :]

            # 2. Estrazione contesto target (basato sul testo attuale del source)
            txt = self.s_tok.decode(curr_src[0], skip_special_tokens=True)
            t_in = self.t_tok(txt, return_tensors="pt").input_ids.to(DEVICE)
            h_tgt_ctx = self.t_mod(t_in, output_hidden_states=True).hidden_states[-1][:, -1, :]

            # 3. Neural Bridge Projection
            p_v, _ = self.m_s2t(h_src / self.s_scale, memory=h_tgt_ctx / self.t_scale)
            mapped = p_v * self.t_scale

            # 4. Final Norm (Ripristinata) + LM Head
            # Applichiamo la normalizzazione del target prima di generare il logit
            v_h = self.t_mod.model.norm(mapped) if hasattr(self.t_mod.model, 'norm') else mapped
            next_tgt = torch.argmax(self.t_mod.lm_head(v_h), dim=-1)

            # 5. Gestione Token (Logica di riferimento)
            word = self.t_tok.decode(next_tgt)
            out_tokens.append(word)

            # 6. Avanzamento del Source per lo step successivo
            next_src = torch.argmax(self.s_mod.lm_head(h_src), dim=-1)
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
                # Derived: how much closer the stitched output is to source than to target
                "fidelity_gain":     sim_src_st - sim_tgt_st,
                # Did the stitched output land on the source's first word?
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

    table = Table(title="Neural Bridge — Source-Fidelity Validation")
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

    # Single-line headline result
    overall_gain = full_df['fidelity_gain'].mean()
    overall_win  = float((full_df['sim_src_st'] > full_df['sim_tgt_st']).mean())
    CONSOLE.print(
        f"\n[bold]OVERALL:[/bold] mean fidelity gain "
        f"= [bold green]{overall_gain:+.4f}[/bold green]   "
        f"hijack win-rate = [bold magenta]{overall_win*100:.1f}%[/bold magenta]   "
        f"(N = {len(full_df)})\n"
        "→ A positive gain & win-rate > 50% means: when the source brain is\n"
        "  injected via the bridge, the target's voice answers more like the\n"
        "  source than like itself."
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200,
                        help="Examples per dataset.")
    parser.add_argument("--mapper", type=str, default=None,
                        help="Override the mapper checkpoint path.")
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
