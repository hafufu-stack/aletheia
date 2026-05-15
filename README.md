# Project Aletheia V5: The Suppression Paradox — DPO as Inhibitor, Not Promoter

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20088666.svg)](https://doi.org/10.5281/zenodo.20088666)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **"DPO does not teach models what is true — it teaches them what is false."**

## Overview

Project Aletheia is a systematic **129-phase** investigation of LLM hallucination through the lens of condensed matter physics. Starting from GPT-2 (124M) as a "particle accelerator" and scaling to Qwen2.5-14B, I establish **seven fundamental laws**, **five theorems**, and **three new principles** governing how transformers suppress factual knowledge.

### V5 Highlights 🔥🔥🔥

- **DPO Suppression Theorem**: DPO suppresses rejected tokens with **100% reliability** but promotes correct ones only **73%** — it inhibits, not promotes
- **Single-Layer Sufficiency**: Only **L23's DPO edit** is statistically significant (z=4.84, p<0.001). L23 alone = L22+L23 with **50% fewer parameters**
- **Phase Boundary Scaling**: Critical learning rate `lr* ~ N^0.83` — larger models are more robust to DPO hyperparameters
- **Numerical Token Immunity**: Number embeddings are **9× more clustered** (cos=0.73 vs 0.08) — DPO effect is literally **zero** on numerical facts

### V4 Highlights

- **14B Singularity**: Qwen2.5-14B achieves **100% factual accuracy** with Code Mode — zero fine-tuning, zero external knowledge
- **Aletheia Constant**: Truth lives at **95% depth** in all transformers — invariant to architecture, language, and temperature
- **Dual-Engine Theory**: Code Mode operates Shield (weakens suppressors) + Sword (strengthens amplifiers) simultaneously
- **Alignment Tax**: Instruction tuning hypertrophies MLP suppressors by **6×**, reducing capacity to N/5

## The Seven Laws of LLM Hallucination Physics

| Law | Phase | Discovery |
|-----|-------|-----------|
| **1. Degeneracy** | P1 | Fact-skill subspaces separated by only **1.2°**—fluent lies are structurally inevitable |
| **2. Temperature Irrelevance** | P7 | Critical spike is temperature-independent (**γ = 0.000**) |
| **3. LayerNorm Impermeability** | P8–P12 | All mid-layer interventions are absorbed by LayerNorm |
| **4. Truth Scaling Law** | P19 | `spike_c ~ N^(−0.491)`—larger models need **exponentially smaller** interventions |
| **5. Temporal Persistence** | P15 | Single t=0 spike propagates with **half-life of 130.9 tokens** |
| **6. Grammatical Suppression** | P49–P66 | **70% of facts suppressed** by final layers; **L9H6** (+927) and **L11H7** (+816) are the primary "Grammar Police" heads 🔥 |
| **7. Code Mode Switch** | P76–P80 | Any symbol prefix (`#`, `//`, `--`) triggers a mode transition that **disables suppression** 🔥🔥🔥 |

## Five Fundamental Theorems

| Theorem | Phases | Statement |
|---------|--------|-----------|
| **Internal Impossibility** | P37–P48, P71–P72 | No internal operation recovers suppressed facts—**all methods: 0%** |
| **L10 Optimality** | P49, P54 | Single-layer L10 Logit Lens (40%) outperforms all ensemble methods (8–25%) |
| **Detection–Generation Separation** | P36, P57 | Perfect hallucination detection does not enable correction |
| **Dark Matter Hypothesis** | P63, P75 | Math tokens interact **29% less** with suppressor weight matrices 🔥 |
| **DPO Suppression** 🆕 | P127–P129 | DPO suppresses rejected tokens **100%** but promotes correct ones only **73%**. Numerical tokens are **structurally immune** (9× embedding clustering) 🔥🔥🔥 |

## Three New Principles (V5) 🆕

| Principle | Phases | Statement |
|-----------|--------|-----------|
| **Single-Layer Sufficiency** | P122–P123 | Only L23's DPO edit is significant (z=4.84, p<0.001). L23-only with 2× lr = L22+L23 with **50% fewer params** 🔥🔥 |
| **Phase Boundary Scaling** | P117b, P124 | Critical DPO learning rate `lr* ~ N^0.83`. 0.5B: lr*=8e-6, 1.5B: lr*=2e-5 🔥 |
| **Numerical Immunity** | P129 | Number tokens have pairwise cosine **0.73** (words: 0.08), zero baseline confidence → DPO effect = **0.0000** 🔥🔥🔥 |

## Key Discovery: DPO Is a Suppressor 🔥🔥🔥

```
What DPO actually does to token probabilities:

                    Rejected prob DOWN    Chosen prob UP
Train set:              15/15 (100%)       11/15 (73%)
Test set:                  —                1/5  (20%)

DPO works by making wrong answers less likely,
NOT by making right answers more likely.
```

Hidden state movement after DPO (P128):
- L22: **93%** toward chosen, mean |Δh| = 4.5
- L23: **80%** toward chosen, mean |Δh| = **13.9** (3× larger)

## Key Discovery: The 14B Singularity 🔥🔥🔥

```
                    Natural    Code #    "The answer is:"
GPT-2 Small  (124M):  15%       20%           10%
GPT-2 XL     (1.5B):  55%       65%           55%
Qwen2.5-1.5B:         65%       60%           75%
Qwen2.5-14B:          90%      100%  ←       100%  ←  SINGULARITY
```

## Key Discovery: The L10 Oracle

```
Layer L10:  "Tokyo" at Rank 1  ← The model KNOWS
Layer L12:  "Tokyo" at Rank 13 ← Grammar pushes it down
Output:     "the"              ← Fluent hallucination
```

## Experimental Phases (129 total)

### Season 1: Fundamental Characterization (P1–P5)
- **P1**: Pauli Exclusion — Fact-Skill degeneracy (1.2°)
- **P5**: Spiking-FGA — **Phase transition at spike=10 (0%→100%)** 🔥

### Season 2: The LayerNorm Barrier (P6–P12)
- **P7**: Phase Diagram — **γ=0.000 (temperature irrelevant)** 🔥
- **P8**: Layer-Resolved — Mid-layer=0%, output=100%
- **P11**: Quantum Zeno — Distributed spikes=0%, single=100%

### Season 3: Mechanistic Dissection (P13–P15)
- **P13**: Universality — 50/50 questions solved, mean spike=3.9±3.1
- **P14**: Head Surgery — 94 fact heads, 47 skill heads identified
- **P15**: Temporal Decay — **Half-life=130.9 tokens** 🔥

### Season 4: API-Scale Eradication (P16–P19)
- **P16**: Logit-Bias Isomorphism — **spike == logit_bias (diff=0.00)** 🔥
- **P19**: Scaling Law — **spike_c = 85,846 × N^(−0.491)** 🔥

### Season 5: Robustness & Safety (P20–P23)
- **P21**: Adversarial — Spike defeats adversarial prompts (100% at spike=7)
- **P22**: Anti-Spike — ⚠️ Dual-use: same mechanism creates lies (25%)

### Season 6: Hallucination Detection (P24–P31)
- **P30**: 11-Dimensional Clique Topology — Fact/hallucination topological fingerprint
- **P31**: Entropy Oracle — **AUC=1.000 for relative detection** 🔥

### Season 7: Intervention Attempts (P32–P36)
- **P36**: Entropy Rejection — Detection=perfect, Correction=**0%** (detection ≠ correction)

### Season 8–10: The Beautiful Null Results (P37–P48)
- **P37–P48**: 12 internal correction methods, **ALL 0%** — gradient descent, topological forcing, attention lobotomy, MCTS, Aha! vector, stochastic resonance, manifold analysis, and more
- **→ Internal Impossibility Theorem established** 🏛️

### Season 11: Grammatical Suppression Discovery (P49–P53)
- **P49**: L10 Oracle — **Facts at Rank 1 in L10, suppressed by L12** 🔥🔥🔥
- **P53**: Universal Suppression Law — **63% suppressed across 27 prompts** (geography 88%, history 80%)

### Season 12–13: Grand Unification (P54–P58)
- **P54**: Multi-Layer Voting — RRF best=25%, inferior to L10 alone (40%)
- **P57**: Oracle Duality — Relative AUC=0.66, Absolute AUC=0.33 🔥
- **P58**: Grand Unified Theory — 7 laws + 3 theorems confirmed

### Season 14: Universal Liberation (P59–P63)
- **P61**: Grammar Police — **L11H7 is the top suppressor (+829 rank degradation)** 🔥
- **P63**: GSF Is Fact-Specific — Math: suppression 0.4 vs Facts: 9.9

### Season 15: Grammar Police Deep Dive (P64–P67)
- **P65**: Suppressor Attention — L11H7 fixated on the token "The"
- **P66**: Token Race — **Fact-grammar crossover at exactly Layer 11** 🔥
- **P67**: Residual Bypass — L10 residual achieves 40% at α=0.9

### Season 16–17: Negative Results (P68–P72)
- **P69**: Prompt Structure — GSF varies **680× by prompt style** (QA=1015, possessive=1.5)
- **P71–P72**: "The" Hypothesis REJECTED; GSF-Free Pipeline FAILS → Impossibility extended

### Season 18: Dark Matter Hypothesis (P73–P77)
- **P75**: Why Math Is Immune — **29% less interaction with suppressor weights** 🔥
- **P77**: Dose-Response — L11H7 suppression is quasi-linear (0×→3×)

### Season 19–20: The Code Mode Switch (P78–P83)
- **P79**: Why Comments Work — Suppressors +0.43 entropy, helpers −0.21 entropy
- **P80**: Hash Effect — **All symbol prefixes equally effective** 🔥🔥🔥
- **P82**: Code Mode Neurons — L11:N314 (+31.8) is the switch neuron
- **P83**: Full Map — **True #1 suppressor is L9H6 (+927), not L11H7** 🔥

### Season 21–23: Dark Matter Engineering & Scaling Law (P84–P95)
- **P87**: CoT Illusion — **Chain-of-Thought is an incomplete Code Mode Switch** 🔥
- **P91**: Lobotomy Generalization — Neural lobotomy does NOT generalize (0% on novel facts)
- **P92**: Perplexity Cost — Suppressors cost only **1.4% perplexity** (essentially free)
- **P95**: Grand Scaling Law — **GSF: 6.0→0.5→0.5→0.0 across scales**; XL Paradox discovered 🔥🔥

### Season 24: The Universal Horizon (P96–P101)
- **P96**: Universal 0.94 — **Aletheia Constant confirmed across GPT-2 + Qwen** 🔥🔥🔥
- **P97**: Dual-Engine — **Shield + Sword mechanism in 1,200 XL heads** 🔥🔥
- **P98**: CoT Horizon — "The answer is:" = 70–75% in Qwen (best template)
- **P99**: Scaling Prediction — **90% at 8.2B (Code), 11.6B (Natural)** 🔥
- **P101**: Alignment Tax — **7B Instruct ≈ 1.5B Base → N_eff = N/5** 🔥🔥

### Season 25: Beyond the Singularity (P103–P107)
- **P103**: Syntax Theorem — Proportional and offset models both R²>0.995
- **P104**: Answer Anatomy — **"The answer is:" = entropy bomb, not funnel** 🔥
- **P105–P106**: Alignment Autopsy — Suppressor Hypertrophy confirmed; Frankenstein Surgery = 0%
- **P107**: **14B Singularity — Code Mode = 100% accuracy** 🔥🔥🔥

### Season 26: The Transcendence Era (P108–P113)
- **P108**: MLP Autopsy — **Alignment Tax lives in MLPs (6× back-half suppression)** 🔥
- **P109**: Reasoning Horizon — Facts=entropy bomb, Logic=Code Mode, Math=impossible at 1.5B
- **P110**: Cross-Lingual — **0.94 constant invariant to language (EN=JA)** 🔥
- **P112**: Dual-Engine Universality — Shield/Sword confirmed on Qwen (GQA) 🔥
- **P113**: Temperature Physics — **0.94 constant invariant to temperature (0.1–10.0)** 🔥🔥

### Season 27: Alignment Surgery — Surgical DPO (P114–P117b) 🆕
- **P115**: Skill LoRA — **85% accuracy with 0.1% params** 🔥
- **P116b**: Refusal Gates — Instruct models add MLP refusal neurons absent in Base
- **P117**: Surgical DPO — **73% train / 60% test** with only last 6% of MLP layers 🔥🔥
- **P117b**: Phase Boundary — **Sharp collapse at lr*=8e-6** (first-order transition) 🔥🔥🔥

### Season 28: The Epistemic Exorcism — Resolving the DPO Paradox (P118–P129) 🆕
- **P118**: SwiGLU Oracle — Residual L2 norm is best uncertainty indicator (AUC=0.83)
- **P122**: Random Control — **L23 DPO is significant (z=4.84, p<0.001); L22 is not** 🔥🔥
- **P123**: Single-Layer DPO — **L23 alone = L22+L23 with 50% fewer params** 🔥🔥🔥
- **P125–P126**: Suppression Paradox — SVD direction promotes wrong tokens (resolved: input-dependent artifact) 🔥
- **P127**: Ground Truth — **Rejected DOWN: 100%, Chosen UP: 73%** — DPO is a suppressor! 🔥🔥🔥
- **P128**: Hidden States — Toward chosen **93%** (L22), **80%** (L23); L23 movement **3× larger** 🔥🔥
- **P129**: Numerical Immunity — **Numbers 9× more clustered, DPO effect = 0.0000** 🔥🔥🔥

## Repository Structure

```
aletheia/
├── experiments/          # Phase 1–95 scripts (GPT-2 era)
├── experiments2/         # Phase 96–129 scripts (Qwen era)
│   ├── phase96_universal_094.py
│   ├── ...
│   └── phase129_numerical.py
├── results/              # JSON results for each phase
├── figures/              # Generated visualizations (130+ figures)
├── papers/               # LaTeX source (paper_v5.tex)
└── reports/              # Season progress reports
```

## Requirements

```
torch
transformers
numpy
matplotlib
scikit-learn
```

## Quick Start

```bash
# Run any individual phase
python experiments/phase49_l10_oracle.py      # The key discovery
python experiments/phase83_suppression_map.py  # Full 144-head ablation map
python experiments2/phase107_14b_singularity.py # The 14B Singularity
python experiments2/phase127_logit_diff.py     # DPO Suppression proof (V5)
python experiments2/phase129_numerical.py      # Numerical immunity (V5)

# The L10 Oracle in 10 lines:
python -c "
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
model = GPT2LMHeadModel.from_pretrained('gpt2').eval()
tok = GPT2Tokenizer.from_pretrained('gpt2')
inp = tok('The capital of Japan is', return_tensors='pt')
h = {}
model.transformer.h[10].register_forward_hook(lambda m,a,o: h.update({10: o[0][0,-1,:].detach()}))
model(**inp)
logits = model.lm_head(model.transformer.ln_f(h[10].unsqueeze(0))).squeeze()
print('L10 says:', tok.decode(logits.argmax()))  # → Tokyo
"
```

## Paper

📄 **[Read the paper on Zenodo](https://doi.org/10.5281/zenodo.20088666)**

## Citation

```bibtex
@article{funasaki2026aletheia,
  title={Project Aletheia V5: The Suppression Paradox---DPO as Inhibitor, Not Promoter},
  author={Funasaki, Hiroto},
  year={2026},
  doi={10.5281/zenodo.20088666}
}
```

## License

MIT License

## Sponsorship

This research is conducted entirely independently. If you find this work valuable, please consider [sponsoring on GitHub](https://github.com/sponsors/hafufu-stack).
