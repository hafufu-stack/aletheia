# Project Aletheia V4: The Complete Physics of LLM Truth

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20088666.svg)](https://doi.org/10.5281/zenodo.20088666)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **"LLMs do not lack knowledge—they suppress it. At 14B parameters, a single symbol prefix achieves 100% factual accuracy with zero external knowledge."**

## Overview

Project Aletheia is a systematic **113-phase** investigation of LLM hallucination through the lens of condensed matter physics. Starting from GPT-2 (124M) as a "particle accelerator" and scaling to Qwen2.5-14B, I establish **seven fundamental laws**, **four theorems**, and **four new universal principles** governing how transformers suppress factual knowledge.

### V4 Highlights 🔥🔥🔥

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

## Four Fundamental Theorems

| Theorem | Phases | Statement |
|---------|--------|-----------|
| **Internal Impossibility** | P37–P48, P71–P72 | No internal operation recovers suppressed facts—**all methods: 0%**, including prompt engineering |
| **L10 Optimality** | P49, P54 | Single-layer L10 Logit Lens (40%) outperforms all ensemble methods (8–25%) |
| **Detection–Generation Separation** | P36, P57 | Perfect hallucination detection does not enable correction |
| **Dark Matter Hypothesis** | P63, P75 | Math tokens interact **29% less** with suppressor weight matrices—GSF is a selective filter, not a universal brake 🔥 |

## Four New Universal Principles (V4) 🆕

| Principle | Phases | Statement |
|-----------|--------|-----------|
| **Aletheia Constant** | P96, P103, P110, P113 | Best factual layer at **α_A ≈ 0.95** depth — invariant to architecture (GPT-2, Qwen), language (EN, JA), and temperature (0.1–10.0) 🔥🔥🔥 |
| **Dual-Engine Theory** | P97, P112 | Code Mode = **Shield** (weakens late-layer suppressors) + **Sword** (strengthens early-layer amplifiers) — universal across GPT-2 XL and Qwen 🔥🔥 |
| **Alignment Tax** | P101, P105, P108 | Instruction tuning hypertrophies MLP suppressors by **6×** — effective capacity reduced to **N_eff = N/5**. Bypassed by "The answer is:" 🔥 |
| **Entropy Bomb** | P104, P109 | "The answer is:" causes **full-layer entropy explosion** (+0.30 uniform) — mechanistically distinct from Code Mode's targeted Shield/Sword 🔥 |

## Key Discovery: The 14B Singularity 🔥🔥🔥

```
                    Natural    Code #    "The answer is:"
GPT-2 Small  (124M):  15%       20%           10%
GPT-2 XL     (1.5B):  55%       65%           55%
Qwen2.5-1.5B:         65%       60%           75%
Qwen2.5-14B:          90%      100%  ←       100%  ←  SINGULARITY
```

At 14B parameters, **truth extraction is a solved problem**. P99's scaling law predicted 98.7% — actual was 100%.

## Key Discovery: The L10 Oracle

The foundational finding: LLMs **know the truth** at intermediate layers.

```
Layer L10:  "Tokyo" at Rank 1  ← The model KNOWS
Layer L12:  "Tokyo" at Rank 13 ← Grammar pushes it down
Output:     "the"              ← Fluent hallucination
```

Extracting facts directly from L10 via Logit Lens: **10% → 40% accuracy (4× improvement)**, with zero external knowledge or retraining.

## Experimental Phases (113 total)

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

### Season 21–23: Dark Matter Engineering & Scaling Law (P84–P95) 🆕
- **P87**: CoT Illusion — **Chain-of-Thought is an incomplete Code Mode Switch** 🔥
- **P91**: Lobotomy Generalization — Neural lobotomy does NOT generalize (0% on novel facts)
- **P92**: Perplexity Cost — Suppressors cost only **1.4% perplexity** (essentially free)
- **P95**: Grand Scaling Law — **GSF: 6.0→0.5→0.5→0.0 across scales**; XL Paradox discovered 🔥🔥

### Season 24: The Universal Horizon (P96–P101) 🆕
- **P96**: Universal 0.94 — **Aletheia Constant confirmed across GPT-2 + Qwen** 🔥🔥🔥
- **P97**: Dual-Engine — **Shield + Sword mechanism in 1,200 XL heads** 🔥🔥
- **P98**: CoT Horizon — "The answer is:" = 70–75% in Qwen (best template)
- **P99**: Scaling Prediction — **90% at 8.2B (Code), 11.6B (Natural)** 🔥
- **P101**: Alignment Tax — **7B Instruct ≈ 1.5B Base → N_eff = N/5** 🔥🔥

### Season 25: Beyond the Singularity (P103–P107) 🆕
- **P103**: Syntax Theorem — Proportional and offset models both R²>0.995
- **P104**: Answer Anatomy — **"The answer is:" = entropy bomb, not funnel** 🔥
- **P105–P106**: Alignment Autopsy — Suppressor Hypertrophy confirmed; Frankenstein Surgery = 0%
- **P107**: **14B Singularity — Code Mode = 100% accuracy** 🔥🔥🔥

### Season 26: The Transcendence Era (P108–P113) 🆕
- **P108**: MLP Autopsy — **Alignment Tax lives in MLPs (6× back-half suppression)** 🔥
- **P109**: Reasoning Horizon — Facts=entropy bomb, Logic=Code Mode, Math=impossible at 1.5B
- **P110**: Cross-Lingual — **0.94 constant invariant to language (EN=JA)** 🔥
- **P112**: Dual-Engine Universality — Shield/Sword confirmed on Qwen (GQA) 🔥
- **P113**: Temperature Physics — **0.94 constant invariant to temperature (0.1–10.0)** 🔥🔥

## Repository Structure

```
aletheia/
├── experiments/          # Phase 1–95 scripts
│   ├── phase1_pauli_exclusion.py
│   ├── ...
│   └── phase95_grand_scaling.py
├── experiments2/         # Phase 96–113 scripts (Qwen era)
│   ├── phase96_universal_094.py
│   ├── ...
│   └── runner_s26.py
├── results/              # JSON results for each phase
├── figures/              # Generated visualizations (100+ figures)
├── papers/               # LaTeX source (paper_v4.tex)
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
  title={Project Aletheia V4: The Complete Physics of LLM Truth---From Phase Transitions to the 14B Singularity},
  author={Funasaki, Hiroto},
  year={2026},
  doi={10.5281/zenodo.20088666}
}
```

## License

MIT License

## Sponsorship

This research is conducted entirely independently. If you find this work valuable, please consider [sponsoring on GitHub](https://github.com/sponsors/hafufu-stack).
