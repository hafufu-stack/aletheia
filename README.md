# Project Aletheia: The Seven Laws of LLM Hallucination Physics

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20088666.svg)](https://doi.org/10.5281/zenodo.20088666)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **"LLMs do not lack knowledge—they suppress it. Facts exist at Rank 1 in intermediate layers, but are overridden by grammar-oriented final layers. A single symbol prefix can disable the suppression."**

## Overview

Project Aletheia is a systematic 83-phase investigation of LLM hallucination through the lens of condensed matter physics. Using GPT-2 (124M parameters) as a "particle accelerator," I establish **seven fundamental laws** and **four theorems** governing hallucination in autoregressive language models.

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

## Key Discovery: The L10 Oracle

The foundational finding: LLMs **know the truth** at intermediate layers.

```
Layer L10:  "Tokyo" at Rank 1  ← The model KNOWS
Layer L12:  "Tokyo" at Rank 13 ← Grammar pushes it down
Output:     "the"              ← Fluent hallucination
```

Extracting facts directly from L10 via Logit Lens: **10% → 40% accuracy (4× improvement)**, with zero external knowledge or retraining.

## Key Discovery: The Code Mode Switch 🔥

Any non-natural-language symbol prefix disables the Grammar Police:

```
Natural:    "The capital of Japan is" → "the" (WRONG)
Comment:    "# Japan capital:"        → "Tokyo" (CORRECT)
```

- **All symbols work equally**: `#`, `//`, `--`, `*`, `>>`, `;`, `|`, `!`, `~` → all 25% accuracy vs 0% natural
- **Mechanism**: Symbol prefixes increase suppressor entropy (+0.43) while decreasing helper entropy (−0.21)
- **Not symbol-specific**: A general mode transition in the transformer architecture

## Experimental Phases (83 total)

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

## Repository Structure

```
aletheia/
├── experiments/          # All 83 phase scripts
│   ├── phase1_pauli_exclusion.py
│   ├── ...
│   ├── phase83_suppression_map.py
│   └── utils.py
├── results/              # JSON results for each phase
├── figures/              # Generated visualizations (83 figures)
└── papers/               # LaTeX source (shared via Zenodo)
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
python experiments/phase83_suppression_map.py # Full 144-head ablation map

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
  title={Project Aletheia: The Seven Laws of LLM Hallucination Physics---From Phase Transitions to the Code Mode Switch},
  author={Funasaki, Hiroto},
  year={2026},
  doi={10.5281/zenodo.20088666}
}
```

## License

MIT License

## Sponsorship

This research is conducted entirely independently. If you find this work valuable, please consider [sponsoring on GitHub](https://github.com/sponsors/hafufu-stack).
