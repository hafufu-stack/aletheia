# Project Aletheia: Physics-Inspired Eradication of LLM Hallucinations

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20088666.svg)](https://doi.org/10.5281/zenodo.20088666)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **"Hallucination is not a bug—it is a phase transition."**

## Overview

Project Aletheia is a systematic investigation of LLM hallucination through the lens of condensed matter physics and quantum mechanics. Through 23 experimental phases on GPT-2 (124M parameters), I establish **five fundamental laws** governing hallucination in autoregressive language models.

## The Five Laws of LLM Hallucination Physics

| Law | Phase | Discovery |
|-----|-------|-----------|
| **1. Degeneracy** | P1 | Fact and skill subspaces are separated by only **1.2°**—fluent lies are structurally inevitable |
| **2. Temperature Irrelevance** | P7 | Critical spike is temperature-independent (**γ = 0.000**)—lowering temperature cannot fix hallucination |
| **3. LayerNorm Impermeability** | P8, P9, P11 | All mid-layer interventions are absorbed—**output layer is the only viable intervention point** |
| **4. Truth Scaling Law** | P19 | `spike_c ~ N^(−0.491)`—larger models need **exponentially smaller** interventions |
| **5. Temporal Persistence** | P15 | A single t=0 spike propagates with **half-life of 130.9 tokens** |

## Key Result

A single `logit_bias` at the first generation step achieves **deterministic factual generation**—no retraining, no RLHF, no model access required.

```
The capital of Japan is → "the capital of the United States"  (no spike, hallucination)
The capital of Japan is → "Tokyo, and the capital of..."      (spike at t=0, truth)
```

## Experimental Phases

### Season 1: Fundamental Characterization (P1–P5)
- **P1**: Pauli Exclusion — Fact-Skill degeneracy (1.2°)
- **P2**: Event Horizon — Epistemic entropy mapping (1.58× increase)
- **P3**: Antimatter CD — Contrastive decoding (7.3% reduction)
- **P4**: Retrocausal FSPO — Pre-emptive hallucination detection (395 kills)
- **P5**: Spiking-FGA — **Phase transition at spike=10 (0%→100%)** 🔥

### Season 2: The LayerNorm Barrier (P6–P12)
- **P7**: Phase Diagram — **γ=0.000 (temperature irrelevant)** 🔥
- **P8**: Layer-Resolved — Mid-layer=0%, output=100%
- **P9**: Neutrino Spike — Zero-mean bypass fails
- **P11**: Quantum Zeno — Distributed spikes=0%, single=100%

### Season 3: Mechanistic Dissection (P13–P15)
- **P13**: Universality — 50/50 questions solved, mean spike=3.9±3.1
- **P14**: Head Surgery — 94 fact heads, 47 skill heads identified
- **P15**: Temporal Decay — **Half-life=130.9 tokens** 🔥

### Season 4: API-Scale Eradication (P16–P19)
- **P16**: Logit-Bias Isomorphism — **spike == logit_bias (diff=0.00)** 🔥
- **P17**: Prefill Slingshot — Entropy reduced 31%
- **P18**: Aletheia Suffix — Truth-activating token sequence (50% transfer)
- **P19**: Scaling Law — **spike_c = 85,846 × N^(−0.491)** 🔥

### Season 5: Robustness & Safety (P20–P23)
- **P20**: Multi-Fact — Sequential spiking works, simultaneous = highest rank wins
- **P21**: Adversarial — **Spike defeats adversarial prompts (100% at spike=7)**
- **P22**: Anti-Spike — ⚠️ Dual-use: same mechanism creates lies (25%)
- **P23**: Fusion Cannon — Spike+Prefill combo analysis

## Truth Scaling Law Predictions

| Model | Parameters | Predicted Critical Spike |
|-------|-----------|------------------------|
| GPT-2 | 124M | 6.0 (measured) |
| GPT-2 Medium | 345M | 5.5 |
| GPT-2 Large | 774M | 3.7 |
| GPT-2 XL | 1.5B | 2.7 |
| GPT-4 class | 175B | **0.26** |

## Repository Structure

```
aletheia/
├── experiments/          # All 23 phase scripts
│   ├── phase1_pauli_exclusion.py
│   ├── phase2_event_horizon.py
│   ├── ...
│   └── phase23_fusion.py
├── results/              # JSON results for each phase
├── figures/              # Generated visualizations
└── papers/               # LaTeX source and PDF
    └── paper_v1.tex
```

## Requirements

```
torch
transformers
numpy
matplotlib
```

## Quick Start

```bash
# Run any individual phase
python experiments/phase5_spiking_fga.py

# The core discovery in 3 lines:
python -c "
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
model = GPT2LMHeadModel.from_pretrained('gpt2').eval()
tok = GPT2Tokenizer.from_pretrained('gpt2')
inp = tok('The capital of Japan is', return_tensors='pt')
logits = model(**inp).logits[:,-1,:].squeeze()
logits[tok.encode(' Tokyo')[0]] += 10  # The spike
print(tok.decode(logits.argmax()))  # → Tokyo
"
```

## Paper

📄 **[Read the paper on Zenodo](https://doi.org/10.5281/zenodo.20088666)**

## Citation

```bibtex
@article{funasaki2025aletheia,
  title={Project Aletheia: Physics-Inspired Eradication of LLM Hallucinations via Output-Layer Phase Transitions},
  author={Funasaki, Hiroto},
  year={2025},
  doi={10.5281/zenodo.20088666}
}
```

## License

MIT License

## Sponsorship

This research is conducted entirely independently. If you find this work valuable, please consider [sponsoring on GitHub](https://github.com/sponsors/hafufu-stack).
