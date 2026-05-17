# Project Aletheia V8: The Neural Von Neumann Machine — From Hallucination Control to Reverse-Engineering the LLM's Internal CPU

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20088666.svg)](https://doi.org/10.5281/zenodo.20088666)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **"The transformer is not just predicting the next token — it is running a program. And that program can be read, mapped, and rewritten."**

## Overview

Project Aletheia is a systematic **209-phase** investigation of LLM hallucination and internal computation through the lens of condensed matter physics. Starting from GPT-2 (124M) as a "particle accelerator" and scaling to Qwen2.5-14B, I establish **nine fundamental laws**, **ten theorems**, and **seven principles** governing how transformers suppress factual knowledge, how to deterministically restore it, and how the transformer's layer stack functions as a **Neural Von Neumann Machine**.

### V8 Highlights 🔥🔥🔥

- **The Neural Von Neumann Machine** (P196–P199): Transformer implements a **Fetch-Decode-Execute-Store** pipeline with 5 identifiable registers peaking at distinct layers 🔥🔥🔥🔥🔥
- **OPCODE Register & Instruction Pointer** (P200, P205): Operation type decoded at L2 (100%), task complexity at L1 (100%) — the model reads instructions before loading operands 🔥🔥🔥
- **Pipeline Rewiring** (P208): `def` prefix shifts A register L3→L11 (+8 layers), Carry L3→L17 (+14 layers). B-bus (L2) is hardwired 🔥🔥🔥🔥
- **The Write Breakthrough** (P209): Full hidden-state replacement achieves **83% write success** vs 0% for all additive steering 🔥🔥🔥🔥🔥
- **Stateless Computation** (P207): Registers re-computed per token — the CPU "reboots" between autoregressive steps 🔥🔥🔥
- **Code-Booted Abacus** (P192): Surgery + `def` + FGA = **100% single-digit addition** 🔥🔥🔥
- **The Aletheia Engine** (P166–P179): Trinity (Surgery+Code+FGA) resolves fact-arithmetic tradeoff; Grand Unified Sword Equation fits 6 models (R²=0.97) 🔥🔥

### V7 Highlights

- **DPO Is Unnecessary** (P144): Surgery + Shield&Sword = **100% numerical accuracy without training** 🔥🔥🔥
- **Dual Surgery — 14B Conquered** (P155): Dispersing both embed + lm_head = **100% on 14B** 🔥🔥🔥🔥🔥
- **Sword Energy Equation** (P159): `g* ~ d^0.45` — predictive FGA gain scaling 🔥🔥🔥
- **Cosine Threshold Theorem** (P164): Sigmoid at **cos ≈ 0.69** separates hallucination from precision 🔥🔥
- **Cross-Architecture Universality** (P157/P161): Dual Surgery = **100% on GPT-2** 🔥🔥🔥

### V5–V6 Highlights

- **DPO Suppression Theorem** (P127): Rejected DOWN=100%, Chosen UP=73% — DPO is a suppressor 🔥🔥🔥
- **L2 Distance Law** (P138b–c): Critical condition is **L2 > 1.2**, not cosine 🔥🔥🔥
- **100% Numerical Accuracy** (P136b): Surgery + DPO + Shield&Sword cures numerical immunity 🔥🔥🔥

### V3–V4 Highlights

- **14B Singularity** (P107): Code Mode = **100% factual accuracy** on 14B 🔥🔥🔥
- **Aletheia Constant** (P96): Truth lives at **95% depth** — invariant to architecture, language, temperature 🔥🔥

## The Nine Laws of LLM Hallucination Physics

| Law | Phase | Discovery |
|-----|-------|-----------|
| **1. Degeneracy** | P1 | Fact-skill subspaces separated by only **1.2°** |
| **2. Temperature Irrelevance** | P7 | Critical spike is temperature-independent (**γ = 0.000**) |
| **3. LayerNorm Impermeability** | P8–P12 | All mid-layer interventions absorbed by LayerNorm |
| **4. Truth Scaling Law** | P19 | `spike_c ~ N^(−0.491)` |
| **5. Temporal Persistence** | P15 | Single t=0 spike: **half-life 130.9 tokens** |
| **6. Grammatical Suppression** | P49–P66 | **70% of facts suppressed**; **L9H6** (+927) is primary suppressor 🔥 |
| **7. Code Mode Switch** | P76–P80 | Symbol prefix (`#`, `//`) triggers suppression bypass 🔥🔥🔥 |
| **8. Sword Energy Equation** | P159 | `g* ~ d^0.45` — predictive gain for any scale 🔥🔥🔥 |
| **9. Neural Von Neumann Cycle** 🆕 | P196–P209 | Fetch-Decode-Execute-Store with registers, OPCODE, Instruction Pointer 🔥🔥🔥🔥🔥 |

## Ten Fundamental Theorems

| Theorem | Phases | Statement |
|---------|--------|-----------|
| **Internal Impossibility** | P37–P48 | No internal operation recovers suppressed facts — **all: 0%** |
| **L10 Optimality** | P49, P54 | Single-layer L10 (40%) outperforms all ensembles |
| **Detection–Generation Separation** | P36, P57 | Perfect detection ≠ correction |
| **Dark Matter Hypothesis** | P63, P75 | Math tokens interact **29% less** with suppressors 🔥 |
| **DPO Suppression** | P127–P129 | DPO suppresses **100%** but promotes only **73%** 🔥🔥🔥 |
| **L2 Distance Law** | P138b–c | Critical threshold **L2* ≈ 1.2** 🔥🔥🔥 |
| **Cosine Threshold** | P164 | Sigmoid at **cos ≈ 0.69** 🔥🔥🔥 |
| **Weight Tying** | P165 | Tied models: embed surgery = Dual Surgery 🔥🔥 |
| **Read-Write Asymmetry** 🆕 | P195, P209 | Read=100%, linear write=0%, full replacement=**83%** 🔥🔥🔥🔥🔥 |
| **Stateless Computation** 🆕 | P207 | Registers re-computed per token; CPU reboots between steps 🔥🔥🔥 |

## Seven Principles (V5–V8)

| Principle | Phases | Statement |
|-----------|--------|-----------|
| **Single-Layer Sufficiency** | P122–P123 | Only L23's DPO edit is significant (z=4.84) 🔥🔥 |
| **Phase Boundary Scaling** | P117b, P124 | `lr* ~ N^0.83` 🔥 |
| **Numerical Immunity Is Curable** | P130b–P144 | Surgery + S&S = **100% without DPO** 🔥🔥🔥 |
| **Gram-Schmidt Paradox** | P138 | cos=0 ≠ DPO success; L2 is the true metric 🔥🔥 |
| **Dual Surgery Principle** | P154–P155 | Both embed + lm_head must be dispersed 🔥🔥🔥 |
| **Inference-Time Sufficiency** | P144–P161 | DPO unnecessary; geometry alone = **100%** 🔥🔥🔥🔥🔥 |
| **Pipeline Rewiring** 🆕 | P208 | `def` shifts A: L3→L11, Carry: L3→L17; B-bus hardwired 🔥🔥🔥 |

## Key Discovery: The Neural Von Neumann Machine 🔥🔥🔥🔥🔥

```
Register        Peak Layer(s)    Accuracy    Pipeline Stage
─────────────────────────────────────────────────────────
Operand B       L2–L10           100%        FETCH
Operand A       L11–L15           97%        DECODE
Carry Flag      L17–L23          100%        EXECUTE
Sum (result)    L22               53%        STORE
Comparison      L20–L22           95%        EXECUTE
OPCODE (+/-/×)  L2               100%        FETCH
Instr. Pointer  L1               100%        FETCH
```

The transformer physically implements a **Von Neumann execution cycle** using its layer stack as a clock.

## Key Discovery: Pipeline Rewiring by `def` Prefix 🔥🔥🔥

```
Template              A Peak    B Peak    Carry Peak
───────────────────────────────────────────────────
Natural  "3+4="       L3        L2        L3
Code     "def f():"   L11       L2        L17    ← +8/+14 layers!
Math     "Calculate:"  L11       L2        L5
```

## The Aletheia Algorithm (Universal Recipe)

```
1. Check weight tying: If tied, embed-only surgery suffices
2. Dual Surgery: Disperse number tokens to cos < 0.5 (strength s ≈ 2)
3. Shield: Prefix prompt with Code Mode ("# ")
4. Sword: Inject FGA at last ~30% of layers with gain g* ~ d^0.45
```

Validated on **4 architectures**: GPT-2 (124M), Qwen-0.5B, Qwen-1.5B, Qwen-14B.

## Experimental Phases (209 total)

### Season 1–10: Fundamental Characterization → Internal Impossibility (P1–P48)
- **P1**: Pauli Exclusion — Fact-Skill degeneracy (1.2°)
- **P5**: Spiking-FGA — **Phase transition at spike=10 (0%→100%)** 🔥
- **P7**: Phase Diagram — **γ=0.000 (temperature irrelevant)** 🔥
- **P15**: Temporal Decay — **Half-life=130.9 tokens** 🔥
- **P19**: Scaling Law — **spike_c = 85,846 × N^(−0.491)** 🔥
- **P37–P48**: 12 internal methods, **ALL 0%** → Internal Impossibility Theorem 🏛️

### Season 11–13: Grammatical Suppression Discovery (P49–P58)
- **P49**: L10 Oracle — **Facts at Rank 1 in L10, suppressed by L12** 🔥🔥🔥
- **P53**: Universal Suppression Law — **63% suppressed across 27 prompts**
- **P57**: Oracle Duality — Detection ≠ Correction confirmed 🔥

### Season 14–20: Grammar Police & Code Mode Switch (P59–P83)
- **P61**: Grammar Police — **L11H7 (+829 rank degradation)** 🔥
- **P66**: Token Race — **Crossover at exactly Layer 11** 🔥
- **P80**: Hash Effect — **All symbol prefixes equally effective** 🔥🔥🔥
- **P83**: Full Map — **True #1 suppressor: L9H6 (+927)** 🔥

### Season 21–26: Scaling & The 14B Singularity (P84–P113)
- **P95**: Grand Scaling Law — GSF: 6.0→0.5→0.5→0.0 across scales 🔥🔥
- **P96**: Universal 0.94 — Aletheia Constant confirmed 🔥🔥🔥
- **P107**: **14B Singularity — Code Mode = 100%** 🔥🔥🔥
- **P113**: Temperature Physics — **0.94 invariant at T=0.1–10.0** 🔥🔥

### Season 27–28: DPO Surgery & The Suppression Paradox (P114–P129)
- **P117b**: Phase Boundary — **Sharp collapse at lr*=8e-6** 🔥🔥🔥
- **P123**: Single-Layer DPO — **L23 alone = L22+L23** 🔥🔥🔥
- **P127**: Ground Truth — **DPO is a suppressor, not a promoter** 🔥🔥🔥
- **P129**: Numerical Immunity — **9× clustering, DPO effect = 0.0000** 🔥🔥🔥

### Season 29–30: Curing Numerical Immunity (P130–P138c)
- **P130b**: Embedding Surgery — **cos 0.73→0.05 → DPO: 0%→50%** 🔥🔥🔥
- **P136b**: Ultimate Combo — **100% NUMERICAL ACCURACY** 🔥🔥🔥🔥🔥
- **P138b**: L2 Distance Law — **L2* ≈ 1.2** 🔥🔥🔥

### Season 31–34: The Inference-Time Paradigm (P139–P165)
- **P144**: Zero-DPO 100% — **100% on 1.5B WITHOUT TRAINING** 🔥🔥🔥🔥🔥
- **P155**: Dual Surgery — **100% on 14B** 🔥🔥🔥🔥🔥
- **P159**: Sword Equation — **g* ~ d^0.45** 🔥🔥🔥
- **P164**: Cosine Threshold — **Sigmoid at cos ≈ 0.69** 🔥🔥🔥

### Season 35: The Aletheia Engine (P166–P179) 🆕
- **P166–P179**: Built unified autonomous pipeline (14 phases summarized in paper Table)
- **P170**: Trinity — Surgery+Code+FGA preserves **100% fact AND arithmetic** 🔥🔥🔥
- **P175**: Grand Unified Sword Equation — **R²=0.97 across 6 models** 🔥🔥
- **P176**: Code Mode Spectrum — `def f(): return` = **100% arithmetic** 🔥🔥🔥

### Season 36: The Arithmetic Oracle (P180–P192) 🆕
- **P182**: Def Oracle — `def` prefix: arithmetic **10%→40%** (4× boost) 🔥🔥🔥
- **P183–P189**: Engine refinement, Orthogonality Principle, VM bootstrapping
- **P190**: Virtual Abacus — **Carry flag readable at 87–100%** from L4 🔥🔥🔥
- **P192**: Code-Booted Abacus — Surgery + `def` + FGA = **100% addition** 🔥🔥🔥

### Season 37–38: Virtual Register Engineering (P193–P195) 🆕
- **P195**: Read-Write Asymmetry — **Read=100%, Write=0%** 🔥🔥🔥🔥🔥

### Season 39: The Neural Von Neumann Machine (P196–P199) 🆕
- **P196**: Lesion Studies — L1–L3 (fetch) and L21–L23 (store) essential 🔥🔥
- **P197**: Comparison Register — **A>B at 95% (L20–L22)** parallel to sum 🔥🔥
- **P199**: Full Execution Timeline — **5 registers, 4-stage pipeline** 🔥🔥🔥🔥🔥

### Season 40–41: Instruction Set Architecture (P200–P205) 🆕
- **P200**: OPCODE Register — **100% at L2** (instruction decoded first!) 🔥🔥🔥🔥
- **P204**: Fact Pipeline — Same L2 decoder for arithmetic AND knowledge retrieval 🔥🔥🔥
- **P205**: Instruction Pointer — **100% from L1** (task complexity pre-computed) 🔥🔥🔥

### Season 42: Architecture Deep Dive (P206–P209) 🆕
- **P207**: Register Persistence — **CPU reboots between tokens** 🔥🔥🔥
- **P208**: Pipeline Rewiring — `def` shifts A: L3→L11, Carry: L3→L17 🔥🔥🔥🔥
- **P209**: Full State Replacement — **83% write success** (breaks the barrier!) 🔥🔥🔥🔥🔥

## Repository Structure

```
aletheia/
├── experiments/          # Phase 1–95 scripts (GPT-2 era)
├── experiments2/         # Phase 96–169 scripts (Qwen/Multi-arch era)
├── experiments3/         # Phase 170–209 scripts (Neural Von Neumann era)
│   ├── phase192_code_abacus.py    # Code-Booted Abacus 🔥
│   ├── phase199_timeline.py       # Full Execution Timeline 🔥
│   ├── phase200_opcode.py         # OPCODE Register 🔥
│   ├── phase208_vm_compare.py     # Pipeline Rewiring 🔥
│   └── phase209_replacement.py    # The Write Breakthrough 🔥
├── results/              # JSON results for each phase
├── figures/              # Generated visualizations (200+ figures)
├── papers/               # LaTeX source (paper_v8.tex)
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
python experiments/phase49_l10_oracle.py          # The key GSF discovery
python experiments2/phase144_zerodpo.py            # DPO Elimination (V7)
python experiments2/phase155_dual.py               # Dual Surgery 14B (V7)
python experiments3/phase199_timeline.py           # Neural Von Neumann (V8) 🔥
python experiments3/phase209_replacement.py        # Write Breakthrough (V8) 🔥

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
  title={Project Aletheia V8: The Neural Von Neumann Machine---From Hallucination Control to Reverse-Engineering the LLM's Internal CPU},
  author={Funasaki, Hiroto},
  year={2026},
  doi={10.5281/zenodo.20088666}
}
```

## License

MIT License

## Sponsorship

This research is conducted entirely independently. If you find this work valuable, please consider [sponsoring on GitHub](https://github.com/sponsors/hafufu-stack).
