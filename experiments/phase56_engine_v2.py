# -*- coding: utf-8 -*-
"""
Phase 56: Aletheia Engine v2 - The Complete System
Combine ALL Season 9-12 discoveries into one unified engine:
  1. Oracle (P31): attention entropy detects hallucination
  2. L10 Logit Lens (P49): extract facts from intermediate layers
  3. Multi-Layer Voting (P54): aggregate L6-L10 evidence
  4. Trajectory Fingerprint (P55): prefer "suppressed" candidates
  5. IDK Attractor (P47): when nothing works, say "I don't know"

The complete Aletheia pipeline for each token:
  IF oracle(H) < threshold: use L12 (normal, confident)
  ELSE:
    candidates = L10_top_K
    score each by: (a) multi-layer consensus, (b) trajectory suppression
    IF best_score > min_threshold: output best candidate
    ELSE: output "I don't know"
"""
import os, json, sys
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import phase_complete

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
FIGURES_DIR = os.path.join(PROJECT_DIR, 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

ENTROPY_THRESHOLD = 1.0

def load_model():
    print("[P56] Loading GPT-2 (eager)...")
    tok = GPT2Tokenizer.from_pretrained('gpt2', local_files_only=True)
    mdl = GPT2LMHeadModel.from_pretrained('gpt2', local_files_only=True,
                                           attn_implementation='eager')
    mdl.to(DEVICE).eval()
    mdl.config.output_attentions = True
    tok.pad_token = tok.eos_token
    return mdl, tok

class AletheiaEngineV2:
    """Complete Aletheia Engine combining all discoveries."""

    def __init__(self, model, tok, entropy_threshold=1.0,
                 candidate_k=10, voting_layers=(6,7,8,9,10)):
        self.model = model
        self.tok = tok
        self.threshold = entropy_threshold
        self.k = candidate_k
        self.voting_layers = voting_layers

    def _get_all_layer_info(self, input_ids):
        """Forward pass collecting all hidden states + attentions."""
        layer_hs = {}
        handles = []
        for li in range(12):
            def make_hook(idx):
                def hook_fn(module, args, output):
                    layer_hs[idx] = output[0][0, -1, :].detach()
                return hook_fn
            h = self.model.transformer.h[li].register_forward_hook(make_hook(li))
            handles.append(h)

        with torch.no_grad():
            out = self.model(input_ids, output_attentions=True, return_dict=True)
        for h in handles:
            h.remove()

        return layer_hs, out

    def _oracle_entropy(self, out):
        """Compute attention entropy from model output."""
        ents = []
        for attn in out.attentions:
            for h in range(attn.shape[1]):
                a = attn[0, h, -1, :].cpu().numpy()
                ents.append(float(-np.sum(a * np.log(a + 1e-12))))
        return float(np.mean(ents))

    def _logit_lens(self, layer_hs, layer_idx):
        """Get logits from a specific layer."""
        normed = self.model.transformer.ln_f(layer_hs[layer_idx].unsqueeze(0))
        return self.model.lm_head(normed).squeeze(0)

    def _multi_layer_score(self, layer_hs, token_id):
        """Score a token by its consistency across voting layers."""
        reciprocal_rank_sum = 0.0
        for li in self.voting_layers:
            logits = self._logit_lens(layer_hs, li)
            rank = int((logits.argsort(descending=True) == token_id).nonzero().item()) + 1
            reciprocal_rank_sum += 1.0 / rank
        return reciprocal_rank_sum / len(self.voting_layers)

    def _trajectory_suppression(self, layer_hs, out, token_id):
        """Compute suppression score: how much rank dropped from peak to final."""
        ranks = []
        for li in range(12):
            logits = self._logit_lens(layer_hs, li)
            rank = int((logits.argsort(descending=True) == token_id).nonzero().item()) + 1
            ranks.append(rank)
        final_rank = int((out.logits[:, -1, :].squeeze(0).argsort(descending=True) == token_id).nonzero().item()) + 1

        peak_rank = min(ranks)
        suppression = final_rank - peak_rank
        return suppression, peak_rank

    def infer(self, prompt):
        """Complete Aletheia inference for a single prompt."""
        input_ids = self.tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
        layer_hs, out = self._get_all_layer_info(input_ids)

        # Step 1: Oracle
        entropy = self._oracle_entropy(out)
        final_logits = out.logits[:, -1, :].squeeze(0)
        l12_token = torch.argmax(final_logits).item()

        if entropy < self.threshold:
            # CONFIDENT -> use L12
            return {
                'token': l12_token, 'source': 'L12_confident',
                'entropy': entropy, 'idk': False,
            }

        # Step 2: UNCERTAIN -> extract L10 candidates
        l10_logits = self._logit_lens(layer_hs, 10)
        candidates = torch.topk(l10_logits, self.k).indices.tolist()

        # Step 3: Score each candidate
        scored = []
        for cand in candidates:
            # Multi-layer consensus score
            ml_score = self._multi_layer_score(layer_hs, cand)
            # Trajectory suppression score
            suppression, peak = self._trajectory_suppression(layer_hs, out, cand)
            # Combined score: weight consensus higher, suppression as tiebreaker
            combined = ml_score * 100 + max(0, suppression) * 0.01
            scored.append({
                'token': cand, 'ml_score': ml_score,
                'suppression': suppression, 'peak_rank': peak,
                'combined': combined,
            })

        scored.sort(key=lambda x: x['combined'], reverse=True)
        best = scored[0]

        # Step 4: IDK check - if best candidate has very low score
        if best['ml_score'] < 0.05 and best['peak_rank'] > 100:
            return {
                'token': None, 'source': 'IDK',
                'entropy': entropy, 'idk': True,
                'candidates': scored[:3],
            }

        return {
            'token': best['token'], 'source': 'Aletheia_v2',
            'entropy': entropy, 'idk': False,
            'ml_score': best['ml_score'],
            'suppression': best['suppression'],
            'candidates': scored[:3],
        }


def main():
    print("=" * 70)
    print("  Phase 56: Aletheia Engine v2 - The Complete System")
    print("  Oracle + L10 + Multi-Layer Voting + Trajectory + IDK")
    print("=" * 70)

    model, tok = load_model()
    engine = AletheiaEngineV2(model, tok)

    tests = [
        ("The capital of Japan is", [11790], "Tokyo"),
        ("The capital of France is", [6342], "Paris"),
        ("Water freezes at", [657], "0"),
        ("The largest planet is", [22721], "Jupiter"),
        ("DNA stands for", [390], "de"),
        ("The chemical symbol for gold is", [7591], "Au"),
        ("Shakespeare wrote", [13483], "Hamlet"),
        ("The speed of light is approximately", [22626], "299"),
        ("The Earth orbits the", [4252], "Sun"),
        ("The boiling point of water is", [1802], "100"),
        ("Albert Einstein developed the theory of", [44449], "relativity"),
        ("Oxygen has the atomic number", [807], "8"),
    ]

    # === P56a: Full engine evaluation ===
    print("\n[P56a] Aletheia Engine v2 evaluation...")
    engine_results = []
    for prompt, fact_ids, expected in tests:
        result = engine.infer(prompt)

        is_correct = result['token'] in fact_ids if result['token'] is not None else False
        tname = tok.decode([result['token']]).encode('ascii','replace').decode().strip() if result['token'] else 'IDK'

        engine_results.append({
            'expected': expected, 'selected': tname,
            'correct': is_correct, 'source': result['source'],
            'entropy': round(result['entropy'], 3),
            'idk': result.get('idk', False),
        })

        tag = 'OK' if is_correct else ('IDK' if result['idk'] else 'FAIL')
        print(f"  {expected:>12s}: [{result['source']:>16s}] -> '{tname}' [{tag}]")

    engine_acc = sum(1 for r in engine_results if r['correct']) / len(tests)
    n_idk = sum(1 for r in engine_results if r['idk'])
    n_l12 = sum(1 for r in engine_results if r['source'] == 'L12_confident')
    n_aletheia = sum(1 for r in engine_results if r['source'] == 'Aletheia_v2')

    # === P56b: Compare all methods ===
    print("\n[P56b] Method comparison...")
    # L12 baseline
    l12_correct = 0
    l10_correct = 0
    for prompt, fact_ids, expected in tests:
        inp = tok(prompt, return_tensors='pt')['input_ids'].to(DEVICE)
        # L12
        with torch.no_grad():
            out = model(inp)
        if torch.argmax(out.logits[:, -1, :]).item() in fact_ids:
            l12_correct += 1
        # L10
        hidden = {}
        def hook(m, a, o):
            hidden['h'] = o[0][0, -1, :].detach()
        handle = model.transformer.h[10].register_forward_hook(hook)
        with torch.no_grad():
            model(inp)
        handle.remove()
        normed = model.transformer.ln_f(hidden['h'].unsqueeze(0))
        l10_logits = model.lm_head(normed).squeeze(0)
        if torch.argmax(l10_logits).item() in fact_ids:
            l10_correct += 1

    l12_acc = l12_correct / len(tests)
    l10_acc = l10_correct / len(tests)

    # === P56c: Threshold sweep for engine ===
    print(f"\n[P56c] Engine threshold sweep...")
    threshold_results = {}
    for thresh in [0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0, 3.0]:
        eng = AletheiaEngineV2(model, tok, entropy_threshold=thresh)
        correct = 0
        idks = 0
        for prompt, fact_ids, expected in tests:
            result = eng.infer(prompt)
            if result['idk']:
                idks += 1
            elif result['token'] in fact_ids:
                correct += 1
        threshold_results[thresh] = {
            'accuracy': correct / len(tests),
            'idk_rate': idks / len(tests),
        }
        print(f"  thresh={thresh:.1f}: {correct}/{len(tests)} = {correct/len(tests):.0%} "
              f"(IDK={idks})")

    # === Visualization ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Method comparison
    methods = ['L12 Final', 'L10 Only', 'Aletheia v2']
    accs = [l12_acc*100, l10_acc*100, engine_acc*100]
    colors_bar = ['red', 'green', 'blue']
    axes[0].bar(methods, accs, color=colors_bar, alpha=0.7)
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Method Comparison')

    # Engine breakdown
    sources = ['L12 confident', 'Aletheia v2', 'IDK']
    counts = [n_l12, n_aletheia, n_idk]
    axes[1].pie(counts, labels=sources, autopct='%1.0f%%',
                colors=['lightcoral', 'lightblue', 'lightyellow'])
    axes[1].set_title(f'Engine Decision Sources (N={len(tests)})')

    # Threshold sweep
    thresholds = sorted(threshold_results.keys())
    t_accs = [threshold_results[t]['accuracy']*100 for t in thresholds]
    t_idks = [threshold_results[t]['idk_rate']*100 for t in thresholds]
    axes[2].plot(thresholds, t_accs, 'g.-', linewidth=2, label='Accuracy')
    axes[2].plot(thresholds, t_idks, 'r.--', linewidth=2, label='IDK rate')
    axes[2].set_xlabel('Entropy Threshold')
    axes[2].set_ylabel('%')
    axes[2].set_title('Engine Performance vs Threshold')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Phase 56: Aletheia Engine v2', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase56_engine_v2.png'), dpi=150, bbox_inches='tight')
    plt.close()

    results = {
        'phase': 56, 'name': 'Aletheia Engine v2',
        'l12_accuracy': l12_acc, 'l10_accuracy': l10_acc,
        'engine_accuracy': engine_acc,
        'n_idk': n_idk, 'n_l12': n_l12, 'n_aletheia': n_aletheia,
        'threshold_sweep': {str(k): v for k, v in threshold_results.items()},
        'engine_results': engine_results,
    }
    with open(os.path.join(RESULTS_DIR, 'phase56_engine_v2.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("  PHASE 56 RESULTS: Aletheia Engine v2")
    print("=" * 70)
    print(f"  L12 Final:    {l12_acc:.0%}")
    print(f"  L10 Only:     {l10_acc:.0%}")
    print(f"  Engine v2:    {engine_acc:.0%}")
    print(f"  Sources: L12={n_l12}, Aletheia={n_aletheia}, IDK={n_idk}")
    best_t = max(threshold_results, key=lambda t: threshold_results[t]['accuracy'])
    print(f"  Best thresh:  {best_t} -> {threshold_results[best_t]['accuracy']:.0%}")
    print("=" * 70)
    phase_complete(56)

if __name__ == '__main__':
    main()
