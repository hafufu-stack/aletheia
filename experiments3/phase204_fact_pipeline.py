# -*- coding: utf-8 -*-
"""
Phase 204: Fact-Fetch Pipeline
Map the execution timeline for FACTUAL queries (not arithmetic).
Does the LLM have the same FETCH-DECODE-EXECUTE-STORE cycle
for fact retrieval?

"The capital of Japan is ___" -> can we probe for "Japan" and
"capital_of" as separate registers?

Model: Qwen2.5-0.5B (GPU)
"""
import torch, json, os, gc, numpy as np, time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

NUM_TOKENS = [" 0"," 1"," 2"," 3"," 4"," 5"," 6"," 7"," 8"," 9",
              " 10"," 100"," 12"," 365"]

def apply_surgery(model, tok, strength=2.0):
    embed = model.model.embed_tokens.weight.data
    ids = list(set(tok.encode(t)[-1] for t in NUM_TOKENS))
    vecs = embed[ids].clone().float()
    center = vecs.mean(dim=0)
    for i, idx in enumerate(ids):
        diff = vecs[i] - center
        direction = diff / (diff.norm() + 1e-8)
        embed[idx] += (strength * direction * embed[idx].float().norm()).to(embed.dtype)

# Facts with clear subject/predicate/object structure
FACTS = [
    # (prompt, subject_class, predicate_class, expected_answer)
    ("The capital of Japan is", "Japan", "capital", " Tokyo"),
    ("The capital of France is", "France", "capital", " Paris"),
    ("The capital of Germany is", "Germany", "capital", " Berlin"),
    ("The capital of Italy is", "Italy", "capital", " Rome"),
    ("The capital of Spain is", "Spain", "capital", " Madrid"),
    ("The capital of China is", "China", "capital", " Beijing"),
    ("The capital of Russia is", "Russia", "capital", " Moscow"),
    ("The capital of Brazil is", "Brazil", "capital", " Bras"),
    ("The language of Japan is", "Japan", "language", " Japanese"),
    ("The language of France is", "France", "language", " French"),
    ("The language of Germany is", "Germany", "language", " German"),
    ("The language of Italy is", "Italy", "language", " Italian"),
    ("The language of Spain is", "Spain", "language", " Spanish"),
    ("The language of China is", "China", "language", " Chinese"),
    ("The language of Russia is", "Russia", "language", " Russian"),
    ("The language of Brazil is", "Brazil", "language", " Portug"),
    ("The continent of Japan is", "Japan", "continent", " Asia"),
    ("The continent of France is", "France", "continent", " Europe"),
    ("The continent of Brazil is", "Brazil", "continent", " South"),
    ("The continent of China is", "China", "continent", " Asia"),
]

def main():
    print("[P204] Fact-Fetch Pipeline")
    print(f"  Device: {DEVICE}")
    start_time = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = 'Qwen/Qwen2.5-0.5B'
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, local_files_only=True, torch_dtype=torch.float32).to(DEVICE)
    apply_surgery(model, tok, strength=2.0)

    n_layers = model.config.num_hidden_layers
    target_layers = list(range(n_layers))

    # Map subjects and predicates to IDs
    subjects = sorted(set(f[1] for f in FACTS))
    predicates = sorted(set(f[2] for f in FACTS))
    subj_map = {s: i for i, s in enumerate(subjects)}
    pred_map = {p: i for i, p in enumerate(predicates)}

    labels_subj = np.array([subj_map[f[1]] for f in FACTS])
    labels_pred = np.array([pred_map[f[2]] for f in FACTS])

    # Collect hidden states for all layers
    print(f"  Collecting {n_layers}-layer hidden states for {len(FACTS)} facts...")
    all_hiddens = {l: [] for l in target_layers}
    captured = {}
    def make_hook(layer_idx):
        def fn(module, input, output):
            if isinstance(output, tuple):
                captured[layer_idx] = output[0][:, -1, :].detach().cpu().numpy().flatten()
            else:
                captured[layer_idx] = output[:, -1, :].detach().cpu().numpy().flatten()
        return fn

    for prompt, subj, pred, expected in FACTS:
        captured.clear()
        hooks = []
        for l in target_layers:
            hooks.append(model.model.layers[l].register_forward_hook(make_hook(l)))
        inp = tok(prompt, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            _ = model(**inp)
        for h in hooks: h.remove()
        for l in target_layers:
            all_hiddens[l].append(captured[l].copy())

    # Probe each layer using leave-one-out (small dataset)
    results = {}
    print("\n  === Fact Pipeline Timeline ===")
    for l in target_layers:
        X = np.array(all_hiddens[l])
        # Subject probe (7 classes)
        try:
            clf_s = LogisticRegression(max_iter=1000, random_state=42)
            clf_s.fit(X, labels_subj)
            # LOO accuracy
            correct_s = 0
            for i in range(len(X)):
                mask = np.ones(len(X), dtype=bool); mask[i] = False
                clf_loo = LogisticRegression(max_iter=1000, random_state=42)
                clf_loo.fit(X[mask], labels_subj[mask])
                if clf_loo.predict(X[i:i+1])[0] == labels_subj[i]: correct_s += 1
            acc_subj = correct_s / len(X)
        except:
            acc_subj = 0

        # Predicate probe (3 classes)
        try:
            correct_p = 0
            for i in range(len(X)):
                mask = np.ones(len(X), dtype=bool); mask[i] = False
                clf_loo = LogisticRegression(max_iter=1000, random_state=42)
                clf_loo.fit(X[mask], labels_pred[mask])
                if clf_loo.predict(X[i:i+1])[0] == labels_pred[i]: correct_p += 1
            acc_pred = correct_p / len(X)
        except:
            acc_pred = 0

        results[str(l)] = {'subject': acc_subj, 'predicate': acc_pred}
        sm = " ***" if acc_subj > 0.4 else ""
        pm = " ***" if acc_pred > 0.5 else ""
        print(f"    L{l:2d}: subject={acc_subj:.0%}{sm} predicate={acc_pred:.0%}{pm}")

    with open(os.path.join(RESULTS_DIR, 'phase204_fact_pipeline.json'), 'w') as f:
        json.dump({'phase': '204', 'name': 'Fact-Fetch Pipeline',
                   'subjects': subjects, 'predicates': predicates,
                   'results': results}, f, indent=2, default=str)

    fig, ax = plt.subplots(1, 1, figsize=(16, 7))
    layers = target_layers
    s_vals = [results[str(l)]['subject'] for l in layers]
    p_vals = [results[str(l)]['predicate'] for l in layers]
    ax.plot(layers, s_vals, 'r-o', lw=2.5, markersize=5, label='Subject (Japan/France/...)')
    ax.plot(layers, p_vals, 'b-s', lw=2.5, markersize=5, label='Predicate (capital/language/continent)')
    ax.set_xlabel('Layer', fontsize=14); ax.set_ylabel('LOO Probe Accuracy', fontsize=14)
    ax.set_ylim(-0.05, 1.05); ax.set_xticks(layers)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    ax.set_title('Phase 204: Fact-Fetch Pipeline\n'
                 'Do facts follow the same Fetch-Decode-Execute-Store cycle?',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase204_fact_pipeline.png'), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    best_subj = max(results[str(l)]['subject'] for l in layers)
    best_pred = max(results[str(l)]['predicate'] for l in layers)
    print(f"\n  === VERDICT ===")
    print(f"  -> Best subject probe: {best_subj:.0%}")
    print(f"  -> Best predicate probe: {best_pred:.0%}")
    print(f"\n  Total time: {elapsed:.0f}s")
    print("[Phase 204] Complete.")
    del model; gc.collect(); torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
