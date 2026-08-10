# =============================================================================
# PHASE 4c — Robustness & validity checks.
#
# Loads BOTH n=75 and n=127 tensors (needed for the disjoint follow-up
# evaluation and cross-sample comparisons). Produces every number in
# Section 4.2's validation-design table, Section 4.3, Appendix C's
# sensitivity check, Appendix D (all three sub-checks), and the
# ground-truth audit scope check.
#
# Independent of Phase 4a/4b -- rebuilds all features from scratch.
# Slower to run (builds H1-H4 twice, once per sample size) but has no
# dependency on prior notebook state, which is the whole point.
# =============================================================================

### CELL 1 — Imports, paths, shared utilities
import numpy as np
import json
import os
from collections import Counter, defaultdict
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

np.random.seed(42)

NPZ_75_PATH   = '/kaggle/input/notebooks/gunjalgupta/phase3-v2/pivot_research/phase3_attention_v2.npz'
META_75_PATH  = '/kaggle/input/notebooks/gunjalgupta/phase3-v2/pivot_research/phase3_metadata_v2.jsonl'
NPZ_127_PATH  = '/kaggle/input/notebooks/gunjalgupta/phase3-v2/pivot_research/phase3_attention_v3.npz'
META_127_PATH = '/kaggle/input/notebooks/gunjalgupta/phase3-v2/pivot_research/phase3_metadata_v3.jsonl'
LABELED_COMBINED_PATH = '/kaggle/input/notebooks/gunjalgupta/phase-2/pivot_research/phase2_labeled_combined.jsonl'
OUTPUT_DIR = '/kaggle/working/pivot_research'
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_LAYERS, N_HEADS, N_QUERIES, EARLY_KEY_RANGE = 28, 12, 30, 50
EARLY_LAYERS = list(range(0, N_LAYERS // 3))
LATE_LAYERS  = list(range(2 * N_LAYERS // 3, N_LAYERS))


def entropy_normalized(row, valid_start):
    real = row[valid_start:]
    real = real[real > 1e-10]
    K_q = len(real)
    if K_q <= 1 or real.sum() == 0:
        return 0.0
    p = real / real.sum()
    raw = -np.sum(p * np.log(p))
    return raw / np.log(K_q)


def load_metadata(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def compute_features(npz_path, metadata):
    """Builds the full 1,009-dim feature matrix (H1+H2+H3+H4) for one sample."""
    npz = np.load(npz_path)
    n = len(metadata)
    weights = np.stack([npz[f'weights_{i:03d}'] for i in range(n)], axis=0)
    outputs = np.stack([npz[f'outputs_{i:03d}'] for i in range(n)], axis=0)
    valid_key_starts = np.array([m['valid_key_start'] for m in metadata])

    h1 = np.zeros((n, N_LAYERS, N_HEADS))
    for i in range(n):
        v = valid_key_starts[i]
        early_end = min(v + EARLY_KEY_RANGE, weights.shape[-1])
        h1[i] = weights[i, :, :, :, v:early_end].sum(axis=-1).mean(axis=-1)

    h2 = np.linalg.norm(outputs, axis=-1).mean(axis=2)

    h3 = np.zeros((n, N_LAYERS, N_HEADS))
    for i in range(n):
        v = valid_key_starts[i]
        for L in range(N_LAYERS):
            for H in range(N_HEADS):
                ents = [entropy_normalized(weights[i, L, H, q, :], v) for q in range(N_QUERIES)]
                h3[i, L, H] = np.mean(ents)

    early_mag = h2[:, EARLY_LAYERS, :].mean(axis=(1, 2))
    late_mag  = h2[:, LATE_LAYERS, :].mean(axis=(1, 2))
    h4 = late_mag / (early_mag + 1e-10)

    X = np.concatenate([h1.reshape(n, -1), h2.reshape(n, -1), h3.reshape(n, -1), h4.reshape(n, 1)], axis=1)
    y = np.array([1 if m['label'] == 'correction' else 0 for m in metadata])
    prompt_ids = np.array([m['prompt_id'] for m in metadata])
    return X, y, prompt_ids


def permutation_test(g1, g2, n_perm=2000, seed=42):
    rng = np.random.default_rng(seed)
    observed = g1.mean() - g2.mean()
    combined = np.concatenate([g1, g2])
    n1 = len(g1)
    diffs = np.zeros(n_perm)
    for i in range(n_perm):
        rng.shuffle(combined)
        diffs[i] = combined[:n1].mean() - combined[n1:].mean()
    return observed, (np.abs(diffs) >= np.abs(observed)).mean()


### CELL 2 — Build all feature matrices (slow -- entropy loop runs twice)
print("Building n=75 features...")
meta_75 = load_metadata(META_75_PATH)
X75, y75, pid75 = compute_features(NPZ_75_PATH, meta_75)

print("Building n=127 features...")
meta_127 = load_metadata(META_127_PATH)
X127, y127, pid127 = compute_features(NPZ_127_PATH, meta_127)

new_mask = np.array([m['prompt_id'].startswith('mis2_') for m in meta_127])
Xnew, ynew, pidnew = X127[new_mask], y127[new_mask], pid127[new_mask]

print(f"n=75:  {dict(Counter(y75))}")
print(f"n=127: {dict(Counter(y127))}")
print(f"new-52: {dict(Counter(ynew))}")
assert len(ynew) == 52, f"Expected 52 new records, got {len(ynew)}"


### CELL 3 — Instance-level CV (baseline, both samples)
def instance_cv(X, y, seed=42):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    preds, probas = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in skf.split(X, y):
        sc = StandardScaler(); Xtr, Xte = sc.fit_transform(X[tr]), sc.transform(X[te])
        clf = LogisticRegression(class_weight='balanced', max_iter=2000, C=0.01, random_state=42)
        clf.fit(Xtr, y[tr])
        preds[te] = clf.predict(Xte); probas[te] = clf.predict_proba(Xte)[:, 1]
    return roc_auc_score(y, probas), balanced_accuracy_score(y, preds), f1_score(y, preds, zero_division=0)

auc_75, ba_75, f1_75 = instance_cv(X75, y75)
auc_127, ba_127, f1_127 = instance_cv(X127, y127)
print(f"Instance-level CV n=75:  AUC={auc_75:.3f}, bal_acc={ba_75:.3f}, F1={f1_75:.3f}")
print(f"Instance-level CV n=127: AUC={auc_127:.3f}, bal_acc={ba_127:.3f}, F1={f1_127:.3f}")


### CELL 4 — Prompt-grouped CV (both samples)
def grouped_cv(X, y, groups, seed=42):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    preds, probas = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in sgkf.split(X, y, groups=groups):
        sc = StandardScaler(); Xtr, Xte = sc.fit_transform(X[tr]), sc.transform(X[te])
        clf = LogisticRegression(class_weight='balanced', max_iter=2000, C=0.01, random_state=42)
        clf.fit(Xtr, y[tr])
        preds[te] = clf.predict(Xte); probas[te] = clf.predict_proba(Xte)[:, 1]
    return roc_auc_score(y, probas), balanced_accuracy_score(y, preds)

g_auc_75, g_ba_75 = grouped_cv(X75, y75, pid75)
g_auc_127, g_ba_127 = grouped_cv(X127, y127, pid127)
print(f"Prompt-grouped CV n=75:  AUC={g_auc_75:.3f}, bal_acc={g_ba_75:.3f}")
print(f"Prompt-grouped CV n=127: AUC={g_auc_127:.3f}, bal_acc={g_ba_127:.3f}")


### CELL 5 — Disjoint follow-up evaluation (train n=75, evaluate on new-52) + bootstrap CI
sc = StandardScaler(); X75s, Xnews = sc.fit_transform(X75), sc.transform(Xnew)
clf = LogisticRegression(class_weight='balanced', max_iter=2000, C=0.01, random_state=42)
clf.fit(X75s, y75)
preds_new, probas_new = clf.predict(Xnews), clf.predict_proba(Xnews)[:, 1]

d_auc = roc_auc_score(ynew, probas_new)
d_ba = balanced_accuracy_score(ynew, preds_new)
d_f1 = f1_score(ynew, preds_new, zero_division=0)
print(f"Disjoint follow-up: AUC={d_auc:.3f}, bal_acc={d_ba:.3f}, F1={d_f1:.3f}")

rng = np.random.default_rng(42)
unique_prompts = np.array(sorted(set(pidnew)))
boot_aucs = []
for _ in range(2000):
    sampled = rng.choice(unique_prompts, size=len(unique_prompts), replace=True)
    idx = np.concatenate([np.where(pidnew == p)[0] for p in sampled])
    y_boot = ynew[idx]
    if len(set(y_boot)) < 2:
        continue
    try:
        boot_aucs.append(roc_auc_score(y_boot, probas_new[idx]))
    except ValueError:
        continue
boot_aucs = np.array(boot_aucs)
ci_low, ci_high = np.percentile(boot_aucs, [2.5, 97.5])
print(f"95% CI (prompt-level bootstrap, {len(boot_aucs)}/2000 valid): [{ci_low:.3f}, {ci_high:.3f}]")


### CELL 6 — Cross-domain ablation (both samples, needed for main table)
print("\nCross-domain ablation:")
for label, X, y, categories in [
    ('n=75', X75, y75, np.array([m['category'] for m in meta_75])),
    ('n=127', X127, y127, np.array([m['category'] for m in meta_127])),
]:
    print(f"\n{label}:")
    for held_out in ['math', 'logic', 'misdirection']:
        test_mask = categories == held_out
        train_mask = ~test_mask
        sc = StandardScaler(); Xtr, Xte = sc.fit_transform(X[train_mask]), sc.transform(X[test_mask])
        clf = LogisticRegression(class_weight='balanced', max_iter=2000, C=0.01, random_state=42)
        clf.fit(Xtr, y[train_mask])
        preds, probas = clf.predict(Xte), clf.predict_proba(Xte)[:, 1]
        print(f"  {held_out}: bal_acc={balanced_accuracy_score(y[test_mask], preds):.3f}, "
              f"AUC={roc_auc_score(y[test_mask], probas):.3f}")


### CELL 7 — D.2: Dimensionality-reduced classification (both samples)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {'balanced_accuracy': 'balanced_accuracy', 'roc_auc': 'roc_auc'}

print("\nD.2: Dimensionality reduction")
for label, X, y, pca_comps in [('n=75', X75, y75, [5, 10, 20, 30, 45]), ('n=127', X127, y127, [5, 10, 20, 50, 100])]:
    print(f"\n{label}:")
    for n_comp in pca_comps:
        pipe = Pipeline([('scale', StandardScaler()), ('pca', PCA(n_components=n_comp, random_state=42)),
                          ('clf', LogisticRegression(class_weight='balanced', max_iter=2000, C=0.1, random_state=42))])
        scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring)
        print(f"  PCA-{n_comp}: bal_acc={scores['test_balanced_accuracy'].mean():.3f}±{scores['test_balanced_accuracy'].std():.3f}, "
              f"AUC={scores['test_roc_auc'].mean():.3f}±{scores['test_roc_auc'].std():.3f}")
    for k in [5, 10, 20, 50, 100]:
        pipe = Pipeline([('scale', StandardScaler()), ('select', SelectKBest(score_func=f_classif, k=k)),
                          ('clf', LogisticRegression(class_weight='balanced', max_iter=2000, C=0.1, random_state=42))])
        scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring)
        print(f"  SelectKBest-{k}: bal_acc={scores['test_balanced_accuracy'].mean():.3f}±{scores['test_balanced_accuracy'].std():.3f}, "
              f"AUC={scores['test_roc_auc'].mean():.3f}±{scores['test_roc_auc'].std():.3f}")
    single_aucs = []
    for i in range(X.shape[1]):
        try:
            auc = roc_auc_score(y, X[:, i]); single_aucs.append(max(auc, 1 - auc))
        except ValueError:
            continue
    single_aucs = np.array(single_aucs)
    print(f"  Best single-feature AUC: {single_aucs.max():.3f}, mean: {single_aucs.mean():.3f}")


### CELL 8 — D.3: Coefficient stability across folds (n=75)
print("\nD.3: Coefficient stability, n=75")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_coefs = []
for tr, te in skf.split(X75, y75):
    sc = StandardScaler(); Xtr = sc.fit_transform(X75[tr])
    clf = LogisticRegression(class_weight='balanced', max_iter=2000, C=0.01, random_state=42)
    clf.fit(Xtr, y75[tr])
    fold_coefs.append(clf.coef_[0].copy())
fold_coefs = np.array(fold_coefs)

corr_matrix = np.corrcoef(fold_coefs)
mean_corr = np.mean([corr_matrix[i, j] for i in range(5) for j in range(i + 1, 5)])
print(f"Mean pairwise correlation: {mean_corr:.3f}")

signs = np.sign(fold_coefs)
unanimous = np.all(signs == signs[0], axis=0)
print(f"Unanimous sign agreement: {unanimous.sum()} / 1009 ({unanimous.mean():.1%})")

mean_abs_coef = np.abs(fold_coefs).mean(axis=0)
stable_and_strong = unanimous & (mean_abs_coef > np.percentile(mean_abs_coef, 95))
idx = np.where(stable_and_strong)[0]
h1_count = sum(1 for i in idx if i < 336)
h2_count = sum(1 for i in idx if 336 <= i < 672)
h3_count = sum(1 for i in idx if 672 <= i < 1008)
h4_count = sum(1 for i in idx if i == 1008)
print(f"Stable+strong: {len(idx)} total -- H1:{h1_count}, H2:{h2_count}, H3:{h3_count}, H4:{h4_count}")


### CELL 9 — Prompt-aware H4 permutation test (both samples)
def within_prompt_permute(y, prompt_ids, rng):
    y_perm = y.copy()
    groups = defaultdict(list)
    for i, pid in enumerate(prompt_ids):
        groups[pid].append(i)
    for pid, idx in groups.items():
        if len(idx) > 1:
            idx = np.array(idx)
            y_perm[idx] = rng.permutation(y[idx])
    return y_perm

def prompt_aware_permutation_test(feature_values, y, prompt_ids, n_perm=10000, seed=42):
    rng = np.random.default_rng(seed)
    prompt_ids = np.array(prompt_ids)
    observed_diff = feature_values[y == 1].mean() - feature_values[y == 0].mean()
    mixed = sum(1 for pid in set(prompt_ids) if len(set(y[prompt_ids == pid])) > 1)
    null_diffs = np.zeros(n_perm)
    for i in range(n_perm):
        y_perm = within_prompt_permute(y, prompt_ids, rng)
        null_diffs[i] = feature_values[y_perm == 1].mean() - feature_values[y_perm == 0].mean()
    p_value = (np.abs(null_diffs) >= np.abs(observed_diff)).mean()
    return observed_diff, p_value, mixed

print("\nH4 prompt-aware permutation test:")
h4_75, h4_127 = X75[:, -1], X127[:, -1]
obs_75, p_75, mixed_75 = prompt_aware_permutation_test(h4_75, y75, pid75)
print(f"  n=75:  diff={obs_75:.4f}, p={p_75:.4f} ({mixed_75} mixed-class prompts of {len(set(pid75))})")
obs_127, p_127, mixed_127 = prompt_aware_permutation_test(h4_127, y127, pid127)
print(f"  n=127: diff={obs_127:.4f}, p={p_127:.4f} ({mixed_127} mixed-class prompts of {len(set(pid127))})")


### CELL 10 — Ground-truth sensitivity check (8 second-reviewer disagreements)
REVIEWER_ALTERNATIVE_LABELS = {
    ('mis2_04', 2):  'hallucination_pivot',
    ('mis2_16', 0):  'hallucination_pivot',
    ('math_27', 2):  'correction',
    ('logic_20', 7): 'correction',
    ('math_23', 8):  'correction',
}

def apply_reviewer_labels(metadata):
    meta_alt = []
    n_changed = 0
    for m in metadata:
        m2 = dict(m)
        key = (m['prompt_id'], m['sample_idx'])
        if key in REVIEWER_ALTERNATIVE_LABELS:
            m2['label'] = REVIEWER_ALTERNATIVE_LABELS[key]
            n_changed += 1
        meta_alt.append(m2)
    return meta_alt, n_changed

print("\nGround-truth sensitivity check (alternative reviewer labels):")
meta_75_alt, n_75_changed = apply_reviewer_labels(meta_75)
X75_alt, y75_alt, _ = compute_features(NPZ_75_PATH, meta_75_alt)
print(f"  n=75: {n_75_changed} labels changed")

meta_127_alt, n_127_changed = apply_reviewer_labels(meta_127)
X127_alt, y127_alt, _ = compute_features(NPZ_127_PATH, meta_127_alt)
new_mask_alt = np.array([m['prompt_id'].startswith('mis2_') for m in meta_127_alt])
Xnew_alt, ynew_alt = X127_alt[new_mask_alt], y127_alt[new_mask_alt]
print(f"  n=127: {n_127_changed} labels changed")

auc_127_alt, ba_127_alt, _ = instance_cv(X127_alt, y127_alt)
print(f"  Pooled n=127 CV, alt labels: AUC={auc_127_alt:.3f} (original: {auc_127:.3f})")

sc = StandardScaler(); X75_alt_s, Xnew_alt_s = sc.fit_transform(X75_alt), sc.transform(Xnew_alt)
clf = LogisticRegression(class_weight='balanced', max_iter=2000, C=0.01, random_state=42)
clf.fit(X75_alt_s, y75_alt)
probas_new_alt = clf.predict_proba(Xnew_alt_s)[:, 1]
d_auc_alt = roc_auc_score(ynew_alt, probas_new_alt)
print(f"  Disjoint follow-up, alt labels: AUC={d_auc_alt:.3f} (original: {d_auc:.3f})")


### CELL 11 — Full 9-prompt ground-truth audit scope check
FLAGGED_PROMPTS = ['logic_17', 'logic_03', 'mis_05', 'mis2_06', 'mis_08', 'math_13', 'logic_02', 'logic_14', 'math_36']

with open(LABELED_COMBINED_PATH) as f:
    all_records = [json.loads(l) for l in f if l.strip()]

print(f"\nGround-truth audit scope check ({len(all_records)} total records):")
total_flagged_usable = 0
for pid in FLAGGED_PROMPTS:
    matches = [r for r in all_records if r['prompt_id'] == pid]
    usable = [r for r in matches if r['label'] in ('correction', 'hallucination_pivot') and r['pivot_token_start'] >= 30]
    total_flagged_usable += len(usable)
    print(f"  {pid}: {len(matches)} total, {len(usable)} usable")

n_overall_usable = sum(1 for r in all_records if r['label'] in ('correction', 'hallucination_pivot') and r['pivot_token_start'] >= 30)
print(f"\nTotal usable from flagged prompts: {total_flagged_usable} / {n_overall_usable}")


### CELL 12 — Final label-count verification
overall = Counter(r['label'] for r in all_records)
batch1 = [r for r in all_records if not r['prompt_id'].startswith('mis2_')]
batch2 = [r for r in all_records if r['prompt_id'].startswith('mis2_')]
print(f"\nFinal count verification:")
print(f"  Overall: {dict(overall)}")
print(f"  Batch 1 ({len(batch1)}): {dict(Counter(r['label'] for r in batch1))}")
print(f"  Batch 2 ({len(batch2)}): {dict(Counter(r['label'] for r in batch2))}")
print(f"  Paper claims: 23 correction / 110 hallucination_pivot / 22 uncertain")
