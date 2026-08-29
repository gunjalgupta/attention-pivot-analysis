# Attention Pivot Analysis

**Can attention-head activations distinguish a genuine self-correction from a hallucinated one before a language model finishes its answer?**

This repository contains a reproducible mechanistic-interpretability case study of that question. The project began with an encouraging small-sample result, then deliberately tested whether it survived additional data, prompt-grouped evaluation, a disjoint follow-up set, reconstructed full-context replay, label-sensitivity checks, and a second model family.

> **Main finding:** the initial Qwen result did not replicate. A class-weighted linear classifier reached ROC-AUC **0.720** on the initial sample (`n=75`), declined to **0.579** after expansion (`n=127`), and reached **0.392** when trained on the original 75 records and evaluated only on the 52 later records. No H1–H3 attention head survived FDR correction in either Qwen sample. Reconstructed full context and the Gemma replication did not recover a stable, generalizable signal.

![Pipeline overview](figures/fig1_pipeline.png)

## Why this project matters

Mechanistic-interpretability studies often test hundreds or thousands of internal features against relatively few labeled examples. In that regime, a plausible-looking result can survive ordinary cross-validation and still fail when new data are collected.

This repository therefore emphasizes the **research process**, not only the final negative result:

- define falsifiable hypotheses before inspecting individual heads;
- separate exploratory and follow-up analyses;
- correct for testing hundreds of heads;
- keep records from the same prompt together during grouped evaluation;
- evaluate on later-collected records;
- audit labels and rerun sensitivity analyses;
- test whether context truncation explains the result;
- repeat the pipeline on another model family.

## Research question

Reasoning traces often contain pivot phrases such as *“Wait, actually…”*. Some pivots genuinely repair an earlier mistake, while others abandon a correct line of reasoning or introduce a new error. Surface text alone may not distinguish them reliably.

We test whether the model's internal attention behavior immediately before the pivot differs between:

- **correction** — the pivot repairs an earlier mistake;
- **hallucination_pivot** — the pivot moves from a correct or defensible line of reasoning to an incorrect one.

Records labeled `uncertain` are retained in the labeled dataset but excluded from the binary hypothesis tests and classifiers.

## Models

- **Primary model:** Qwen2.5-1.5B-Instruct
- **Second-model check:** Gemma instruction-tuned model used in the included Gemma notebooks

The Qwen pipeline generated **1,320 traces from 88 prompts** across math, logic, and misdirection domains. The final labeled Qwen dataset contains **155 aligned pivots**:

| Label | Count |
|---|---:|
| correction | 23 |
| hallucination_pivot | 110 |
| uncertain | 22 |

The completion-only Qwen analyses use:

- **Initial sample:** `n=75` — 14 corrections, 61 hallucination pivots
- **Expanded sample:** `n=127` — 22 corrections, 105 hallucination pivots
- **Later records:** 52 newly collected misdirection-domain records — 8 corrections, 44 hallucination pivots

The expanded sample contains the original 75 records plus the 52 later records. It is therefore a deliberate expansion and follow-up test, not a fully independent replication sample.

## Attention extraction

For each pivot, the completion-only extraction stores:

- the final **30 pre-pivot query tokens**;
- up to **250 earlier generated-text key positions**;
- attention weights for every layer and head;
- per-head output vectors for the same query window.

When more than 250 pre-pivot generated tokens exist, the earliest generated tokens are truncated. The full-context notebooks separately reconstruct prompt-plus-generation context to test whether this truncation explains the result.

![Attention extraction window](figures/fig2_extraction_window.png)

## Hypotheses

| Hypothesis | Feature | Predicted pattern |
|---|---|---|
| **H1: Early-context attention mass** | Attention mass on the first 50 real positions retained in the key window, per head | Corrections attend more strongly to early retained context |
| **H2: Head-output magnitude** | Mean L2 norm of each head's output vector over the 30-query window | Specific heads activate more strongly before corrections |
| **H3: Structurally normalized attention entropy** | Shannon entropy over causally available, non-padding keys, normalized by `log(K_q)` | Corrections show more diffuse attention |
| **H4: Late/early magnitude ratio** | Mean late-layer head magnitude divided by mean early-layer magnitude | Hallucination pivots show stronger late-layer dominance |

For H3, `K_q` is defined structurally for each query position from padding and causal availability. Attention is restricted to those keys, renormalized, and entropy is divided by `log(K_q)`.

## Statistical analysis

The primary analysis uses:

- two-sided permutation tests for H1–H4;
- Benjamini–Hochberg FDR correction across all per-head tests;
- class-weighted logistic regression with `C=0.01`;
- five-fold stratified cross-validation;
- prompt-grouped cross-validation with `StratifiedGroupKFold`;
- prompt-level bootstrap confidence intervals;
- PCA and SelectKBest robustness checks fit inside each training fold;
- coefficient-stability analysis;
- label-sensitivity and prompt-audit analyses.

Random seeds are fixed to `42` in the final notebooks.

## Main Qwen results

### Per-head hypothesis tests

No H1, H2, or H3 head survived FDR correction in either sample.

| Result | Initial `n=75` | Expanded `n=127` |
|---|---:|---:|
| H1 FDR-significant heads | 0 / 336 | 0 / 336 |
| H2 FDR-significant heads | 0 / 336 | 0 / 336 |
| H3 FDR-significant heads | 0 / 336 | 0 / 336 |
| H3 raw `p < .05` | 10.4% | 10.1% |
| H4 permutation `p` | 0.057 | 0.305 |

![Qwen head-level hypothesis tests](figures/fig3_qwen_hypothesis_tests.png)

### Classifier and generalization results

| Evaluation | ROC-AUC | Balanced accuracy | F1, correction class |
|---|---:|---:|---:|
| Initial instance-level CV, `n=75` | **0.720** | 0.572 | 0.312 |
| Expanded instance-level CV, `n=127` | **0.579** | 0.514 | 0.208 |
| Initial prompt-grouped CV | 0.679 | 0.536 | — |
| Expanded prompt-grouped CV | 0.473 | 0.434 | — |
| Train on initial 75, test on later 52 | **0.392** | 0.494 | 0.244 |

The prompt-level bootstrap 95% confidence interval for the later-52 ROC-AUC is **[0.179, 0.605]**.

Cross-domain ROC-AUC also declined after expansion:

| Held-out domain | Initial `n=75` | Expanded `n=127` |
|---|---:|---:|
| math | 0.707 | 0.222 |
| logic | 0.525 | 0.412 |
| misdirection | 0.554 | 0.421 |

![Classifier and cross-domain performance](figures/fig4_performance_comparison.png)

## Full-context validation

The completion-only analysis intentionally extracts attention from generated text only. To test whether missing prompt context caused the decline, the Qwen full-context notebooks reconstruct prompt-plus-generation input and repeat the feature and classifier analyses under teacher-forced replay.

Key result on the later 52 records:

- completion-only ROC-AUC: **0.392**
- reconstructed full-context ROC-AUC: **0.409**
- difference: **+0.017**
- paired prompt-bootstrap 95% CI: **[-0.174, 0.307]**

The expanded prompt-grouped ROC-AUC was **0.473** for completion-only features and **0.484** for reconstructed full-context features. Full context therefore did not materially rescue the signal.

> The reconstructed full-context analysis is teacher-forced replay. It is not a literal replay of the original generation-time activations because the exact original token IDs and checkpoint revisions were not archived.

![Qwen full-context validation](figures/fig5_full_context_validation.png)

## Second-model check: Gemma

The Gemma completion-only dataset contains **76 usable records**: 13 corrections and 63 hallucination pivots.

- instance-level ROC-AUC: **0.503**
- prompt-grouped ROC-AUC: **0.614**
- within-prompt grouped permutation test: **p=0.458**
- grouped bootstrap 95% CI: **[0.447, 0.774]**
- reconstructed full-context instance-level ROC-AUC: **0.540**
- reconstructed full-context prompt-grouped ROC-AUC: **0.614**
- H1–H3 FDR-significant heads: **0**

The nominal grouped AUC is not treated as confirmatory because it did not exceed the within-prompt permutation null and remained unstable under correction-level leave-one-out checks.

## P-value diagnostics

The diagnostic plots show the full distribution of raw per-head p-values rather than only the heads that survive correction. H3 shows an excess of small raw p-values in both Qwen samples, but the pattern does not produce any FDR-significant head and does not translate into stable classifier generalization.

![Qwen p-value diagnostics](figures/figD1_qwen_pvalue_diagnostics.png)

## Repository structure

```text
.
├── README.md
├── figures/
│   ├── fig1_pipeline.png
│   ├── fig2_extraction_window.png
│   ├── fig3_qwen_hypothesis_tests.png
│   ├── fig4_performance_comparison.png
│   ├── fig5_full_context_validation.png
│   └── figD1_qwen_pvalue_diagnostics.png
└── notebooks/
    ├── qwen/
    │   ├── 01_Qwen_Data_Generation.ipynb
    │   ├── 02_Qwen_Labeling.ipynb
    │   ├── 03_Qwen_Completion_Only_Extraction.ipynb
    │   ├── 04a_Qwen_Completion_Only_Primary_Analysis.ipynb
    │   ├── 04b_Qwen_Replication_or_Expanded_Sample.ipynb
    │   ├── 04c_Qwen_Completion_Only_Robustness.ipynb
    │   ├── 05a_Qwen_Full_Context_Raw_Extraction.ipynb
    │   └── 05b_Qwen_Full_Context_Validation.ipynb
    └── gemma/
        ├── 01_Gemma_Generation.ipynb
        ├── 02_Gemma_Labeling.ipynb
        ├── 03_Gemma_Completion_Only_Extraction.ipynb
        ├── 04a_Gemma_Completion_Only_Primary_Analysis.ipynb
        ├── 04c_Gemma_Completion_Only_Robustness.ipynb
        ├── 05a_Gemma_Full_Context_Raw_Extraction.ipynb
        └── 05b_Gemma_Full_Context_Validation.ipynb
```

## Notebook guide

### Qwen

| Notebook | Purpose |
|---|---|
| `01_Qwen_Data_Generation.ipynb` | Generate reasoning traces across math, logic, and misdirection prompts |
| `02_Qwen_Labeling.ipynb` | Detect pivots, align pivot tokens, apply automatic labels, and record manual corrections |
| `03_Qwen_Completion_Only_Extraction.ipynb` | Extract completion-only attention weights and per-head outputs |
| `04a_Qwen_Completion_Only_Primary_Analysis.ipynb` | Initial `n=75` H1–H4 tests and classifier |
| `04b_Qwen_Replication_or_Expanded_Sample.ipynb` | Expanded `n=127` H1–H4 tests and classifier |
| `04c_Qwen_Completion_Only_Robustness.ipynb` | Grouped CV, disjoint evaluation, bootstrap CI, dimensionality reduction, coefficient stability, and label audits |
| `05a_Qwen_Full_Context_Raw_Extraction.ipynb` | Reconstruct and extract full-context tensors |
| `05b_Qwen_Full_Context_Validation.ipynb` | Compare completion-only and reconstructed full-context analyses |

### Gemma

The Gemma notebooks repeat the same generation, labeling, extraction, robustness, and full-context checks for the second-model analysis.

## Running the project

The notebooks were developed for Kaggle and several input paths point to Kaggle datasets. To rerun them elsewhere, update the path constants near the top of each notebook.

Recommended order:

```text
Qwen: 01 → 02 → 03 → 04a → 04b → 04c → 05a → 05b
Gemma: 01 → 02 → 03 → 04a → 04c → 05a → 05b
```

The analysis notebooks rebuild their own feature matrices and do not depend on hidden in-memory state from another notebook. They do require the extracted NPZ tensors and aligned metadata produced by the earlier stages or provided as input datasets.

### Main Python dependencies

```text
numpy
pandas
scipy
statsmodels
scikit-learn
matplotlib
seaborn
torch
transformers
transformer-lens
tqdm
```

GPU access is recommended for generation and attention extraction. The statistical-analysis notebooks are CPU-compatible but load large tensor archives into memory.

## Reproducibility notes

- The final Qwen completion-only tensor archives contain 75 and 127 usable records and are approximately 1.3 GB combined.
- Per-head tests use permutation testing followed by Benjamini–Hochberg correction.
- Prompt-grouped analyses keep all records from the same prompt in the same fold.
- Feature selection and PCA are fit inside each cross-validation training fold.
- The later-52 analysis is a disjoint follow-up evaluation under domain shift because all 52 later records are from the misdirection domain.
- The expanded `n=127` analysis is not fully independent because it includes the original 75 records.
- This is a **probing analysis**, not a causal intervention study. The results do not establish that any attention head causes self-correction behavior.

## Interpretation

The results do not show that internal model state is incapable of encoding genuine correction. They show that this particular set of attention-derived features, at this scale and under these evaluation designs, does not provide a stable linear signature that generalizes to later data.

The strongest methodological lesson is broader: **high-dimensional interpretability findings should be treated as provisional until they survive deliberately collected follow-up data. Cross-validation within the original sample is not a substitute for that test.**

## Status

- Qwen completion-only pipeline: complete
- Qwen expanded and robustness analyses: complete
- Qwen reconstructed full-context validation: complete
- Gemma completion-only and full-context checks: complete
- Paper: in preparation

