# attention-pivot-analysis
Do attention heads distinguish genuine self-correction from hallucinated pivots in LLMs? A tested-and-non-replicating negative result.

# Attention Pivot Analysis

Can attention head activations distinguish a genuine self-correction from a hallucinated one, before a language model finishes its answer? We built a pipeline to test this and report what we found — including a result that looked promising at first and did not survive replication.

**Paper:** [link to be added]
**Status:** negative / non-replicating result — see [Key Finding](#key-finding) below

---

## Overview

Large language models frequently produce mid-reasoning "pivots" — phrases like *"Wait, actually..."* — that either genuinely correct an earlier mistake or abandon a correct line of reasoning for an incorrect one. From the surface text alone, these two cases are often indistinguishable. This project tests whether they're distinguishable from the model's *internal* attention activations in the tokens immediately preceding the pivot.

We generate reasoning traces from an instruction-tuned LLM, detect and label pivot instances as genuine corrections or hallucination pivots, extract per-head attention weights and output magnitudes via [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens), and test four pre-registered hypotheses about what should differ between the two classes.

## Key Finding

An initial sample (n = 82, 14 correction examples) produced a classifier with ROC-AUC 0.683 — an encouraging result on its face. We deliberately expanded the dataset (to n = 134, 22 correction examples) specifically to test whether this held up. **It did not.** ROC-AUC fell to 0.590, and cross-domain generalization collapsed to below-chance levels (AUC 0.144) on held-out arithmetic prompts. No individual attention head reached statistical significance under FDR correction, at either sample size, for any of the four tested hypotheses.

We interpret the initial result as a small-sample artifact rather than a genuine signal, and report the full pipeline and negative result as a methodological contribution — including a demonstration of what non-replication looks like in an exploratory interpretability study.

Full writeup, related work, and discussion are in the paper (link above).

---

## Repository Structure

```
attention-pivot-analysis/
├── README.md
├── LICENSE
├── requirements.txt
├── notebooks/
│   ├── 01_data_generation.ipynb       # Phase 1 — generate CoT traces, detect pivots
│   ├── 02_pivot_labeling.ipynb        # Phase 2 — token alignment + correction/hallucination labels
│   ├── 03_attention_extraction.ipynb  # Phase 3 — TransformerLens attention extraction
│   └── 04_hypothesis_testing.ipynb    # Phase 4 — permutation tests + classifier + ablations
├── data/
│   ├── phase2_labeled_combined.jsonl  # 155 labeled pivot instances
│   └── phase4_results.json            # summary statistics from all analyses
└── figures/
    └── (figures used in the paper)
```

Large intermediate files (raw generations, extracted attention tensors, ~850 MB) are hosted separately — see [Data](#data) below.

---

## Pipeline

| Phase | Notebook | Input → Output |
|---|---|---|
| 1. Data Generation | `01_data_generation.ipynb` | Prompt set → raw CoT generations (Qwen2.5-1.5B-Instruct) |
| 2. Pivot Labeling | `02_pivot_labeling.ipynb` | Raw generations → token-aligned, labeled pivot instances |
| 3. Attention Extraction | `03_attention_extraction.ipynb` | Labeled pivots → per-head attention weights & output magnitudes |
| 4. Hypothesis Testing | `04_hypothesis_testing.ipynb` | Attention tensors → permutation tests, classifier, cross-domain ablation |

Each notebook is runnable independently given its documented input file; see the markdown cell at the top of each for exact requirements.

**Model:** [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
**Interpretability library:** [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
**Compute:** developed on Kaggle's free T4 / T4×2 GPU tier

---

## Results Summary

| Metric | n = 82 (14 correction) | n = 134 (22 correction) |
|---|---|---|
| Balanced accuracy | 0.504 | 0.515 |
| ROC-AUC (overall) | 0.683 | 0.590 |
| Cross-domain AUC — math held out | 0.396 | 0.144 |
| Cross-domain AUC — logic held out | 0.512 | 0.412 |
| Cross-domain AUC — misdirection held out | 0.506 | 0.441 |
| Significant heads (FDR, any hypothesis) | 0 / 336 | 0 / 336 |

Full hypothesis-by-hypothesis breakdown, fold-level detail, and figures are in the paper.

---

## Reproducing

```bash
git clone https://github.com/<your-username>/attention-pivot-analysis.git
cd attention-pivot-analysis
pip install -r requirements.txt
```

Run notebooks in order (`01` → `04`). Each expects the previous notebook's output as input — update the input path at the top of each notebook if your files live somewhere other than the default `data/` location.

**Note:** Notebooks were developed on Kaggle and use `kaggle_secrets` for API tokens. If running elsewhere (Colab, local), replace the secrets-loading cell with your own token-handling method — **never hardcode API tokens directly in the notebook.**

---

## Data

| File | Size | Location |
|---|---|---|
| `phase2_labeled_combined.jsonl` | ~30 KB | in this repo, `data/` |
| `phase4_results.json` | ~5 KB | in this repo, `data/` |
| Raw generations (`raw_outputs*.jsonl`) | ~15 MB | [Kaggle Dataset link — to be added] |
| Extracted attention tensors (`phase3_attention_v3.npz`) | ~850 MB | [Kaggle Dataset / HuggingFace Dataset link — to be added] |

---

## Citation

If you use this pipeline or dataset, please cite:

```bibtex
@misc{yourname2026attentionpivot,
  title  = {Can Attention Heads Distinguish Genuine Self-Correction from Hallucinated Pivots in LLMs? A Negative Result},
  author = {[Author Name]},
  year   = {2026},
  note   = {Preprint},
  url    = {[paper link to be added]}
}
```

## License

[MIT](LICENSE) — code and analysis scripts. Model outputs generated using Qwen2.5-1.5B-Instruct, subject to the [Qwen license](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/LICENSE).

## Contact

[Author name / email — to be added]
