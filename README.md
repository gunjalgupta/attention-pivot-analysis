# Attention Pivot Analysis

Can attention head activations distinguish a genuine self-correction from a hallucinated one, before a language model finishes its answer? We built a pipeline to test this and report what we found — including a result that looked promising at first and did not survive replication.

**Paper:** in preparation — link coming soon
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
└── figures/
    └── (figures used in the paper)
```

All code (as runnable Kaggle notebooks) and all data are hosted on Kaggle — this repo is the landing page, paper, and figures. See [Pipeline & Notebooks](#pipeline--notebooks) and [Data](#data) below.

---

## Pipeline & Notebooks

| Phase | Kaggle Notebook | Input → Output |
|---|---|---|
| 1. Data Generation | [kaggle.com/code/gunjalgupta/phase-1-v2](https://www.kaggle.com/code/gunjalgupta/phase-1-v2) | Prompt set → raw CoT generations (Qwen2.5-1.5B-Instruct) |
| 2. Pivot Labeling | [kaggle.com/code/gunjalgupta/phase-2](https://www.kaggle.com/code/gunjalgupta/phase-2) | Raw generations → token-aligned, labeled pivot instances |
| 3. Attention Extraction | [kaggle.com/code/gunjalgupta/phase3-v2](https://www.kaggle.com/code/gunjalgupta/phase3-v2) | Labeled pivots → per-head attention weights & output magnitudes (run twice: n=82 and n=134) |
| 4. Hypothesis Testing | [kaggle.com/code/gunjalgupta/phase-4](https://www.kaggle.com/code/gunjalgupta/phase-4) | Attention tensors → permutation tests, classifier, cross-domain ablation |

Each notebook documents its exact input path and requirements in its first markdown cell. Click "Copy & Edit" on Kaggle to fork and rerun with your own compute, or "Code" → download `.ipynb` to run elsewhere (see [Reproducing Elsewhere](#reproducing-elsewhere) below).

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

## Reproducing Elsewhere

The fastest way to reproduce is forking the notebooks directly on Kaggle (see [Pipeline & Notebooks](#pipeline--notebooks) — "Copy & Edit" preserves the GPU environment and dataset attachments automatically).

To run outside Kaggle (locally, Colab, another cloud):

```bash
pip install -r requirements.txt
```

Download each notebook's `.ipynb` from its Kaggle page, then download its input data from the previous notebook's Output tab (see [Data](#data) above) and update the hardcoded `/kaggle/input/...` paths at the top of each notebook to point at your local copy.

**Note:** Notebooks use `kaggle_secrets` for API tokens, which only exists in Kaggle's environment. If running elsewhere, replace the secrets-loading cell with your own token-handling method — **never hardcode API tokens directly in the notebook.**

---

## Data

All data files are available from the **Output tab** of the notebook that produced them (Kaggle sidebar → Output, on each notebook linked above). No separate dataset pages — everything traces back to one of the four notebooks.

| File | Produced by | Size |
|---|---|---|
| `raw_outputs.jsonl`, `raw_outputs_batch2.jsonl` | Phase 1 notebook, Output tab | small |
| `pivot_candidates.jsonl`, `pivot_candidates_batch2.jsonl` | Phase 1 notebook, Output tab | small |
| `phase2_labeled_combined.jsonl` (155 records — the master labeled dataset used in all downstream analysis) | Phase 2 notebook, Output tab | ~0.3 MB |
| `phase3_attention_v2.npz` + `phase3_metadata_v2.jsonl` (n=82 attention tensors) | Phase 3 notebook, Output tab (first run) | ~550 MB |
| `phase3_attention_v3.npz` + `phase3_metadata_v3.jsonl` (n=134 attention tensors) | Phase 3 notebook, Output tab (second run) | ~857 MB |
| `phase4_results.json` (summary statistics behind every table in the paper) | Phase 4 notebook, Output tab | ~5 KB |

**Note:** both attention tensor files (n=82 and n=134) are needed to reproduce the paper's central finding — the n=82 result is not superseded by n=134, since the whole point of the paper is the comparison between them.

Field-level schema for the labeled JSONL files is documented in the paper, Section 3.2, and in the first markdown cell of the Phase 2 notebook.

---

## Citation

A paper describing this work is in preparation. Citation details will be added here once available — check back, or open an issue if you'd like to be notified.

## License

[MIT](LICENSE) — code and analysis scripts. Model outputs generated using Qwen2.5-1.5B-Instruct, subject to the [Qwen license](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/LICENSE).

## Contact

Author - Gunjal Gupta, Vidhi Shah, Nikita Goyal | Gunjal1gupta@gmail.com
