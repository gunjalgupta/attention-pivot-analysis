# Attention Pivot Analysis

Can attention-head activations distinguish a successful self-correction from an unsuccessful reasoning pivot *before* a language model finishes its answer?

This repository accompanies our study of attention-based signals preceding within-generation reasoning pivots in large language models. We built a complete pipeline for generating reasoning traces, detecting and labeling pivots, extracting internal attention activations, and evaluating whether attention contains predictive information about the outcome of a reasoning pivot.

**Paper:** Coming soon

---

## Key Finding

An initial dataset (n = 82, 14 successful corrections) produced an encouraging classifier (ROC-AUC = **0.683**). Rather than treating this as evidence, we deliberately collected additional data and repeated the analysis.

The apparent signal **did not persist**.

- Expanded dataset (n = 134): ROC-AUC fell to **0.590**
- Prompt-grouped evaluation: ROC-AUC = **0.468**
- Disjoint follow-up evaluation: ROC-AUC = **0.384**
- No attention head reached statistical significance after false-discovery-rate correction for any tested hypothesis.

Our conclusion is **not** that attention cannot encode self-correction. Rather, this case study demonstrates how apparently promising results from high-dimensional exploratory analyses can disappear under additional validation.

---

## Project Overview

The complete pipeline consists of four stages:

| Stage | Description |
|-------|-------------|
| **1. Data Generation** | Generate chain-of-thought reasoning traces from Qwen2.5-1.5B-Instruct |
| **2. Pivot Labeling** | Detect reasoning pivots and label them as successful corrections or unsuccessful pivots |
| **3. Attention Extraction** | Extract per-head attention weights and output magnitudes using TransformerLens |
| **4. Analysis** | Test four pre-specified hypotheses using permutation tests and classifier evaluation |

---

## Pipeline

| Stage | Kaggle Notebook |
|------|----------------|
| Phase 1 – Data Generation | https://www.kaggle.com/code/gunjalgupta/phase-1-v2 |
| Phase 2 – Pivot Labeling | https://www.kaggle.com/code/gunjalgupta/phase-2 |
| Phase 3 – Attention Extraction | https://www.kaggle.com/code/gunjalgupta/phase3-v2 |
| Phase 4 – Statistical Analysis | https://www.kaggle.com/code/gunjalgupta/phase-4 |

Each notebook documents its required inputs and outputs and can be copied directly in Kaggle for reproduction.

---

## Experimental Setup

**Model**

- Qwen2.5-1.5B-Instruct

**Interpretability Library**

- TransformerLens

**Development Environment**

- Kaggle (T4 / T4×2 GPU)

---

## Results

| Metric | n = 82 | n = 134 |
|---------|-------:|--------:|
| Balanced Accuracy | 0.504 | 0.515 |
| ROC-AUC | **0.683** | **0.590** |
| Cross-domain (Math) | 0.396 | 0.144 |
| Cross-domain (Logic) | 0.512 | 0.412 |
| Cross-domain (Misdirection) | 0.506 | 0.441 |
| Significant Attention Heads (FDR) | 0 / 336 | 0 / 336 |

Additional validation:

- Prompt-grouped ROC-AUC: **0.468**
- Disjoint follow-up ROC-AUC: **0.384**

Complete statistical analyses, robustness checks, and figures are described in the accompanying paper.

---

## Repository

This repository serves as the paper landing page.

All executable notebooks and datasets are hosted on Kaggle.

```
attention-pivot-analysis/
├── README.md
├── LICENSE
├── requirements.txt
└── figures/
```

---

## Reproducing the Results

The simplest approach is to fork the Kaggle notebooks directly.

To run locally:

```bash
pip install -r requirements.txt
```

Download the notebooks and their corresponding output datasets from Kaggle, update the input paths, and execute the four stages sequentially.

---

## Data

The notebooks produce the complete intermediate datasets used in the paper, including

- Raw model generations
- Detected pivot candidates
- Final labeled pivot dataset
- Attention tensors
- Metadata
- Statistical summaries

The attention tensors from **both** experimental rounds are retained because the comparison between the two analyses is itself part of the reported result.

---

## Limitations

This study evaluates:

- one instruction-tuned language model (Qwen2.5-1.5B-Instruct),
- attention-based features only,
- elicited within-generation reasoning pivots.

The results should not be interpreted as evidence that self-correction cannot be represented internally by language models. They apply only to the attention-derived features examined in this study.

---

## Citation

The accompanying paper is currently in preparation. Citation information will be added here after publication.

---

## License

MIT License for code and analysis scripts.

Model outputs were generated using Qwen2.5-1.5B-Instruct and remain subject to the Qwen model license.