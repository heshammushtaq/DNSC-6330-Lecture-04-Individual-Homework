# DNSC 6330 — Lecture 04 Individual Homework

## From Accuracy to Accountability: Stress Testing a Predictive Model

This notebook is the individual coding component of **Lecture 04 (Robustness, Generalization, and Dataset Drift)** for **DNSC 6330: Responsible Machine Learning**.

---

## Purpose

The analysis audits two classifiers trained on the **ProPublica COMPAS dataset**, a logistic regression and a gradient-boosted tree — against the responsible-ML framework taught in Lecture 04. The goal is to move past aggregate test-set accuracy and ask whether either model would be defensible if deployed.

The notebook is organized in **five parts**, each addressing a separate failure mode covered in the lecture:

| Part | Focus | Key Methods |
|------|-------|-------------|
| **A** | Distribution drift | PSI, KS, MMD, score-distribution comparison |
| **B** | Generalization | Train vs test AUC / accuracy / log loss; permutation importance |
| **C** | Spurious-correlation probe | Counterfactual swaps on race, gender, charge degree |
| **D** | Robustness | Stress sweep on `priors_count`, ICE curves, sensitivity index |
| **E** | Slice-based evaluation | Per-slice metrics across race, gender, age, and charge type |

Each part ends with an **Interpretation** and an **Audit Takeaway**. The interpretation describes what the numbers show; the audit takeaway names actions a model risk reviewer would take.

---

## Python Libraries Used

- **`pandas`** and **`numpy`** — data manipulation
- **`scikit-learn`** — logistic regression and gradient-boosted tree pipelines, train/test split, preprocessor, permutation importance
- **`scipy.stats.ks_2samp`** — Kolmogorov-Smirnov drift test
- **`statsmodels.formula.api`** — baseline GLM specification
- **`matplotlib`** — ICE curve plots
- **Custom helper functions** defined inline in the notebook for PSI, MMD (RBF kernel), counterfactual swap shifts, stress testing, slice metrics, and the global sensitivity index

---

## How to Reproduce

The notebook is a **single self-contained file**. It pulls the COMPAS Broward County two-year recidivism extract directly from the ProPublica GitHub repository, so no local data file is required.

### Option 1: Google Colab

Open the notebook in Colab and run all cells from the top. All required libraries are already available in the default Colab environment.

### Option 2: Local Jupyter

Clone the repo and run:

```bash
pip install pandas numpy scikit-learn scipy statsmodels matplotlib jupyter
jupyter notebook DNSC6330_Lecture04_Individual_Hesham_Mushtaq_G23607459.ipynb
```

Then run all cells. The notebook is **deterministic** given the random seed of `42` in the train/test split and the gradient-boosted tree, so reruns reproduce the same numbers.

---

## Interpretation of Results

### Part A — Distribution Drift

PSI on numeric features (**0.0104**, **0.0008**), KS p-values (**0.21**, **0.99**), and MMD² (**-0.000272**) show no detectable drift between train and test. This is a baseline check rather than a clearance, since the two samples come from one random split rather than a real deployment gap.

### Part B — Generalization

The **logistic regression** generalizes cleanly with no measurable train/test gap. The **gradient-boosted tree** shows a **0.024 AUC gap**, a **0.038 log-loss gap** moving the wrong way, and relies about **twice as heavily** on `race_factor`. Test AUC is **0.8329** for the LR and **0.8309** for the GBT — statistically tied.

### Part C — Spurious-Correlation Probe

Counterfactual attribute swaps move predicted probability the most for race in both models (**0.0806** LR, **0.0976** GBT). The GBT shifts more on every attribute, with the gender response growing from **0.0251** on the LR to **0.0888** on the GBT. Both models fail conditional invariance.

### Part D — Robustness

Both models respond strongly to perturbations of `priors_count`. ICE curves are **smooth and parallel** for the LR but **jagged** for the GBT. The global sensitivity index is slightly higher for the LR (**0.0488 vs 0.0412**), but the LR's per-case behavior is more documentable.

### Part E — Slice-Based Evaluation

Aggregate accuracy hides large slice disparities. African-American defendants have a **3.5x higher false-positive rate** than Caucasian defendants under both models. The under-25 slice has **FPR of 0.66** — the largest disparate error in the table. Both models produce the same disparity pattern, so model substitution is not the lever.

### Overall Recommendation

**Prefer the logistic regression as the deployment candidate** on this evidence:

- Smaller train/test gap
- Equivalent test AUC
- Lower reliance on `race_factor`
- More documentable response curves

The GBT does win on global sensitivity index and on the racial gap in positive prediction rate (**2.57x vs 3.05x**), but the responsible-ML weighting puts stability and explainability ahead of those margins. The deeper issue — that **both models reproduce the same disparity pattern** — points to feature set and label definition as the layer where this audit's findings actually live.

---

## Files

- **`DNSC6330_Lecture04_Individual_Hesham_Mushtaq.ipynb`** — the homework notebook
- **`README.md`** — this file

