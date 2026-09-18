# Cochran's Q + McNemar Model Comparison

Compares 5 classifiers (DT, XGBoost, SVM, CNN, Transformer) on the same
bacterial strain classification task, to test whether their accuracy
differs in a statistically meaningful way.

## What it does

1. **Cochran's Q test** — a global test across all 5 models at once,
   answering: "does at least one model differ from the others in
   accuracy?"
2. **Pairwise McNemar tests** — if Cochran's Q is significant, this runs
   every pairwise comparison between models (10 pairs for 5 models),
   with Holm correction so the false-positive rate doesn't inflate
   across all those comparisons.
3. **(Optional) Per-strain Cochran's Q** — repeats step 1 separately
   within each true strain class, to check whether model disagreement
   is concentrated on specific strains.

Both tests only need one thing per sample per model: was the
prediction **correct or incorrect**. The actual algorithm (tree-based,
kernel-based, deep learning, whatever) doesn't matter — everything is
reduced to a binary correct/incorrect outcome before either test runs.

## Input requirements

- All 5 models must be evaluated on the **exact same test set** (same
  samples, same order). This is the one hard requirement — if any
  model was tested on a different split, it needs to be rerun on the
  shared test set.
- One CSV with:
  - a column for the **true strain label**
  - one column per model with that model's **predicted strain label**
    for each sample

Example (`predictions.csv`):

```
sample_id, true_label,  DT,         XGBoost,    SVM,        CNN,        Transformer
1,         E_coli,      E_coli,     E_coli,     Salmonella, E_coli,     E_coli
2,         Salmonella,  E_coli,     Salmonella, Salmonella, Salmonella, Salmonella
3,         Listeria,    Listeria,   Listeria,   Listeria,   Salmonella, Listeria
```

Column names don't have to match this exactly — just pass the right
names via the command-line flags below.

## Setup

```bash
pip install statsmodels pandas numpy scipy
```

## Usage

```bash
python cochrans_q_analysis.py \
    --csv predictions.csv \
    --true-col true_label \
    --model-cols DT XGBoost SVM CNN Transformer \
    --per-strain
```

Flags:
- `--csv` — path to your predictions file
- `--true-col` — name of the true-label column
- `--model-cols` — names of the predicted-label columns, one per model
- `--per-strain` — optional, adds the per-strain Cochran's Q breakdown
- `--correction` — multiple-comparison correction for the pairwise
  McNemar tests (default `holm`; also accepts `bonferroni`, `fdr_bh`,
  etc. — anything supported by `statsmodels.stats.multitest`)

Running the script with no `--csv` uses built-in synthetic demo data,
so you can see the expected output format before real data is ready.

## Output

The script prints results to the terminal and also saves:
- `correctness_matrix.csv` — the binary correct/incorrect matrix per
  model per sample
- `pairwise_mcnemar_results.csv` — all pairwise test results
- `per_strain_cochrans_q.csv` — only if `--per-strain` is used

## Notes for implementation

- The script only needs **predicted labels**, not saved model objects
  or probability scores — whatever format the predictions are already
  in should work, as long as they're for the same test set.
- If you'd rather feed in a matrix that's already correct/incorrect
  (0/1) instead of predicted labels, you can skip
  `build_correctness_matrix()` and call `cochrans_q()` /
  `pairwise_mcnemar()` directly on your own binary DataFrame.
