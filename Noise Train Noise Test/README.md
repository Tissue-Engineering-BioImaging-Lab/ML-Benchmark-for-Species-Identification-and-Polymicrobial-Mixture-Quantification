# Cochran's Q + McNemar Model Comparison (noisy train / noisy test)

Same comparison as the `Clean Train Clean Test` folder, with one difference:
each model is trained **and** tested on a single dataset made of the clean
spectra plus a noisy copy of every one of them. Noise is present on both sides
of the split, so this measures how the models do when noise is part of the
training distribution rather than a shift at test time.

Compares 4 classifiers (RandomForest, XGBoost, CNN, Transformer) on the same
bacterial strain-ratio classification task.

## What it does

1. **Cochran's Q test** — a global test across all 4 models at once,
   answering: "does at least one model differ from the others in
   accuracy?"
2. **Pairwise McNemar tests** — if Cochran's Q is significant, this runs
   every pairwise comparison between models (6 pairs for 4 models),
   with Holm correction so the false-positive rate doesn't inflate
   across all those comparisons.
3. **(Optional) Per-strain Cochran's Q** — repeats step 1 separately
   within each true strain-ratio class, to check whether model disagreement
   is concentrated on specific ratios.

Both tests only need one thing per sample per model: was the
prediction **correct or incorrect**. The actual algorithm (tree-based,
kernel-based, deep learning, whatever) doesn't matter — everything is
reduced to a binary correct/incorrect outcome before either test runs.

## How the combined dataset is built

Each notebook loads the 61,754 clean spectra from `../data/mixed_ratio`, makes
a noisy copy of all of them with `add_white_noise_to_features(noise_level=0.05,
random_state=42, relative=True)` — Gaussian white noise scaled per row by that
row's spectral standard deviation — and concatenates the two:

```
combined_df = pd.concat([clean_df, noisy_df], ignore_index=True)   # 123,508 rows
```

Rows `0 .. 61753` are the clean spectra and rows `61754 .. 123507` are their
noisy counterparts, so `sample_id` tells you which half a sample came from.
Only then does the 80/20 `train_test_split(..., test_size=0.2, random_state=42,
stratify=y)` happen, which is why both training and test sets contain a mix of
clean and noisy rows (the test split works out to roughly 12,300 clean and
12,400 noisy).

Because the concat order, split parameters and seed are identical in all four
notebooks, the 24,702 test rows are the same samples in every notebook — which
is what the paired tests require.

One caveat worth knowing when you write this up: RF and XGBoost add the noise
to the **raw** features, while the CNN and Transformer add it to features cast
to `float32` and then standardize the combined dataset with `StandardScaler`
(that's where each notebook's pipeline already sits). Same seed and same
per-row relative scaling in both cases, so the perturbation is equivalent up to
float32 rounding — but it is not the bit-identical perturbation across all four
models. The paired tests only require the same test *samples*, which holds.

## Per-notebook results

Every notebook reports the same things as the `Clean Train Clean Test` set:

| | RF | XGBoost | CNN | Transformer |
|---|---|---|---|---|
| Train vs test accuracy + gap | ✓ | ✓ | ✓ (§7b) | ✓ (§7b) |
| 5-fold cross-validation | ✓ | ✓ | ✓ (§8) | ✓ (§8) |
| Learning curve | ✓ (sklearn) | ✓ (sklearn) | ✓ (per-epoch, §6b) | ✓ (per-epoch, §6b) |
| Confusion matrix | ✓ | ✓ | ✓ | ✓ |
| Classification report | ✓ | ✓ | ✓ | ✓ |
| Weighted F1 | ✓ | ✓ | — | — |
| ROC (one-vs-rest) | — | — | ✓ | ✓ |
| `model_predictions.csv` export | ✓ | ✓ | ✓ | ✓ |

Cross-validation in every notebook runs on the combined clean + noisy dataset,
matching the data the model is trained on.

## Input requirements

- All 4 models must be evaluated on the **exact same test set** (same
  samples, same order). This is the one hard requirement — if any
  model was tested on a different split, it needs to be rerun on the
  shared test set.
- One CSV with:
  - a column for the **true strain label**
  - one column per model with that model's **predicted strain label**
    for each sample

`model_predictions.csv` is written by the notebooks themselves and looks
like this:

```
sample_id, true_label,        RandomForest,      XGBoost,           CNN,               Transformer
9,         1:9 | Staph : PAO, 1:9 | Staph : PAO, 1:9 | Staph : PAO, 1:9 | Staph : PAO, 1:9 | Staph : PAO
16,        1:9 | Staph : PAO, 1:9 | Staph : PAO, 3:7 | Staph : PAO, 1:9 | Staph : PAO, 1:9 | Staph : PAO
```

## Setup

```bash
pip install statsmodels pandas numpy scipy
```

## Usage

Run the four notebooks first — each one appends its own column to the
shared `model_predictions.csv`:

1. `RF_mixed_combined.ipynb` → `RandomForest`
2. `XG_mixed_combined.ipynb` → `XGBoost`
3. `CNN_mixed_combined.ipynb` → `CNN`
4. `Transformer_mixed_combined.ipynb` → `Transformer`

The `OLD *.ipynb` files are the previous generation of these notebooks, kept
for reference. They run the same regime but with the older model definitions,
and they write to the same `model_predictions.csv` — so don't mix the two sets
in one run.

Order doesn't matter, and re-running a notebook overwrites just its own
column. The export cell raises an error if the `sample_id` set or any
`true_label` disagrees with what's already in the CSV, so a mismatched
split can't silently corrupt the comparison. If you change the split or
the noise settings, delete `model_predictions.csv` and rerun all four.

Note that the `sample_id` set is unchanged from the older notebooks, so the
guard will *not* catch a half-old, half-new CSV. Delete
`model_predictions.csv` before the first full run with the updated models.

The two Keras notebooks can be run headlessly on the GPU:

```bash
.\train_gpu_headless.ps1 CNN_mixed_combined.ipynb
```

Then run the stats:

```bash
python cochrans_q_analysis.py --csv model_predictions.csv --true-col true_label --model-cols RandomForest XGBoost CNN Transformer --per-strain
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

To keep a copy of the terminal output the way the clean-test folder does:

```bash
python cochrans_q_analysis.py --csv model_predictions.csv --true-col true_label --model-cols RandomForest XGBoost CNN Transformer --per-strain > test_terminal_output.txt
```

Since `sample_id < 61754` marks a clean row, you can also split
`correctness_matrix.csv` on that boundary to compare clean-row and noisy-row
accuracy within the same trained model.

## Notes for implementation

- The script only needs **predicted labels**, not saved model objects
  or probability scores — whatever format the predictions are already
  in should work, as long as they're for the same test set.
- If you'd rather feed in a matrix that's already correct/incorrect
  (0/1) instead of predicted labels, you can skip
  `build_correctness_matrix()` and call `cochrans_q()` /
  `pairwise_mcnemar()` directly on your own binary DataFrame.
- `target_encoding_map.json` and `feature_map.json` are regenerated by
  each notebook's preprocessing cells, so they don't need to be checked
  in ahead of a run.
- Every notebook must read `../data/mixed_ratio`. An earlier CNN notebook read
  `../data` recursively, which pulled in the single-species folders and gave it
  7 classes over a different row count than the other three — with that, the
  four models are not on a shared test set and the paired tests cannot be run.
  All four current notebooks read `../data/mixed_ratio`.
- `XG_mixed_combined.ipynb` was built from the clean-test `XG_mixed.ipynb` by
  applying the same clean + noisy concatenation the other three use, since the
  refreshed notebooks arrived without an XGBoost version for this folder. It
  ships with no stored outputs — run it once to populate them.
