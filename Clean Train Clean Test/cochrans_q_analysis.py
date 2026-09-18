"""
Cochran's Q Test + Pairwise McNemar Follow-up
For comparing multiple classifiers (e.g., DT, XGBoost, SVM, CNN, Transformer)
on the same multiclass classification task (bacterial strain ID).

WHAT THIS DOES
---------------
1. Takes per-sample predictions from each model + the true label.
2. Converts each model's predictions into a binary correct(1)/incorrect(0)
   outcome per sample (this is what Cochran's Q actually operates on --
   the number of classes/strains doesn't matter at this stage).
3. Runs the global Cochran's Q test: "do the models differ in overall
   accuracy across the same samples?"
4. If significant, runs all pairwise McNemar tests as a post-hoc, with
   Holm correction for multiple comparisons.
5. Optionally repeats Cochran's Q PER STRAIN (i.e., only on samples whose
   true label is that strain) to see if model disagreement is
   strain-specific.

EXPECTED INPUT FORMAT (CSV)
----------------------------
One row per test sample, one column per model containing that model's
PREDICTED label, plus a column with the TRUE label. Example:

    sample_id, true_label, DT, XGBoost, SVM, CNN, Transformer
    1,         E_coli,     E_coli,   E_coli,   Salmonella, E_coli,   E_coli
    2,         Salmonella, E_coli,   Salmonella, Salmonella, Salmonella, Salmonella
    ...

USAGE
-----
    python cochrans_q_analysis.py --csv predictions.csv \
        --true-col true_label \
        --model-cols DT XGBoost SVM CNN Transformer

If no --csv is given, the script runs on synthetic demo data so you can
see the expected output format before your real data is ready.
"""

import argparse
import itertools
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.stats.contingency_tables import mcnemar


# ----------------------------------------------------------------------
# Core stats functions
# ----------------------------------------------------------------------

def build_correctness_matrix(df: pd.DataFrame, true_col: str, model_cols: list[str]) -> pd.DataFrame:
    """Convert per-sample predicted labels into a binary correct/incorrect matrix."""
    correctness = pd.DataFrame(index=df.index)
    for col in model_cols:
        correctness[col] = (df[col] == df[true_col]).astype(int)
    return correctness


def cochrans_q(binary_matrix: pd.DataFrame):
    """
    Cochran's Q test on an n_samples x k_models binary (0/1) matrix.

    Returns dict with Q statistic, degrees of freedom, and p-value.
    """
    X = binary_matrix.to_numpy(dtype=float)
    n, k = X.shape

    col_sums = X.sum(axis=0)   # total correct per model
    row_sums = X.sum(axis=1)   # total correct per sample (across models)
    N = X.sum()                # grand total correct

    numerator = k * (k - 1) * np.sum((col_sums - N / k) ** 2)
    denominator = k * N - np.sum(row_sums ** 2)

    if denominator == 0:
        # happens if every sample is either all-correct or all-incorrect
        # across models -> no disagreement to test
        return {"Q": np.nan, "df": k - 1, "p_value": np.nan,
                "note": "No disagreement between models (denominator=0); Q undefined."}

    Q = numerator / denominator
    df = k - 1
    p_value = chi2.sf(Q, df)

    return {"Q": Q, "df": df, "p_value": p_value}


def pairwise_mcnemar(binary_matrix: pd.DataFrame, correction: str = "holm"):
    """
    Run McNemar's test for every pair of models on the same binary matrix.
    Uses exact McNemar (recommended for typical sample sizes) with
    Holm-Bonferroni correction across all pairwise comparisons.
    """
    from statsmodels.stats.multitest import multipletests

    models = binary_matrix.columns.tolist()
    pairs = list(itertools.combinations(models, 2))

    raw_pvals = []
    records = []

    for m1, m2 in pairs:
        c1 = binary_matrix[m1].to_numpy()
        c2 = binary_matrix[m2].to_numpy()

        # 2x2 contingency table of (m1 correct/incorrect) x (m2 correct/incorrect)
        both_correct = np.sum((c1 == 1) & (c2 == 1))
        m1_only = np.sum((c1 == 1) & (c2 == 0))
        m2_only = np.sum((c1 == 0) & (c2 == 1))
        both_wrong = np.sum((c1 == 0) & (c2 == 0))

        table = [[both_correct, m1_only],
                 [m2_only, both_wrong]]

        # exact=True uses the binomial exact test, recommended when
        # the discordant count (m1_only + m2_only) is small (<25 or so)
        use_exact = (m1_only + m2_only) < 25
        result = mcnemar(table, exact=use_exact, correction=not use_exact)

        raw_pvals.append(result.pvalue)
        records.append({
            "model_1": m1,
            "model_2": m2,
            "model_1_only_correct": m1_only,
            "model_2_only_correct": m2_only,
            "statistic": result.statistic,
            "p_raw": result.pvalue,
        })

    reject, p_adj, _, _ = multipletests(raw_pvals, method=correction)

    out = pd.DataFrame(records)
    out["p_adj"] = p_adj
    out["significant"] = reject
    return out


def per_strain_cochrans_q(df: pd.DataFrame, true_col: str, model_cols: list[str]):
    """Run Cochran's Q separately within each true strain class."""
    results = []
    for strain, sub in df.groupby(true_col):
        cm = build_correctness_matrix(sub, true_col, model_cols)
        res = cochrans_q(cm)
        res["strain"] = strain
        res["n_samples"] = len(sub)
        results.append(res)
    return pd.DataFrame(results)[["strain", "n_samples", "Q", "df", "p_value"]]


# ----------------------------------------------------------------------
# Demo / synthetic data (used only if no --csv is passed)
# ----------------------------------------------------------------------

def make_demo_data(n=200, seed=42):
    rng = np.random.default_rng(seed)
    strains = ["E_coli", "Salmonella", "Listeria", "Staph_aureus"]
    true_label = rng.choice(strains, size=n)

    # simulate models with different accuracy levels + some correlated errors
    def simulate(acc):
        correct = rng.random(n) < acc
        pred = np.where(correct, true_label,
                         rng.choice(strains, size=n))
        return pred

    df = pd.DataFrame({
        "sample_id": np.arange(1, n + 1),
        "true_label": true_label,
        "DT": simulate(0.78),
        "XGBoost": simulate(0.87),
        "SVM": simulate(0.81),
        "CNN": simulate(0.90),
        "Transformer": simulate(0.93),
    })
    return df


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cochran's Q + pairwise McNemar for multiclass classifiers")
    parser.add_argument("--csv", type=str, default=None, help="Path to predictions CSV")
    parser.add_argument("--true-col", type=str, default="true_label", help="Column name for the true label")
    parser.add_argument("--model-cols", nargs="+", default=None,
                         help="Column names for each model's predictions")
    parser.add_argument("--correction", type=str, default="holm",
                         help="Multiple comparison correction method (holm, bonferroni, fdr_bh, etc.)")
    parser.add_argument("--per-strain", action="store_true",
                         help="Also run Cochran's Q separately within each strain")
    args = parser.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
        if args.model_cols is None:
            parser.error("--model-cols is required when using --csv")
        model_cols = args.model_cols
        true_col = args.true_col
    else:
        print("No --csv provided -- running on synthetic demo data so you can preview the output.\n")
        df = make_demo_data()
        model_cols = ["DT", "XGBoost", "SVM", "CNN", "Transformer"]
        true_col = "true_label"

    print(f"Loaded {len(df)} samples, comparing models: {model_cols}\n")

    # 1. Global Cochran's Q
    binary_matrix = build_correctness_matrix(df, true_col, model_cols)
    print("Per-model accuracy on this sample set:")
    print((binary_matrix.mean().round(4) * 100).astype(str) + "%")
    print()

    q_result = cochrans_q(binary_matrix)
    print("=" * 60)
    print("GLOBAL COCHRAN'S Q TEST")
    print("=" * 60)
    if np.isnan(q_result.get("Q", np.nan)):
        print(q_result.get("note", "Q undefined."))
    else:
        print(f"Q statistic : {q_result['Q']:.4f}")
        print(f"df          : {q_result['df']}")
        print(f"p-value     : {q_result['p_value']:.6f}")
        if q_result["p_value"] < 0.05:
            print("-> Significant: at least one model differs from the others.")
            print("   Proceeding to pairwise McNemar post-hoc tests...\n")
        else:
            print("-> Not significant: no evidence models differ in accuracy overall.")
            print("   (Pairwise post-hoc tests are optional/exploratory only.)\n")

    # 2. Pairwise McNemar post-hoc
    print("=" * 60)
    print(f"PAIRWISE McNEMAR TESTS ({args.correction}-corrected)")
    print("=" * 60)
    pairwise_results = pairwise_mcnemar(binary_matrix, correction=args.correction)
    pd.set_option("display.width", 120)
    print(pairwise_results.to_string(index=False))
    print()

    # 3. Optional per-strain breakdown
    if args.per_strain:
        print("=" * 60)
        print("PER-STRAIN COCHRAN'S Q")
        print("=" * 60)
        strain_results = per_strain_cochrans_q(df, true_col, model_cols)
        print(strain_results.to_string(index=False))
        print()

    # Save outputs
    binary_matrix.to_csv("correctness_matrix.csv", index=False)
    pairwise_results.to_csv("pairwise_mcnemar_results.csv", index=False)
    if args.per_strain:
        strain_results.to_csv("per_strain_cochrans_q.csv", index=False)
    print("Saved: correctness_matrix.csv, pairwise_mcnemar_results.csv"
          + (", per_strain_cochrans_q.csv" if args.per_strain else ""))


if __name__ == "__main__":
    sys.exit(main())
