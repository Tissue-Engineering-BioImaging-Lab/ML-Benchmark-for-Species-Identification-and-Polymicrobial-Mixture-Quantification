"""
combine_csvs_by_prefix.py

Combine Raman spectra CSV files that belong to the same classification into a
single CSV. Files are grouped by which y name appears in their file name
(Corn, E.coli, FAE, Malt, PAO1, STAPH), so every CSV of the same species is
stacked together. This mirrors the notebooks' `targets` matching logic.

Like the notebook's `read_all_csvs_to_df`, each CSV's first row is a header row
(the wavelength axis), so when stacking a group the first file keeps its header
row and every file added after it has its first row skipped.

For each group, the combined data is written out as "<category>_final.csv".

Usage:
    python combine_csvs_by_prefix.py [input_folder] [-o OUTPUT_FOLDER]

Examples:
    python combine_csvs_by_prefix.py
    python combine_csvs_by_prefix.py ../data/six_single_species -o ../data/combined
"""

import os
import argparse
from collections import defaultdict

import pandas as pd


# category names to match against each file name; a file joins the first
# category whose name appears in it (order matters on ties)
CATEGORIES = ["Corn", "E.coli", "FAE", "Malt", "PAO1", "STAPH"]

OUTPUT_SUFFIX = "_final"  # combined files are written as "<category>_final.csv"

# default input folder, matching the notebooks (os.path.join("..", "data", "six_single_species"))
DEFAULT_INPUT_FOLDER = os.path.join("..", "data", "six_single_species")


def match_category(file_name):
    """Return the first category whose name appears in file_name, or None."""
    return next((cat for cat in CATEGORIES if cat in file_name), None)


def group_files_by_category(folder_path):
    """Return {category: [file_name, ...]} for the .csv files in folder_path."""
    groups = defaultdict(list)

    for file_name in sorted(os.listdir(folder_path)):
        if not file_name.endswith(".csv"):
            continue
        # skip previously combined outputs so re-running doesn't fold them back in
        if file_name.endswith(f"{OUTPUT_SUFFIX}.csv"):
            continue

        category = match_category(file_name)
        if category is None:
            print(f"  [skip] no category matched: {file_name}")
            continue

        groups[category].append(file_name)

    return groups


def combine_group(folder_path, file_names):
    """Stack one group's CSVs, skipping the header row of every file after the first."""
    frames = []

    for i, file_name in enumerate(file_names):
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path, header=None)

        # each csv's first row is a header row: keep it for the first file in the
        # group, skip it for every file added after that
        if i != 0:
            df = df.iloc[1:].reset_index(drop=True)

        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def combine_csvs_by_category(input_folder, output_folder=None):
    """Combine every group of same-category CSVs in input_folder into one CSV each."""
    output_folder = output_folder or input_folder
    os.makedirs(output_folder, exist_ok=True)

    groups = group_files_by_category(input_folder)
    if not groups:
        print(f"No matching CSV files found in: {input_folder}")
        return

    for category, file_names in groups.items():
        combined_df = combine_group(input_folder, file_names)

        output_name = f"{category}{OUTPUT_SUFFIX}.csv"
        output_path = os.path.join(output_folder, output_name)

        # write without pandas' own header/index so the original header row
        # (row 0 from the first file) is preserved exactly
        combined_df.to_csv(output_path, index=False, header=False)

        print(f"[{category}] combined {len(file_names)} file(s), "
              f"{len(combined_df)} rows -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine CSVs that share the same category name in their file name."
    )
    parser.add_argument(
        "input_folder", nargs="?", default=DEFAULT_INPUT_FOLDER,
        help="Folder containing the CSV files to combine "
             "(default: ../data/six_single_species, matching the notebooks)."
    )
    parser.add_argument(
        "-o", "--output-folder", default=None,
        help="Where to write combined CSVs (default: same as input folder)."
    )
    args = parser.parse_args()

    combine_csvs_by_category(args.input_folder, args.output_folder)
