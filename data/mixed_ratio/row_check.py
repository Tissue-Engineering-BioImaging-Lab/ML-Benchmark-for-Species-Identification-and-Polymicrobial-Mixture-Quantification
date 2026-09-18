#!/usr/bin/env python3
"""
Group CSV files by their header (first) row.

Drop this file into a folder of CSVs and run it (double-click, or `python
csv_headers.py` from anywhere) — with no arguments it scans the folder it is
sitting in, not the current working directory.

Usage:
    python csv_headers.py                       # the folder this script lives in
    python csv_headers.py data/                 # all .csv files in a directory
    python csv_headers.py a.csv b.csv c.csv     # explicit files
    python csv_headers.py -r                    # recurse into subdirectories
    python csv_headers.py --normalize           # ignore case/whitespace differences
    python csv_headers.py --ignore-order        # treat reordered columns as the same
    python csv_headers.py --diff                # show how each group differs from the largest
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


def pause(skip):
    """Keep a double-clicked terminal window open long enough to read the output."""
    if skip or not sys.stdin or not sys.stdin.isatty():
        return
    try:
        input("\nPress Enter to close...")
    except (EOFError, KeyboardInterrupt):
        pass


def collect_files(paths, recursive):
    """Expand directories into .csv files; keep explicit file paths as-is."""
    files = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            files.extend(sorted(p.rglob("*.csv") if recursive else p.glob("*.csv")))
        elif p.is_file():
            files.append(p)
        else:
            print(f"warning: no such file or directory: {p}", file=sys.stderr)
    # de-duplicate while preserving order
    seen, unique = set(), []
    for f in files:
        resolved = f.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(f)
    return unique


def read_header(path, encoding, delimiter):
    """Return the first row as a tuple of fields, or raise ValueError."""
    # utf-8-sig transparently strips a BOM if one is present.
    with open(path, "r", newline="", encoding=encoding, errors="replace") as fh:
        reader = csv.reader(fh, delimiter=delimiter) if delimiter else csv.reader(fh)
        for row in reader:
            return tuple(row)
    raise ValueError("file is empty")


def make_key(header, normalize, ignore_order):
    """Build the value used to decide whether two headers count as the same."""
    fields = header
    if normalize:
        fields = tuple(f.strip().lower() for f in fields)
    if ignore_order:
        fields = tuple(sorted(fields))
    return fields


def main():
    ap = argparse.ArgumentParser(
        description="Group CSV files by their first row and report which files share which header."
    )
    ap.add_argument("paths", nargs="*",
                    help="CSV files and/or directories (default: the folder this script is in)")
    ap.add_argument("-r", "--recursive", action="store_true",
                    help="recurse into subdirectories when given a directory")
    ap.add_argument("-e", "--encoding", default="utf-8-sig",
                    help="file encoding (default: utf-8-sig, which tolerates a BOM)")
    ap.add_argument("-d", "--delimiter", default=None,
                    help="field delimiter (default: comma). Use $'\\t' for TSV.")
    ap.add_argument("--normalize", action="store_true",
                    help="ignore surrounding whitespace and letter case in column names")
    ap.add_argument("--ignore-order", action="store_true",
                    help="treat headers with the same columns in a different order as identical")
    ap.add_argument("--diff", action="store_true",
                    help="for each minority group, show columns added/missing vs. the largest group")
    ap.add_argument("--no-pause", action="store_true",
                    help="don't wait for Enter before exiting (useful in scripts)")
    args = ap.parse_args()

    # No paths given -> scan the folder this script file lives in. Using __file__
    # rather than the working directory is what makes drag-and-drop + double-click
    # work: double-clicking often starts the process in some unrelated directory.
    if args.paths:
        targets = args.paths
        script_dir_mode = False
    else:
        targets = [Path(__file__).resolve().parent]
        script_dir_mode = True
        print(f"No folder given — scanning the folder this script is in:\n  {targets[0]}\n")

    files = collect_files(targets, args.recursive)
    if not files:
        print("No CSV files found.", file=sys.stderr)
        if script_dir_mode:
            print("Put this script in the same folder as your .csv files, or pass a "
                  "folder path as an argument.", file=sys.stderr)
        pause(args.no_pause)
        return 1

    groups = defaultdict(list)   # key -> [(path, original header), ...]
    problems = []                # (path, message)

    for path in files:
        try:
            header = read_header(path, args.encoding, args.delimiter)
        except ValueError as exc:
            problems.append((path, str(exc)))
            continue
        except OSError as exc:
            problems.append((path, f"could not read: {exc}"))
            continue
        groups[make_key(header, args.normalize, args.ignore_order)].append((path, header))

    def show(p):
        """Print bare filenames when scanning the script's folder, full paths otherwise."""
        if not script_dir_mode:
            return str(p)
        try:
            return str(p.resolve().relative_to(targets[0]))
        except ValueError:
            return str(p)

    # Largest group first, so the "normal" header is at the top and oddballs stand out.
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))

    print(f"Scanned {len(files)} file(s): {len(groups)} distinct header(s) found.\n")

    baseline = set(ordered[0][0]) if ordered else set()

    for i, (key, members) in enumerate(ordered, start=1):
        paths_, headers = zip(*members)
        print(f"=== Header group {i} — {len(members)} file(s), {len(headers[0])} column(s) ===")
        print("  Columns: " + ", ".join(headers[0]))

        # If normalization merged headers that aren't byte-identical, show the variants.
        variants = {h for h in headers}
        if len(variants) > 1:
            print("  (grouped despite these literal differences:)")
            for v in sorted(variants):
                print("    - " + ", ".join(v))

        if args.diff and i > 1:
            current = set(key)
            missing = baseline - current
            extra = current - baseline
            if missing:
                print("  Missing vs. group 1: " + ", ".join(sorted(missing)))
            if extra:
                print("  Extra vs. group 1:   " + ", ".join(sorted(extra)))
            if not missing and not extra:
                print("  Same column set as group 1 (order differs)")

        print("  Files:")
        for p in paths_:
            print(f"    {show(p)}")
        print()

    if problems:
        print("=== Skipped ===")
        for path, msg in problems:
            print(f"  {show(path)}: {msg}")
        print()

    pause(args.no_pause)
    return 0


if __name__ == "__main__":
    sys.exit(main())