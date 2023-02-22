#! /usr/bin/env python3
import argparse
import glob
import os

import pandas as pd

class colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'

def diff_two_files(old_filename, new_filename):
    old_df = pd.read_csv(old_filename)
    new_df = pd.read_csv(new_filename)

    old_df = old_df[old_df.name.str.contains("mean")]
    new_df = new_df[new_df.name.str.contains("mean")]

    old_without_matching_new = []
    processed_old_names = []

    longest_bm_name = max(old_df.name.map(len).max(), new_df.name.map(len).max())

    for old_row in old_df.itertuples():
        new_row = new_df.loc[new_df["name"] == old_row.name]
        assert len(new_row) == 0 or len(new_row) == 1

        if new_row.empty:
            old_without_matching_new.append(old_row)
            continue
        processed_old_names.append(old_row.name)

        change = (float(new_row.cpu_time) - float(old_row.cpu_time)) / float(old_row.cpu_time)

        color = colors.RESET
        if change <= -0.05:
            color = colors.GREEN
        if change >= 0.05:
            color = colors.RED

        change_percent_str = f"{change * 100:+.2f}%"

        print(f"{color}{change_percent_str:7}{colors.RESET} {old_row.name:{longest_bm_name}}  ({float(old_row.cpu_time):.2f} -> {float(new_row.cpu_time):.2f})")

    new_without_matching_old = new_df[~new_df.name.isin(processed_old_names)]

    if old_without_matching_new:
        print(f"\n{colors.RED}UNMATCHED OLD{colors.RESET}")
        for row in old_without_matching_new:
            print(f"{row.cpu_time:7.2f} {row.name}")

    if not new_without_matching_old.empty:
        print(f"\n{colors.RED}UNMATCHED NEW{colors.RESET}")
        for row in new_without_matching_old.itertuples():
            print(f"{row.cpu_time:7.2f} {row.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Diff google benchmark results in csv format')
    parser.add_argument("old", nargs="?", default="", help="CSV file with the old benchmark results")
    parser.add_argument("new", nargs="?", default="", help="CSV file with the new benchmark results")
    args = parser.parse_args()

    if args.old:
        assert args.new != "", "require either zero or two input files"
        diff_two_files(args.old, args.new)
        exit()

    print("No input files given. Diffing all files in working directory against repository files.\n")

    repo_results_dir = os.path.normpath(os.path.relpath(os.path.dirname(os.path.realpath(__file__))) + "/../results")
    for filename in sorted(glob.glob("*.csv")):
        repo_path = repo_results_dir + "/" + filename
        if os.path.isfile(repo_path):
            print(f"Diffing {repo_path} (old) and {filename} (new)")
            diff_two_files(repo_path, filename)
        else:
            print(f"{colors.RED}ERROR{colors.RESET}: No matching file found for {filename} at {repo_path}")

        print("\n" + "-"*80 + "\n")
