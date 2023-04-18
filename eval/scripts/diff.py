#! /usr/bin/env python3
import argparse
import glob
import os
import math

import pandas as pd

class colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'

    def from_change(change):
        if change <= -0.05:
            return colors.GREEN
        if change >= 0.05:
            return colors.RED

        return colors.RESET


def diff_two_files(old_filename, new_filename):
    old_df = pd.read_csv(old_filename, skipinitialspace=True)
    new_df = pd.read_csv(new_filename, skipinitialspace=True)

    if set(old_df) != set(new_df):
        print(f"\n{colors.RED}WARNING: input files have differing columns: {set(old_df).symmetric_difference(set(new_df))}{colors.RESET}\n")

    google_benchmark = {"name", "cpu_time"} <= set(old_df)
    velox_benchmark = {"query", "duration"} <= set(old_df)
    processed_velox_benchmark = {"query", "mean"} <= set(old_df)

    if google_benchmark:
        # only include the mean of multiple runs
        old_df = old_df[old_df.name.str.contains("mean")]
        new_df = new_df[new_df.name.str.contains("mean")]
        name_column = "name"
        result_column = "cpu_time"
    elif velox_benchmark:
        name_column = "query"
        result_column = "duration"
    elif processed_velox_benchmark:
        name_column = "query"
        result_column = "mean"
    else:
        assert False, "unexpected columns in input files"

    old_without_matching_new = []
    processed_old_names = []
    changes = []

    longest_bm_name = max(old_df[name_column].map(len).max(), new_df[name_column].map(len).max())

    def change_as_percent_str(change):
        return f"{change * 100:+.2f}%"

    for old_row in old_df.itertuples():
        bm_name = getattr(old_row, name_column)

        new_row = new_df.loc[new_df[name_column] == bm_name]
        assert len(new_row) == 0 or len(new_row) == 1

        if new_row.empty:
            old_without_matching_new.append(old_row)
            continue
        processed_old_names.append(bm_name)

        new_result = float(new_row[result_column])
        old_result = float(getattr(old_row, result_column))

        change = (new_result - old_result) / old_result
        changes.append(change)

        print(f"{colors.from_change(change)}{change_as_percent_str(change):7}{colors.RESET} {bm_name:{longest_bm_name}}  ({old_result:.2f} -> {new_result:.2f})")

    average_change = sum(changes) / len(changes)
    print(f"arithmetic mean: {colors.from_change(average_change)}{change_as_percent_str(average_change)}{colors.RESET}")

    geo_mean_of_new_performance = math.exp(sum((math.log(1 + change) for change in changes)) / len(changes))
    geo_mean_change = geo_mean_of_new_performance - 1
    print(f"geometric mean: {colors.from_change(geo_mean_change)}{change_as_percent_str(geo_mean_change)}{colors.RESET}")

    new_without_matching_old = new_df[~new_df[name_column].isin(processed_old_names)]

    if old_without_matching_new:
        print(f"\n{colors.RED}UNMATCHED OLD{colors.RESET}")
        for row in old_without_matching_new:
            print(f"{getattr(row, result_column):7.2f} {getattr(row, name_column)}")

    if not new_without_matching_old.empty:
        print(f"\n{colors.RED}UNMATCHED NEW{colors.RESET}")
        for row in new_without_matching_old.itertuples():
            print(f"{getattr(row, result_column):7.2f} {getattr(row, name_column)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Diff google benchmark results in csv format')
    parser.add_argument("old", help="CSV dir/file with the old benchmark results")
    parser.add_argument("new", nargs="?", default=".", help="CSV dir/file with the new benchmark results")
    args = parser.parse_args()

    if os.path.isfile(args.old):
        assert os.path.isfile(args.new), "either both are file or both are directory"
        diff_two_files(args.old, args.new)
        exit()

    assert os.path.isdir(args.old) and os.path.isdir(args.new), "either both are file or both are directory"
    print("Input directories given. Diffing all files in both.\n")

    for filename in sorted(glob.glob(f"{args.new}/*.csv")):
        new_path = filename
        old_path = args.old + "/" + os.path.basename(filename)
        if os.path.isfile(old_path):
            print(f"Diffing {old_path} (old) and {new_path} (new)")
            diff_two_files(old_path, new_path)
        else:
            print(f"{colors.RED}ERROR{colors.RESET}: No matching file found for {new_path} at {old_path}")

        print("\n" + "-"*80 + "\n")
