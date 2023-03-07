#! /usr/bin/env python3
import argparse
import os
import re
from statistics import mean, stdev
from contextlib import redirect_stdout
from collections import defaultdict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert (repeated) velox tpch benchmark output to a usable csv file')
    parser.add_argument("velox_output_file", help="Velox tpch output file")
    args = parser.parse_args()

    input_path = args.velox_output_file
    output_path = input_path.rsplit(".", 1)[0] + ".csv"

    if os.path.exists(output_path):
        if input(f"WARNING: Output file path {output_path} exists. Are you sure you want to continue? y/N: ") != "y":
            exit()

    results_by_query = defaultdict(list)
    result_line_regex = re.compile(r"^(q\d+)\s+(\d+.\d+)ms\s+\d+.\d+\n$")

    with open(input_path, "r") as input_file:
        for line in input_file:
            match_result = result_line_regex.match(line)
            if not match_result:
                continue

            results_by_query[match_result.group(1)].append(float(match_result.group(2)))

    with open(output_path, "w") as output_file:
        with redirect_stdout(output_file):
            print("query,    mean,     min,     max,  stddev,      cv")
            for query, results in results_by_query.items():
                mean_ = mean(results)
                min_ = min(results)
                max_ = max(results)
                stddev_ = stdev(results)
                cv_percent = f"{stddev_ / mean_ * 100:.2f}%"
                print(f"{query:5},{mean_:>8.2f},{min_:>8.2f},{max_:>8.2f},{stddev_:>8.2f},{cv_percent:>8}")
