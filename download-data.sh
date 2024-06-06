#!/bin/bash
# Downloads data from https://github.com/fchollet/ARC into a subfolder.

set -euo pipefail

arc_github_folder=arc_github_folder
raw_data_folder=data/raw

git clone https://github.com/fchollet/ARC "${arc_github_folder}"

mkdir -p "${raw_data_folder}"
cp -ri ${arc_github_folder}/data/{evaluation,training} "${raw_data_folder}"

rm -rf "${arc_github_folder}"
