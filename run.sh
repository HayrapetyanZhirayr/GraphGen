#!/bin/bash
# set -e  # stop if any command fails

bash scripts/generate/generate_multi_hop.sh
bash scripts/generate/generate_aggregated.sh
bash scripts/generate/generate_cot.sh