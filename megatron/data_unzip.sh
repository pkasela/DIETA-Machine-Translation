#!/bin/bash

INPUT_PATH="/leonardo_work/IscrB_modMNMT/ale/data"
OUTPUT_FILE="/workspace/data/raw/allfiltered_merged_raw.txt"

for file in $INPUT_PATH/allfiltered*; do
  echo $file
  gzip -cd $file >> $OUTPUT_FILE
done