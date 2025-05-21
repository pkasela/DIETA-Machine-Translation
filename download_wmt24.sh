#!/bin/bash

# Download the WMT 2024 dataset
# This script downloads the WMT 2024 dataset from the provided URL
# and extracts it to the specified directory.
# Usage: ./download_wmt24.sh <output_directory>
# Check if the output directory is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <output_directory>"
    exit 1
fi
# Set the output directory
OUTPUT_DIR=$1
# Create the output directory if it doesn't exist
mkdir -p $OUTPUT_DIR
# Download the dataset
git clone https://huggingface.co/datasets/google/wmt24pp --depth 1
# Remove the .git directory to save space
rm -rf wmt24pp/.git
# Move the dataset to the output directory
mv wmt24pp $OUTPUT_DIR
