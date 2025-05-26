#!/bin/bash

# Download the WMT 2024 dataset
# This script downloads the WMT 2024 dataset from the provided URL
# and extracts it to the specified directory.
# Usage: ./download_ntrex.sh <output_directory>
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
git clone https://github.com/MicrosoftTranslator/NTREX.git --depth 1
# Remove the .git directory to save space
rm -rf NTREX/.git
# Move the dataset to the output directory
mv NTREX/NTREX-128 $OUTPUT_DIR
# Remove the original NTREX directory
rm -rf NTREX