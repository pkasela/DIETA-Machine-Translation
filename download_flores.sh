#!/bin/bash

# Download the FLORES-200 dataset
# This script downloads the FLORES-200 dataset from the provided URL
# and extracts it to the specified directory.
# Usage: ./download_flores.sh <output_directory>
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
wget -O $OUTPUT_DIR/flores200.tar.gz https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz
# Extract the dataset
tar -xzf $OUTPUT_DIR/flores200.tar.gz -C $OUTPUT_DIR
# Remove the tar.gz file
rm $OUTPUT_DIR/flores200.tar.gz
# Print the location of the extracted dataset
echo "FLORES-200 dataset downloaded and extracted to $OUTPUT_DIR"
