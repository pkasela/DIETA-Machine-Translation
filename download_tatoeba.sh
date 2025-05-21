#!/bin/bash

# Download the Tatoeba dataset
# This script downloads the Tatoeba dataset from the provided URL
# and extracts it to the specified directory.
# Usage: ./download_tatoeba.sh <output_directory>
# Check if the output directory is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <output_directory>"
    exit 1
fi
# Set the output directory
OUTPUT_DIR=$1
# Create the output directory if it doesn't exist
mkdir -p $OUTPUT_DIR/tatoeba
# Download the dataset
wget -O tatoeba_test.tar https://object.pouta.csc.fi/Tatoeba-Challenge-devtest/test.tar
# Extract the dataset
tar -xf tatoeba_test.tar -C $OUTPUT_DIR/tatoeba
# Remove the tar file
rm tatoeba_test.tar