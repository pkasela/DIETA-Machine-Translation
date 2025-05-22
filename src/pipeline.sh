#!/bin/bash

red='\033[0;31m'
green='\033[0;32m'
clear='\033[0m'

MODELS=(
    "Helsinki-NLP/opus-mt-en-it"
    "Helsinki-NLP/opus-mt-tc-big-en-it"
    "ModelSpace/GemmaX2-28-9B-v0.1"
    "ModelSpace/GemmaX2-28-2B-v0.1"
    "facebook/mbart-large-50-many-to-many-mmt"
    "jbochi/madlad400-3b-mt"
    "google/madlad400-3b-mt"
    "jbochi/madlad400-7b-mt-bt"
    "facebook/nllb-200-distilled-600M"
    "facebook/nllb-200-distilled-1.3B"
    "facebook/nllb-200-3.3B"
)
DATASETS="wmt24 flores tatoeba"

FLORES_PATH="../datasets/flores200_dataset"
TATOEBA_PATH="../datasets/tatoeba"
WMT24_PATH="../datasets/wmt24pp"
RESULTS_PATH="../results"
BATCH_SIZE=8
DEVICE="cuda:1"
NUM_BEAM=5
METRICS="bleu,chrf,chrf++"
COMET_MODEL="Unbabel/wmt22-comet-da"

for MODEL in "${MODELS[@]}"; do
    for DATASET in $DATASETS; do
        # Set dataset path based on dataset name
        if [ "$DATASET" = "flores" ]; then
            DATAPATH="$FLORES_PATH"
        elif [ "$DATASET" = "tatoeba" ]; then
            DATAPATH="$TATOEBA_PATH"
        elif [ "$DATASET" = "wmt24" ]; then
            DATAPATH="$WMT24_PATH"
        fi

        printf ${clear}"Translating with model: $MODEL on dataset: $DATASET\n"
        python3 translate.py \
            --model_name "$MODEL" \
            --dataset_name "$DATASET" \
            --dataset_path "$DATAPATH" \
            --results_path "$RESULTS_PATH" \
            --batch_size $BATCH_SIZE \
            --num_beam $NUM_BEAM \
            --device $DEVICE

        printf ${red}"Evaluating model: $MODEL on dataset: $DATASET\n"${green}
        python3 evaluation.py \
            --results_path "$RESULTS_PATH" \
            --model_name "$MODEL" \
            --dataset_name "$DATASET" \
            --NUM_BEAM $NUM_BEAM \
            --metrics "$METRICS" \
            --comet_model "$COMET_MODEL"
        
    done
done

printf ${clear}"All translations and evaluations are done.\n"

METRICS="bleu,chrf,chrf++,comet"
for DATASET in $DATASETS; do
    python3 evaluation_table.py \
        --results_path "$RESULTS_PATH" \
        --dataset_name "$DATASET" \
        --metrics "$METRICS" \
        --comet_model "$COMET_MODEL"
done