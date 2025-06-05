#!/bin/bash

red='\033[0;31m'
green='\033[0;32m'
clear='\033[0m'

MODELS=(
    "Helsinki-NLP/opus-mt-it-en"
     "Helsinki-NLP/opus-mt-tc-big-it-en"
     "ModelSpace/GemmaX2-28-9B-v0.1"
     "ModelSpace/GemmaX2-28-2B-v0.1"
     "facebook/mbart-large-50-many-to-many-mmt"
     "google/madlad400-3b-mt"
     "google/madlad400-7b-mt"
     "facebook/nllb-200-distilled-600M"
     "facebook/nllb-200-distilled-1.3B"
     "facebook/nllb-200-3.3B"
     "mii-llm/maestrale-chat-v0.4-beta"
     "sapienzanlp/modello-italia-9b"
     "sapienzanlp/Minerva-7B-instruct-v1.0"
     "DeepMount00/Llama-3.1-8b-ITA"
     "swap-uniba/LLaMAntino-3-ANITA-8B-Inst-DPO-ITA"
    # "sag-uniroma2/extremITA-Camoscio-7b"
     "Unbabel/TowerInstruct-7B-v0.2"
     "LeonardPuettmann/PhiMaestra-3-Translation"
     "galatolo/cerbero-7b"
     "utter-project/EuroLLM-9B-Instruct"
     "utter-project/EuroLLM-1.7B-Instruct"
)
# DATASETS="flores tatoeba wmt24 ntrex"
DATASETS=(
    "wikinews"
)
FLORES_PATH="../datasets/flores200_dataset"
TATOEBA_PATH="../datasets/tatoeba"
WMT24_PATH="../datasets/wmt24pp"
NTREX_PATH="../datasets/NTREX-128"
WIKINEWS_PATH="../datasets/wikinews"
RESULTS_PATH="../results/it_en"
BATCH_SIZE=1
DEVICE="cuda"
NUM_BEAM=1
METRICS="bleu"
COMET_MODEL="Unbabel/wmt22-comet-da"

for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        # Set dataset path based on dataset name
        if [ "$DATASET" = "flores" ]; then
            DATAPATH="$FLORES_PATH"
        elif [ "$DATASET" = "tatoeba" ]; then
            DATAPATH="$TATOEBA_PATH"
        elif [ "$DATASET" = "wmt24" ]; then
            DATAPATH="$WMT24_PATH"
        elif [ "$DATASET" = "ntrex" ]; then
            DATAPATH="$NTREX_PATH"
        elif [ "$DATASET" = "wikinews" ]; then
            DATAPATH="$WIKINEWS_PATH"
        fi

        printf ${clear}"Translating with model: $MODEL on dataset: $DATASET\n"
        python3 translate_it_en.py \
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
            --num_beam $NUM_BEAM \
            --metrics "$METRICS" \
            --comet_model "$COMET_MODEL"
    done
    rm -rf /home/pranavkasela/.cache/
done

printf ${clear}"All translations and evaluations are done.\n"

METRICS="bleu,chrf,chrf++"
for DATASET in $DATASETS; do
    python3 evaluation_table.py \
        --results_path "$RESULTS_PATH" \
        --dataset_name "$DATASET" \
        --metrics "$METRICS" \
        --comet_model "$COMET_MODEL" \
        --sort_by "bleu"
done
