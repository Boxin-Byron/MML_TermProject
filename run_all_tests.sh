#!/bin/bash
set -e

# Set HF Mirror for offline/restricted environment
export HF_ENDPOINT=https://hf-mirror.com

# IIW-400
echo "Running IIW-400 Captioning..."
bash scripts/llava/captioning_iiw400.sh

echo "Running IIW-400 CLAIR Evaluation..."
bash scripts/clair_iiw_eval.sh

# DOCCI
echo "Running DOCCI Captioning..."
bash scripts/llava/captioning_docci.sh

echo "Running DOCCI CLAIR Evaluation..."
bash scripts/clair_docci_eval.sh

# # COCO Tests
# echo "Running COCO Captioning..."
# bash scripts/llava/captioning_coco.sh

# echo "Running COCO Captioning (LLaVA-NeXT)..."
# bash scripts/llava_next/captioning_coco.sh

# bash scripts/chair_eval.sh


# # LLaVA-NeXT Tests

# # IIW-400 for LLaVA-NeXT
# echo "Running IIW-400 Captioning (LLaVA-NeXT)..."
# bash scripts/llava_next/captioning_iiw400.sh

# echo "Running IIW-400 CLAIR Evaluation (LLaVA-NeXT)..."
# # Evaluation for Next using direct command to avoid script modification
# export EXP_NAME="IIW-400_llava-next"
# export ANNOTATION_FILE="./dataset/iiw400/data.jsonl"
# export ANSWER_FOLDER="./results"
# export SAVE_FOLDER="./results"
# # Ensure OPENAI_API_KEY is set in your environment or replace here
# # export OPENAI_API_KEY="YOUR_KEY_HERE" 

# python -m clair \
#     --annotation_file "$ANNOTATION_FILE" \
#     --answer_folder "$ANSWER_FOLDER" \
#     --save_folder "$SAVE_FOLDER" \
#     --openai_api_key "$OPENAI_API_KEY" \
#     --experiment_name "$EXP_NAME" \
#     --data_type iiw

# # DOCCI for LLaVA-NeXT
# echo "Running DOCCI Captioning (LLaVA-NeXT)..."
# bash scripts/llava_next/captioning_docci.sh

# echo "Running DOCCI CLAIR Evaluation (LLaVA-NeXT)..."
# export EXP_NAME="DOCCI_llava-next"
# export ANNOTATION_FILE="./dataset/docci/docci_descriptions.jsonlines"

# python -m clair \
#     --annotation_file "$ANNOTATION_FILE" \
#     --answer_folder "$ANSWER_FOLDER" \
#     --save_folder "$SAVE_FOLDER" \
#     --openai_api_key "$OPENAI_API_KEY" \
#     --experiment_name "$EXP_NAME" \
#     --data_type docci \
#     --seed 42


