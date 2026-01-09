#!/bin/bash
set -e

# Set HF Mirror for offline/restricted environment
export HF_ENDPOINT=https://hf-mirror.com

# Load environment variables (MODEL_PATH_V1_5, OPENAI_API_KEY, etc.)
if [ -f .env ]; then
    source .env
else
    echo "Warning: .env file not found. Ensure MODEL_PATH_V1_5 and OPENAI_API_KEY are set."
fi

# Common Parameters
DEVICE=0
ALPHA=1.1
BETA=0.1
TAU=1.5
LAYER=20
SAVE_FOLDER="./results"
DECAY=0.6
GATE_TYPE="bounded_relu"
SHARPNESS=1.5
SEED=42

# # ==========================================
# # 1. IIW-400
# # ==========================================
# EXP_NAME="IIW-400_improved"
# DATASET_TYPE="iiw"
# IMAGE_FOLDER="./dataset/docci/images_aar"
# ANNOTATION_FILE="./dataset/iiw400/data.jsonl"
# SEED=42

# echo "Running IIW-400 Captioning (Improved)..."
# CUDA_VISIBLE_DEVICES=$DEVICE python -m eval \
#     --experiment_name "$EXP_NAME" \
#     --model-path "$MODEL_PATH_V1_5" \
#     --dataset_type "$DATASET_TYPE" \
#     --alpha "$ALPHA" \
#     --beta "$BETA" \
#     --image-folder "$IMAGE_FOLDER" \
#     --annotation-file "$ANNOTATION_FILE" \
#     --save_path "$SAVE_FOLDER" \
#     --start_layer 0 \
#     --end_layer 31 \
#     --selected_layer "$LAYER" \
#     --tau "$TAU" \
#     --seed "$SEED" \
#     --use_improved \
#     --decay "$DECAY" \
#     --gate_type "$GATE_TYPE" \
#     --sharpness "$SHARPNESS"

# echo "Running IIW-400 CLAIR Evaluation (Improved)..."
# # Evaluation
# # Note: CLAIR evaluation script usually expects just the experiment name if answers are in SAVE_FOLDER
# python -m clair \
#     --annotation_file "$ANNOTATION_FILE" \
#     --answer_folder "$SAVE_FOLDER" \
#     --save_folder "$SAVE_FOLDER" \
#     --openai_api_key "$OPENAI_API_KEY" \
#     --experiment_name "$EXP_NAME" \
#     --data_type iiw


# # ==========================================
# # 2. DOCCI
# # ==========================================
# EXP_NAME="DOCCI_improved"
# DATASET_TYPE="docci"
# IMAGE_FOLDER="./dataset/docci/images"
# ANNOTATION_FILE="./dataset/docci/docci_descriptions.jsonlines"

# echo "Running DOCCI Captioning (Improved)..."
# CUDA_VISIBLE_DEVICES=$DEVICE python -m eval \
#     --experiment_name "$EXP_NAME" \
#     --model-path "$MODEL_PATH_V1_5" \
#     --dataset_type "$DATASET_TYPE" \
#     --alpha "$ALPHA" \
#     --beta "$BETA" \
#     --image-folder "$IMAGE_FOLDER" \
#     --annotation-file "$ANNOTATION_FILE" \
#     --save_path "$SAVE_FOLDER" \
#     --start_layer 0 \
#     --end_layer 31 \
#     --selected_layer "$LAYER" \
#     --tau "$TAU" \
#     --seed "$SEED" \
#     --use_improved \
#     --decay "$DECAY" \
#     --gate_type "$GATE_TYPE" \
#     --sharpness "$SHARPNESS"

# echo "Running DOCCI CLAIR Evaluation (Improved)..."
# python -m clair \
#     --annotation_file "$ANNOTATION_FILE" \
#     --answer_folder "$SAVE_FOLDER" \
#     --save_folder "$SAVE_FOLDER" \
#     --openai_api_key "$OPENAI_API_KEY" \
#     --experiment_name "$EXP_NAME" \
#     --data_type docci \
#     --seed "$SEED"

# ==========================================
# 3. COCO
# ==========================================
EXP_NAME="COCO_improved"
DATASET_TYPE="coco"
IMAGE_FOLDER="./dataset/coco/val2014"
# Using defaults for other params or relying on script logic that coco implies certain paths if needed, 
# but here we follow captioning_coco.sh which doesn't pass annotation file.

echo "Running COCO Captioning (Improved)..."
CUDA_VISIBLE_DEVICES=$DEVICE python -m eval \
    --experiment_name "$EXP_NAME" \
    --model-path "$MODEL_PATH_V1_5" \
    --dataset_type "$DATASET_TYPE" \
    --alpha "$ALPHA" \
    --beta "$BETA" \
    --image-folder "$IMAGE_FOLDER" \
    --save_path "$SAVE_FOLDER" \
    --start_layer 0 \
    --end_layer 31 \
    --selected_layer "$LAYER" \
    --tau "$TAU" \
    --seed "$SEED" \
    --use_improved \
    --decay "$DECAY" \
    --gate_type "$GATE_TYPE" \
    --sharpness "$SHARPNESS"

echo "Running COCO CHAIR Evaluation (Improved)..."
ANNOTATION_FOLDER="./dataset/coco/annotations"
# NOTE: Ensure ./dataset/coco/annotations exists and contains captions_val2014.json or similar expected by chair.py
python chair.py \
    --cap_file "$SAVE_FOLDER/${EXP_NAME}.jsonl" \
    --image_id_key image_id \
    --caption_key caption \
    --coco_path "$ANNOTATION_FOLDER" \
    --save_path "$SAVE_FOLDER/${EXP_NAME}_chair.jsonl" \
    > "$SAVE_FOLDER/${EXP_NAME}.log"

echo "All improved tests finished!"
