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

# ==========================================
# SPARC V2 Improved Configuration
# ==========================================
# Improvements:
# 1. Adaptive Thresholding (Idea B): Uses L=2.2 (hardcoded default in utils) based on ratio stats
# 2. Semantic-Aware Momentum (Idea C): Decay drops when attention shifts
# 3. Bounded ReLU Activation: Best of both worlds (Hard 0, Soft 1)

DEVICE=0
ALPHA=1.1
BETA=0.1
# Tau is now dynamic (\mu + 2.2\sigma), so this fixed value is ignored by V2 logic
TAU=1.5 
LAYER=20
SAVE_FOLDER="./results"

# Max base decay for Semantic Momentum (dropped automatically when shift occurs)
DECAY=0.6 

GATE_TYPE="bounded_relu"
SHARPNESS=1.5
SEED=42

# ==========================================
# 3. COCO (Focus on this first for debugging)
# ==========================================
EXP_NAME="COCO_improved_v2"
DATASET_TYPE="coco"
IMAGE_FOLDER="./dataset/coco/val2014"

# L-factor for Adaptive Thresholding
# Med: ~2.2. Higher = Stricter (Shorter). Lower = More permissive (Longer).
L_FACTOR=2

echo "Running COCO Captioning (Improved V2)..."
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
    --use_v2 \
    --decay "$DECAY" \
    --l_factor "$L_FACTOR" \
    --gate_type "$GATE_TYPE" \
    --sharpness "$SHARPNESS"

echo "Running COCO CHAIR Evaluation (Improved V2)..."
ANNOTATION_FOLDER="./dataset/coco/annotations"
python chair.py \
    --cap_file "$SAVE_FOLDER/${EXP_NAME}.jsonl" \
    --image_id_key image_id \
    --caption_key caption \
    --coco_path "$ANNOTATION_FOLDER" \
    --save_path "$SAVE_FOLDER/${EXP_NAME}_chair.jsonl" \
    > "$SAVE_FOLDER/${EXP_NAME}.log"

echo "All improved V2 tests finished!"
