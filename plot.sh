#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com

# Configuration
MODEL_PATH="/home/boxin/llava-v1.5-7b"
# Using the image presumably used in previous tests
IMAGE_PATH="dataset/docci/images_aar/aar_test_04600.jpg"

# 1. Naive (SPARC with Alpha=1.0)
echo "Running Naive Experiment..."
python reproduce_sparc_figures.py \
    --model-path $MODEL_PATH \
    --image-path $IMAGE_PATH \
    --method naive \
    --alpha 1.0 \
    --save-dir sparc_plots

# 2. SPARC (Original)
echo "Running SPARC Experiment..."
python reproduce_sparc_figures.py \
    --model-path $MODEL_PATH \
    --image-path $IMAGE_PATH \
    --method sparc \
    --alpha 1.1 \
    --save-dir sparc_plots

# 3. SPARC Improved V1
echo "Running Improved V1 Experiment..."
python reproduce_sparc_figures.py \
    --model-path $MODEL_PATH \
    --image-path $IMAGE_PATH \
    --method improved_v1 \
    --alpha 1.1 \
    --save-dir sparc_plots

# 4. SPARC Improved V2
echo "Running Improved V2 Experiment..."
python reproduce_sparc_figures.py \
    --model-path $MODEL_PATH \
    --image-path $IMAGE_PATH \
    --method improved_v2 \
    --alpha 1.1 \
    --save-dir sparc_plots

echo "All plotting experiments completed."