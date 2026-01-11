# MML Term Project: Reproduction and Improvement of SPARC

**Title:** Reproduction and Improvement of SPARC for Detailed Image Captioning  
**Course:** Multimodal Machine Learning, Peking University  
**Authors:** Boxin Zhang, Ruijie Zhao

## Abstract

This project focuses on improving **SPARC** (Selective Progressive Attention ReCalibration), a training-free method proposed to mitigate hallucinations in Multimodal Large Language Models (MLLMs) by reinforcing visual attention during decoding. We implemented the vanilla SPARC pipeline on the LLaVA-1.5 architecture and proposed an **Adaptive SPARC** extension with two phases of improvements:
1.  **Phase I:** Time-Decayed Momentum & Bounded Activation.
2.  **Phase II:** Adaptive Dynamics & Semantic Gating.

## Repository Structure

- `analyze_stats.py`: Statistical analysis scripts.
- `attn_util_*.py`: Various implementations of the attention utility (Baseline, Improved V1, Improved V2).
- `dataset/`: Directory for datasets (COCO, DOCCI, IIW400).
- `LLaVA/`: The LLaVA-1.5 codebase.
- `scripts/`: Helper scripts for evaluation and data processing.
- `sparc_plots/`: Generated plots and text outputs.
- `run_all_tests_*.sh`: Main execution scripts for different experiments.

## Getting Started

### Prerequisites

1.  **Environment Setup**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Download Data & Models**:
    ```bash
    bash scripts/download_images.sh
    bash scripts/download_models.sh
    ```

### Usage

Before running evaluations, ensure `OPENAI_API_KEY` and `BASE_URL` are set for CLAIR metric evaluation using LLMs.

Run the reproduction and improvement experiments:

1.  **Run Baseline & Vanilla SPARC**:
    ```bash
    bash run_all_tests.sh
    ```

2.  **Run Improved Method (Phase I & II)**:
    ```bash
    bash run_all_tests_improved.sh
    bash run_all_tests_v2.sh
    ```

## Methodology Highlights

-   **Bounded ReLU Activation**: Replaces hard thresholds to prevent noise leakage and signal explosion.
-   **Momentum Map**: Uses a continuous history of visual attention rather than discrete selection counts.
-   **Dynamic Thresholding**: Adapts sensitivity based on the global statistics ($\mu, \sigma$) of the attention map at each step.
-   **Semantic Gating**: Modulates momentum decay using cosine similarity between current and historical attention.

## Results

Our Adaptive SPARC method achieves a better balance between precision (reduced hallucinations) and recall (comprehensiveness) compared to the original implementation on the COCO and DOCCI datasets.

---
*For detailed experimental results and analysis, please refer to the project report.*
