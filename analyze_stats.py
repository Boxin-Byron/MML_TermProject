
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from PIL import Image

# Dummy function to simulate attention ratio distribution
# In a real scenario, this would hook into the model's generation process.
# Since we cannot run the full LLaVA model interactively easily here to get stats, 
# I will create a script that modifies 'attn_util_improved.py' to LOG statistics to a file
# during a small run, and then we analyze that file.

def analyze_ratio_stats(log_file="ratio_stats.jsonl"):
    if not os.path.exists(log_file):
        print(f"Log file {log_file} not found. Please run the data collection first.")
        return

    ratios = []
    means = []
    stds = []
    
    print("Loading stats...")
    with open(log_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            # data['ratios'] is a list of float ratios for one step
            r_step = np.array(data['ratios'])
            ratios.extend(r_step)
            means.append(r_step.mean())
            stds.append(r_step.std())

    ratios = np.array(ratios)
    means = np.array(means)
    stds = np.array(stds)
    
    print(f"Total steps analyzed: {len(means)}")
    print(f"Global Ratio Mean: {ratios.mean():.4f}")
    print(f"Global Ratio Std: {ratios.std():.4f}")
    print(f"Average Per-Step Mean: {means.mean():.4f}")
    print(f"Average Per-Step Std: {stds.mean():.4f}")
    
    # We want to find L such that mu + L * sigma approx 1.5 (since tau=1.5 is good)
    # L = (1.5 - mu) / sigma
    
    # Calculate implicit L for each step assuming target tau=1.5
    # Filter out steps with very low std to avoid division by zero
    valid_indices = stds > 1e-4
    implicit_L = (1.5 - means[valid_indices]) / stds[valid_indices]
    
    suggested_L = np.median(implicit_L)
    print(f"\n--- Analysis Results ---")
    print(f"Target Fixed Tau: 1.5")
    print(f"Calculated Implicit L (Median): {suggested_L:.4f}")
    print(f"Calculated Implicit L (Mean): {implicit_L.mean():.4f}")

    return suggested_L

if __name__ == "__main__":
    # execute
    analyze_ratio_stats()
