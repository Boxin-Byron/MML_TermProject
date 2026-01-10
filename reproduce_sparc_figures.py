
import os
import sys
import argparse
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from scipy.stats import wasserstein_distance
from PIL import Image


# Ensure project root is in path
sys.path.append(os.getcwd())

from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.utils import disable_torch_init
from LLaVA.llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from LLaVA.llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from LLaVA.llava.conversation import conv_templates

# Import actual SPARC implementations
from attn_util import add_custom_attention_layers, SelectedIndexBuffer
from attn_util_improved import add_custom_attention_layers_improved, SoftMomentumBuffer
from attn_util_improved_v2 import add_custom_attention_layers_improved_v2, AdaptiveMomentumBuffer

CAPTURED_ATTENTION = []
from attn_util_improved import add_custom_attention_layers_improved, SoftMomentumBuffer
from attn_util_improved_v2 import add_custom_attention_layers_improved_v2, AdaptiveMomentumBuffer

# Global list to capture attention weights
CAPTURED_ATTENTION = []

def prepare_input(image_path, model_path, model_base, device="cuda"):
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name, load_4bit=True, device=device
    )
    
    # Prepare prompt and image
    qs = "Please describe this image in detail."
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image = Image.open(image_path).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)[0]
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    
    return {
        "model": model, 
        "tokenizer": tokenizer, 
        "input_ids": input_ids.unsqueeze(0).to(device),
        "image_tensor": image_tensor.unsqueeze(0).half().to(device),
        "image_sizes": [image.size],
        "original_image": image
    }

def run_inference_with_hooks(data, args):
    model = data["model"]
    
    # 1. Reset Global Capture
    global CAPTURED_ATTENTION
    CAPTURED_ATTENTION = []
    
    # 2. Select Method and Apply Patching
    if args.method == "naive":
        # Naive = SPARC but alpha=1.0
        buffer = SelectedIndexBuffer()
        add_custom_attention_layers(
            model=model,
            selected_layer=args.selected_layer,
            se_layers=(0, 31),
            indices_buffer=buffer,
            alpha=1.0, # Force 1.0
            beta=0.0
        )
    elif args.method == "sparc":
        buffer = SelectedIndexBuffer()
        add_custom_attention_layers(
            model=model,
            selected_layer=args.selected_layer,
            se_layers=(0, 31),
            indices_buffer=buffer,
            alpha=args.alpha,
            beta=0.0
        )
    elif args.method == "improved_v1":
        buffer = SoftMomentumBuffer(decay=0.9, gate_type="sigmoid") 
        add_custom_attention_layers_improved(
            model=model,
            selected_layer=args.selected_layer,
            se_layers=(0, 31),
            indices_buffer=buffer,
            alpha=args.alpha
        )
    elif args.method == "improved_v2":
        buffer = AdaptiveMomentumBuffer(decay=0.9, l_factor=2.2, gate_type="bounded_relu")
        add_custom_attention_layers_improved_v2(
            model=model,
            selected_layer=args.selected_layer,
            se_layers=(0, 31),
            indices_buffer=buffer,
            alpha=args.alpha
        )
    else:
        # Fallback / Default ?
        print(f"Warning: Unknown method '{args.method}', defaulting to Naive")
        buffer = SelectedIndexBuffer()
        add_custom_attention_layers(
            model=model,
            selected_layer=args.selected_layer,
            se_layers=(0, 31),
            indices_buffer=buffer,
            alpha=1.0,
            beta=0.0
        )

    # 3. Add a wrapper/hook to the selected layer to CAPTURE output attention
    layer_module = model.model.layers[args.selected_layer].self_attn
    
    def hook_fn(module, input, output):
        # output is likely (attn_output, attn_weights, past_key_value)
        if isinstance(output, tuple) and len(output) > 1:
            attn_weights = output[1]
            if attn_weights is not None:
                # attn_weights shape: [bsz, n_heads, q_len, k_len]
                # Average over heads to save space
                saved = attn_weights.detach().float().mean(dim=1).cpu() 
                CAPTURED_ATTENTION.append(saved)
            else:
                 # Some implementations might not return weights if not asked
                 pass

    # Register the hook
    handle = layer_module.register_forward_hook(hook_fn)

    # Initial input length update
    if hasattr(buffer, "input_len"): 
        buffer.input_len = data["input_ids"].shape[1] - 1
    elif hasattr(buffer, "update_input_len"): 
        # Note: some buffers might not have this method, check file first ideally
        # But based on previous reads, SelectedIndexBuffer has assignments.
        pass
    
    try:
        with torch.inference_mode():
            output_ids = model.generate(
                data["input_ids"],
                images=data["image_tensor"],
                image_sizes=data["image_sizes"],
                do_sample=True,
                temperature=args.tau if hasattr(args, 'tau') else 0.2, 
                max_new_tokens=128,
                use_cache=True,
                output_attentions=True, # Critical for hook to work
                return_dict_in_generate=True
            )
            final_sequences = output_ids.sequences
            
    finally:
        handle.remove()
        
    outputs = data["tokenizer"].batch_decode(final_sequences, skip_special_tokens=True)[0].strip()
    print(f"Generated ({args.method}): {outputs}")
    
    return CAPTURED_ATTENTION, outputs

# --- Visualization Functions ---

def plot_attention_diversity(attn_sequence, save_path="figure3_diversity.png"):
    """
    Figure 3: Wasserstein Distance Matrix
    attn_sequence: List of numpy arrays [1, seq_len] or similar.
    We need to extract only IMAGE attention diversity? Or global?
    The user description says "Visual Attention Diversity", so we focus on image tokens.
    """
    # Assuming the image tokens are at specific indices.
    # LLAVA typically puts image tokens early.
    # We will assume indices 35 to 35+576 are image tokens (standard llava-1.5)
    # But let's be dynamic.
    
    # Pre-process: stack and normalize
    # Each step has different length attention (growing text).
    # But image part is fixed size.
    
    T = len(attn_sequence)
    # Extract Image Attention for all steps
    # Note: attn_sequence[t] shape is [1, 1, seq_len] (from our hook)
    
    img_start_idx = 35 # Standard start default
    img_len = 576 # Standard 24x24
    
    # Check if we have enough tokens
    img_attns = []
    
    print(f"[DEBUG] Processing sequence of length {T} for Diversity Plot")
    # print(f"[DEBUG] Processing sequence of length {T} for Diversity Plot")
    for t in range(T):
        attn = attn_sequence[t].squeeze() 
        # Debug print
        # if t < 3: # Print first few
        #     print(f"[DEBUG-Div] Step {t}: Shape={attn.shape}")

        if attn.ndim != 1:
            # Skip if not 1D (e.g. prefill or unexpected shape)
            continue
            
        if attn.shape[0] < img_start_idx + img_len:
            continue
        
        # Extract and Normalize Image Attention
        curr_img_attn = attn[img_start_idx : img_start_idx + img_len].numpy()
        
        # Normalize to sum=1 for Wasserstein
        sum_val = curr_img_attn.sum()
        if sum_val > 0:
            curr_img_attn = curr_img_attn / sum_val
        else:
            curr_img_attn = np.ones_like(curr_img_attn) / img_len
            
        img_attns.append(curr_img_attn)
    
    if len(img_attns) < 2:
        print("Not enough steps for Diversity Plot")
        return

    T_valid = len(img_attns)
    dist_matrix = np.zeros((T_valid, T_valid))
    
    print(f"Calculating Wasserstein Matrix ({T_valid}x{T_valid})...")
    # Using 1D grid for simplicity as stated in pseudo-code
    grid = np.arange(img_len)
    
    for i in range(T_valid):
        for j in range(T_valid):
            dist_matrix[i, j] = wasserstein_distance(
                grid, grid,
                u_weights=img_attns[i], 
                v_weights=img_attns[j]
            )
            
    plt.figure(figsize=(10, 8))
    sns.heatmap(dist_matrix, cmap="viridis", square=True)
    plt.title("Visual Attention Diversity (Wasserstein Distance)")
    plt.xlabel("Token Step")
    plt.ylabel("Token Step")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")

def plot_attention_dynamics(attn_sequence, save_path="figure4_dynamics.png"):
    """
    Figure 4: Attention Dynamics (Heatmap)
    """
    img_start_idx = 35
    img_len = 576
    
    dynamics = []
    
    # print(f"[DEBUG] Processing sequence of length {len(attn_sequence)} for Dynamics Plot")
    for t, attn_raw in enumerate(attn_sequence):
        attn = attn_raw.squeeze()
        
        # if t < 3: # Debug print first few
        #    print(f"[DEBUG-Dyn] Step {t}: Raw Shape={attn_raw.shape}, Squeezed={attn.shape}")

        if attn.ndim != 1:
            continue
            
        if attn.shape[0] < img_start_idx + img_len:
            continue
            
        curr_img_attn = attn[img_start_idx : img_start_idx + img_len].numpy()
        dynamics.append(curr_img_attn)
        
    if len(dynamics) == 0:
        print(f"No valid attention dynamics found for {save_path}")
        return

    try:
        dynamics = np.array(dynamics) # [T, num_img_tokens]
    except Exception as e:
        print(f"[ERROR] Failed to convert dynamics to array: {e}")
        # Debug why
        shapes = [d.shape for d in dynamics]
        # print(f"[DEBUG] Shapes collected: {shapes[:5]} ...")
        return

    if dynamics.shape[0] == 0:
        return

    # Sort tokens by total attention to make the plot cleaner (put active tokens at top)
    # shapes: dynamics [T, 576]
    total_attn_per_token = dynamics.sum(axis=0)
    sorted_indices = np.argsort(total_attn_per_token)[::-1] # Descending
    sorted_dynamics = dynamics[:, sorted_indices]

    plt.figure(figsize=(14, 8))
    
    # Transpose so x-axis is Time, y-axis is Sorting Image Token Index
    # Use log scale or robust quantile for better contrast
    
    vmax = np.percentile(sorted_dynamics, 99.5)
    ax = sns.heatmap(sorted_dynamics.T, cmap="magma", cbar=True, vmin=0, vmax=vmax)
    
    plt.title("Temporal Dynamics of Image Attention (Sorted by Relevance)")
    plt.xlabel("Generation Step")
    plt.ylabel("Image Token Index (Sorted by Total Attention)")
    
    # Reduce Tick density
    plt.yticks(ticks=np.arange(0, 576, 50), labels=np.arange(0, 576, 50))
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")
    
    # Top-K Lines (Keep this, effectively shows the first K rows of our sorted matrix)
    plt.figure(figsize=(12, 6))
    if dynamics.shape[0] > 0:
        # We already sorted them, so just pick first 5
        top_k_indices_original = sorted_indices[:5]
        
        for i, original_idx in enumerate(top_k_indices_original):
            plt.plot(dynamics[:, original_idx], label=f"Token {original_idx} (Rank {i+1})", linewidth=1.5)
            
        plt.legend()
        plt.title("Attention Weight of Top-5 Visual Tokens over Time")
        plt.xlabel("Generation Step")
        plt.ylabel("Attention Weight")
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path.replace(".png", "_lines.png"), dpi=300)
        plt.close()

def plot_img_text_ratio(attn_sequence, save_path="figure5_ratio.png"):
    """
    Figure 5: Image vs Text Attention Ratio
    """
    img_start_idx = 35
    img_len = 576
    
    steps = []
    img_means = []
    text_means = []
    
    for t, attn_tensor in enumerate(attn_sequence):
        attn = attn_tensor.squeeze().numpy()
        
        # Determine ranges
        # [0...34] -> System/User Text (Instruction) + Image Start
        # [35...35+576] -> Image
        # [35+576...] -> Text (Instruction End + Generated)
        
        if len(attn) <= img_start_idx + img_len:
            continue
            
        img_part = attn[img_start_idx : img_start_idx + img_len]
        
        # Text is everything else combined (Instruction + Generation)
        # Note: simplistic split
        text_before = attn[:img_start_idx]
        text_after = attn[img_start_idx + img_len:]
        text_part = np.concatenate([text_before, text_after])
        
        img_means.append(np.mean(img_part))
        text_means.append(np.mean(text_part))
        steps.append(t)
        
    plt.figure(figsize=(8, 6))
    plt.plot(steps, img_means, label="Avg Attn to Image", color='blue')
    plt.plot(steps, text_means, label="Avg Attn to Text", color='orange')
    plt.xlabel("Generation Step")
    plt.ylabel("Average Attention Weight")
    plt.title("Shift of Focus: Image vs. Text")
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")

def visualize_attn_map(attn_sequence, original_image, step_idx, save_path):
    """
    Figure 2: Overlay Attention Map on Image
    """
    if step_idx >= len(attn_sequence):
        print(f"Step {step_idx} out of range for visualization")
        return

    attn = attn_sequence[step_idx].squeeze().numpy()
    img_start_idx = 35
    img_len = 576
    
    if len(attn) < img_start_idx + img_len:
        return
        
    img_attn = attn[img_start_idx : img_start_idx + img_len]
    
    # Reshape to 24x24
    w_feat = int(np.sqrt(img_len))
    h_feat = w_feat
    attn_map = img_attn.reshape(h_feat, w_feat)
    
    # Normalize for visualization
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    attn_map = np.uint8(255 * attn_map)
    
    # Resize to original image size
    img_np = np.array(original_image)
    h, w = img_np.shape[:2]
    
    attn_high_res = cv2.resize(attn_map, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # Apply colormap
    heatmap = cv2.applyColorMap(attn_high_res, cv2.COLORMAP_JET)
    
    # Overlay
    alpha = 0.5
    overlay = cv2.addWeighted(img_np, 1 - alpha, heatmap, alpha, 0)
    
    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"Saved {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--image-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--selected_layer", type=int, default=20)
    
    # New Arguments for SPARC variants
    parser.add_argument("--method", type=str, default="naive", choices=["naive", "sparc", "improved_v1", "improved_v2"])
    parser.add_argument("--alpha", type=float, default=1.0, help="Alpha scale factor")
    parser.add_argument("--tau", type=float, default=0.2, help="Temperature")
    parser.add_argument("--save-dir", type=str, default="sparc_plots", help="Directory to save plots")
    
    args = parser.parse_args()

    # 1. Prepare
    data = prepare_input(args.image_path, args.model_path, args.model_base)
    
    # 2. Run Inference
    print(f"Running Inference with Method: {args.method}, Alpha: {args.alpha}")
    attn_history, generated_text = run_inference_with_hooks(data, args)
    
    print(f"Collected {len(attn_history)} steps of attention history.")
    
    # 3. Generate Plots
    # Create a specific subdirectory for this experiment settings
    # e.g. sparc_plots/sparc_alpha1.1
    sub_dir = f"{args.save_dir}/{args.method}_alpha{args.alpha}"
    os.makedirs(sub_dir, exist_ok=True)
    
    # Save text as well
    with open(f"{sub_dir}/generated_text.txt", "w") as f:
        f.write(generated_text)
    
    plot_attention_diversity(attn_history, f"{sub_dir}/fig3_diversity.png")
    plot_attention_dynamics(attn_history, f"{sub_dir}/fig4_dynamics.png")
    plot_img_text_ratio(attn_history, f"{sub_dir}/fig5_ratio.png")
    
    # Figure 2 Visualization (Early vs Late)
    if len(attn_history) > 10:
        visualize_attn_map(attn_history, data["original_image"], 10, f"{sub_dir}/fig2_early_step10.png")
    if len(attn_history) > 0:
        visualize_attn_map(attn_history, data["original_image"], len(attn_history)-1, f"{sub_dir}/fig2_late_laststep.png")
        
    print(f"All results saved to {sub_dir}")

if __name__ == "__main__":
    main()
