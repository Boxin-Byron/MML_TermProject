import torch
from types import MethodType
from functools import partial
from typing import Dict, Any, Optional, Tuple, Literal
import torch.nn.functional as F
import torch.nn as nn
import logging
from dataclasses import dataclass
import math
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    rotate_half,
    repeat_kv,
)
from transformers.cache_utils import Cache

logger = logging.getLogger(__name__)

class SoftMomentumBuffer:
    """
    Improved Buffer for SPARC implementing Soft-Gated Activation and Time-Decayed Momentum.
    """
    def __init__(self, decay=0.9, gate_type: Literal['sigmoid', 'relu'] = 'sigmoid', 
                 sharpness=1.0):
        self.momentum_map = None  # Stores the momentum of activations for each image patch
        self.input_len = 0
        self.num_image_patches = None
        
        # Hyperparameters for improvements
        self.decay = decay  # Decay factor for time-decayed momentum (0 < decay < 1)
        self.gate_type = gate_type # Type of soft gating
        self.sharpness = sharpness # Controls steepness of sigmoid

    def update_momentum(self, ratio, tau, image_token_index):
        """
        Calculates soft activation from attention ratio and updates the momentum map.
        
        Args:
            ratio: (Tensor) (image_attention - avg) / avg
            tau: (float) Threshold used in original binary selection (now used as soft bias)
        """
        # 1. Soft-Gated Activation
        # We shift the ratio by tau so that 'tau' is the center/start of activation
        # Logic: If ratio > tau, we want positive activation.
        
        # Input to gate
        x = ratio - tau
        
        if self.gate_type == 'sigmoid':
            # Sigmoid gating: maps (-inf, inf) -> (0, 1)
            # We apply sharpness factor. 
            # If ratio = tau, activation is 0.5.
            activation = torch.sigmoid(x * self.sharpness)
        elif self.gate_type == 'relu':
            # ReLU gating: maps (-inf, inf) -> [0, inf)
            # Preserves magnitude differences better for very strong signals
            activation = F.relu(x)
        else:
            raise ValueError(f"Unknown gate_type: {self.gate_type}")

        # 2. Time-Decayed Momentum
        # Update momentum: M_t = decay * M_{t-1} + activation
        # We need to initialize momentum_map if it's None.
        if self.momentum_map is None:
             self.momentum_map = torch.zeros_like(activation)
        
        # Ensure shapes match (handle potential batch size differences if any, usually 1)
        if self.momentum_map.shape != activation.shape:
             self.momentum_map = torch.zeros_like(activation)

        self.momentum_map = self.decay * self.momentum_map + activation
        
        # Detach to prevent gradient explosion over long sequences if not needed
        # Assuming inference-time optimization or no backprop through time here
        # self.momentum_map = self.momentum_map.detach() 

    def update_input_len(self, length):
        self.input_len = length

    def reset(self):
        self.momentum_map = None
        self.input_len = 0
        self.num_image_patches = None

    def calibrate(self, value, alpha, image_token_index, layer_idx=None):
        """
        Applies calibration to token representations using the accumulated momentum map.
        
        Args:
            value: (Tensor) The Value cache tensor [bsz, num_heads, seq_len, head_dim]
            alpha: (float) The base scaling factor.
            image_token_index: (int) Start index of image tokens.
        """
        if self.momentum_map is None:
            return

        # We need to apply the boost to the image tokens in the value cache.
        # Image tokens are located at: image_token_index : image_token_index + num_patches
        
        # Ensure we have image patches count
        if self.num_image_patches is None:
             return

        # Slice the value tensor to get image tokens
        # shape: [bsz, num_heads, seq_len, head_dim]
        # We want to modify value[:, :, start:end, :]
        start = image_token_index
        end = start + self.num_image_patches
        
        if value.shape[2] < end:
            # Current sequence might calculate attention for tokens *before* full image is processed? 
            # Or this might happen during prefill if we chunk.
            # Usually LLaVA prefill has full image.
            # During generation, seq_len grows. image tokens are static in past.
            pass

        # Prepare modulation factor
        # alpha is the target multiplier for "selected" tokens.
        # New multiplier = 1 + momentum * (alpha - 1)
        # If momentum is approx 1 (sigmoid saturated), we get alpha.
        # If momentum is higher (accumulated), we get > alpha.
        
        # Map shape: [bsz, num_patches]. Broadcast to [bsz, num_heads, num_patches, head_dim]
        # momentum_map is [bsz, patches] or [patches]
        if self.momentum_map.dim() == 1:
             # Assume single batch if 1D
             momentum_factor = self.momentum_map.unsqueeze(0).unsqueeze(1).unsqueeze(-1) # [1, 1, patches, 1]
        elif self.momentum_map.dim() == 2:
            momentum_factor = self.momentum_map.unsqueeze(1).unsqueeze(-1) # [bsz, 1, patches, 1]
        else:
             # Unexpected shape
             return
        
        # Calculate scale
        # We use (alpha - 1) as the gain factor.
        scale = 1.0 + momentum_factor * (alpha - 1.0)
        
        # Apply scale
        # value is [bsz, num_kv_heads, seq_len, head_dim]
        # We need to handle num_kv_heads vs num_heads used in momentum calculation?
        # Momentum is calculated from attention weights which are usually averaged or per-head?
        # In original code: indices = (ratio >= tau).nonzero()
        # image_attention was averaged over heads: .mean(dim=1)
        # So momentum is per-patch (averaged over heads).
        
        # Safe slice update
        current_len = value.shape[2]
        valid_end = min(end, current_len)
        valid_start = start
        
        if valid_start < valid_end:
            patch_len = valid_end - valid_start
            # Slice momentum to match (in case of weird partial updates)
            # Scale factor should only match the patches we are actually scaling
            # Scale shape: [bsz, 1, patches, 1] 
            # We need to slice usage of scale too if we are only scaling a subset
             
            # scale slice needs to have 4 dimensions.
            scale_slice = scale[:, :, :patch_len, :]
            
            value[:, :, valid_start:valid_end, :] *= scale_slice.to(value.dtype)

    def update_patch_num(self, num_image_patches):
        self.num_image_patches = num_image_patches


# Improved attention forward pass
def forward_improved(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    image_token_index: Optional[int] = 35,
    alpha: Optional[float] = 1.0,
    beta: Optional[float] = 0.0,
    tau: Optional[float] = 2,
    selected: Optional[bool] = False,
    se_layers: Optional[Tuple[int, int]] = None,
    indices_buffer: Optional[SoftMomentumBuffer] = None, # Changed type hint
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    # Standard LLaMA Attention Projection Logic
    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0)
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)
    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    # --- SPARC IMPROVED LOGIC START ---
    # Apply Calibration using Momentum Buffer from previous steps
    # Note: We calibrate the cached values *before* current step update if possible, 
    # but efficient cache implementation might require calibrating the stored cache.
    # The original code calibrates 'past_key_value.value_cache[self.layer_idx]'.
    
    if self.layer_idx >= se_layers[0] and self.layer_idx <= se_layers[1]:
         # In improved version, we trust the buffer's momentum state
         if indices_buffer is not None and len(past_key_value.value_cache) > self.layer_idx:
             indices_buffer.calibrate(past_key_value.value_cache[self.layer_idx], alpha, image_token_index, self.layer_idx)
    # --- SPARC IMPROVED LOGIC END ---

    if len(past_key_value.key_cache) > self.layer_idx:
        gen_new_token = True
    else:
        gen_new_token = False

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    
    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is {attn_weights.size()}")

    if attention_mask is not None:
         attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

    # Calculate Image Attention stats for SPARC
    if gen_new_token == False and self.layer_idx == 0:
        if indices_buffer is not None:
            indices_buffer.update_patch_num(attn_weights.shape[-1] - indices_buffer.input_len)

    # Get attention on image tokens
    # Taking the mean across heads
    num_im_patches = indices_buffer.num_image_patches if indices_buffer.num_image_patches else 0
    if num_im_patches > 0:
        image_attention = attn_weights[
            :,
            :,
            -1,
            image_token_index : image_token_index + num_im_patches,
        ].mean(dim=1) # [bsz, num_patches]

        if gen_new_token:
            if selected:
                # Calculating Ratio
                # ratio = (current - avg) / avg
                ratio = (image_attention - self.image_attention) / (self.image_attention + 1e-6)
                ratio = ratio.squeeze(dim=0) # [num_patches] or [bsz, num_patches] usually bsz=1
                
                # --- SPARC IMPROVED UPDATE ---
                # Update Soft Momentum Buffer
                if indices_buffer is not None:
                    indices_buffer.update_momentum(ratio, tau, image_token_index)
                # -----------------------------

        if not gen_new_token:
            self.image_attention = image_attention
        else:
            # Baseline EMA update
            self.image_attention = (1 - beta) * image_attention + beta * self.image_attention

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def add_custom_attention_layers_improved(
    model,
    lm_model="llama",
    alpha=1,
    beta=0,
    tau=2,
    selected_layer=20,
    se_layers=(0, 31),
    indices_buffer=None,
):
    """
    Injects the improved SPARC attention mechanism into the model.
    """
    for i, layer in enumerate(model.model.layers):
        selected = True if selected_layer == i else False
        forward_ = partial(
            forward_improved,
            alpha=alpha,
            beta=beta,
            tau=tau,
            selected=selected,
            se_layers=se_layers,
            indices_buffer=indices_buffer,
        )
        layer.self_attn.forward = MethodType(forward_, layer.self_attn)
