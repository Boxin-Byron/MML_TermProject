
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

# Based on statistical analysis
DEFAULT_L_FACTOR = 2.5 # Median value from analysis was ~2.2171

class AdaptiveMomentumBuffer:
    """
    SPARC Improved V2:
    1. Adaptive Thresholding (Tau = Mean + L * Std)
    2. Semantic-Aware Momentum (Decay drops when attention shifts)
    3. Bounded ReLU Activation
    """
    def __init__(self, decay=0.9, l_factor=DEFAULT_L_FACTOR, 
                 gate_type: Literal['sigmoid', 'relu', 'bounded_relu'] = 'bounded_relu', 
                 sharpness=1.0):
        self.momentum_map = None 
        self.input_len = 0
        self.num_image_patches = None
        
        self.decay_base = decay
        self.l_factor = l_factor
        self.gate_type = gate_type
        self.sharpness = sharpness
        
        self.cached_scale = None
        self.prev_activation_norm = None # To detect semantic shift

    def update_momentum(self, ratio, image_token_index, image_attention=None):
        """
        Calculates adaptive activation and updates momentum.
        Args:
            ratio: (Tensor) The attention ratio (z-score like) [bsz, num_patches]
                   Ideally calculated as (current - avg) / (avg + eps)
            image_attention: (Tensor, optional) Raw attention [bsz, num_patches].
                             Used for adaptive thresholding statistics if ratio is just raw difference.
        """
        # --- 1. Adaptive Thresholding ---
        # NOTE: We now assume `ratio` passed in is already a normalized score or we apply normalization here.
        # But wait, Adaptive Thresholding requires us to know the statistics of the CURRENT ATTENTION distribution.
        # The user wants: \tau_dynamic = \mu + L * \sigma
        # And we want to select Ratio > \tau_dynamic ? NO.
        # We want to select Attention > \tau_dynamic.
        
        # Correction based on user feedback and logical consistency:
        # We need to operate on the same distribution we analyzed.
        # If we analyzed Ratios: L=2.2 means we want Ratio > Mean_Ratio + 2.2 * Std_Ratio.
        
        # Calculate statistics of the input distribution (Ratio)
        mu = ratio.mean(dim=-1, keepdim=True) # [bsz, 1]
        sigma = ratio.std(dim=-1, keepdim=True) + 1e-6 # [bsz, 1]
        
        # Determine dynamic Tau for Ratios
        # tau_dynamic = mu + L * sigma
        tau_dynamic = mu + self.l_factor * sigma
        
        # Calculate 'x' (distance from threshold)
        x = ratio - tau_dynamic

        # --- 2. Activation ---
        if self.gate_type == 'sigmoid':
            # Soft but leaky
            activation = torch.sigmoid(x * self.sharpness)
        elif self.gate_type == 'relu':
            # Hard but unbounded
            activation = F.relu(x)
        elif self.gate_type == 'bounded_relu':
            # Hard 0, Soft transition to 1
            # Normalizing X by sigma might make sharpness more consistent across images?
            # Let's try: x_norm = x / sigma
            x_norm = x / (sigma + 1e-6)
            activation = F.relu(torch.tanh(x_norm * self.sharpness))
        else:
            raise ValueError(f"Unknown gate_type: {self.gate_type}")

        # --- 3. Semantic-Aware Momentum Decay ---
        # Detect if attention focus has shifted significantly
        current_decay = self.decay_base
        
        if self.momentum_map is not None:
            if activation.dim() == 1:
                act_flat = activation.view(1, -1)
                mom_flat = self.momentum_map.view(1, -1)
            else:
                act_flat = activation.view(activation.size(0), -1)
                mom_flat = self.momentum_map.view(self.momentum_map.size(0), -1)
            
            # Optimization & Stability: 
            # If activation is sparse (near zero), skip cosine sim implies orthogonality -> decay=0 -> WIPE MEMORY.
            # Instead, if signal is weak, we should preserve memory (decay=base).
            
            act_norm = act_flat.norm(dim=-1, keepdim=True)
            mom_norm = mom_flat.norm(dim=-1, keepdim=True)
            
            # Valid comparison requires both vectors to have magnitude
            valid_mask = (act_norm > 1e-6) & (mom_norm > 1e-6)
            
            # Default to base decay
            current_decay_tensor = torch.full((act_flat.size(0), 1), self.decay_base, 
                                            device=activation.device, dtype=activation.dtype)
            
            if valid_mask.any():
                # Only compute sim where valid
                cos_sim = F.cosine_similarity(act_flat, mom_flat, dim=-1).unsqueeze(-1)
                adaptive_decay = self.decay_base * (cos_sim ** 2)
                current_decay_tensor = torch.where(valid_mask, adaptive_decay, current_decay_tensor)
            
            current_decay = current_decay_tensor

        # --- Update Momentum ---
        if self.momentum_map is None:
             self.momentum_map = torch.zeros_like(activation)
        
        if self.momentum_map.shape != activation.shape:
             self.momentum_map = torch.zeros_like(activation)

        # M_t = decay * M_{t-1} + (1 - decay) * activation
        # Note: decay is now dynamic per batch item
        
        # Python float decay
        if isinstance(current_decay, float):
            alpha_mix = 1.0 - current_decay
            self.momentum_map.mul_(current_decay).add_(activation, alpha=alpha_mix)
        else:
            # Tensor decay
            alpha_mix = 1.0 - current_decay
            self.momentum_map = self.momentum_map * current_decay + activation * alpha_mix
        
        self.cached_scale = None

    def update_input_len(self, length):
        self.input_len = length

    def reset(self):
        self.momentum_map = None
        self.input_len = 0
        self.num_image_patches = None
        self.cached_scale = None
        self.prev_activation_norm = None

    def calibrate(self, value, alpha, image_token_index, layer_idx=None):
        if self.momentum_map is None:
            return

        if self.num_image_patches is None:
             return

        # Optimization: use cached scale
        if self.cached_scale is None:
            # momentum_map is usually [1, num_patches] or [num_patches].
            # We want [bsz, 1, num_patches, 1]
            
            num_patches = self.momentum_map.shape[-1]
            momentum_factor = self.momentum_map.reshape(-1, 1, num_patches, 1)
            
            # Use (alpha - 1.0) directly
            self.cached_scale = 1.0 + momentum_factor * (alpha - 1.0)
        
        scale = self.cached_scale

        start = image_token_index
        end = start + self.num_image_patches
        
        current_len = value.shape[2]
        valid_end = min(end, current_len)
        valid_start = start
        
        if valid_start < valid_end:
            patch_len = valid_end - valid_start
            scale_slice = scale[:, :, :patch_len, :]
            value[:, :, valid_start:valid_end, :].mul_(scale_slice.to(value.dtype))

    def update_patch_num(self, num_image_patches):
        self.num_image_patches = num_image_patches


def forward_improved_v2(
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
    tau: Optional[float] = 2, # Kept for interface compatibility, but ignored in V2
    selected: Optional[bool] = False,
    se_layers: Optional[Tuple[int, int]] = None,
    indices_buffer: Optional[AdaptiveMomentumBuffer] = None, 
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    # Standard LLaMA Logic ...
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
            raise ValueError("layer_idx required")
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    # --- SPARC V2 LOGIC START ---
    if self.layer_idx >= se_layers[0] and self.layer_idx <= se_layers[1]:
         if indices_buffer is not None and len(past_key_value.value_cache) > self.layer_idx:
             indices_buffer.calibrate(past_key_value.value_cache[self.layer_idx], alpha, image_token_index, self.layer_idx)
    # --- SPARC V2 LOGIC END ---

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
    
    if attention_mask is not None:
         attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

    if gen_new_token == False and self.layer_idx == 0:
        if indices_buffer is not None:
            indices_buffer.update_patch_num(attn_weights.shape[-1] - indices_buffer.input_len)

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

                if indices_buffer is not None:
                    # V2: Update momentum with Adaptive Thresholding logic
                    # Pass ratio, buffer handles unique thresholding statistics
                    indices_buffer.update_momentum(ratio, image_token_index)

        if not gen_new_token:
            self.image_attention = image_attention
        else:
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


def add_custom_attention_layers_improved_v2(
    model,
    lm_model="llama",
    alpha=1,
    beta=0,
    tau=2,
    selected_layer=20,
    se_layers=(0, 31),
    indices_buffer=None,
):
    for i, layer in enumerate(model.model.layers):
        selected = True if selected_layer == i else False
        forward_ = partial(
            forward_improved_v2,
            alpha=alpha,
            beta=beta,
            tau=tau,
            selected=selected,
            se_layers=se_layers,
            indices_buffer=indices_buffer,
        )
        layer.self_attn.forward = MethodType(forward_, layer.self_attn)
