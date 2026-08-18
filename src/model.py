"""
Model definition: LandmarkProjector feeding into 4-bit NF4 quantized T5-Large
with LoRA adapters on the Q and V attention projections.

The projector maps each frame's 345-dim landmark vector into T5's 1024-dim
encoder embedding space, so we can call T5's encoder/decoder using
`inputs_embeds` instead of token ids - effectively treating the landmark
sequence as a "soft prompt" the same length as the video.
"""

import math

import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional embedding, added to frame embeddings
    so the model knows frame ordering (T5 itself has no positional bias on
    inputs_embeds the way it does on token ids via relative position bias,
    but explicit positions still help the projector's output be ordering-aware
    before it reaches T5's relative-position attention)."""

    def __init__(self, dim, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, dim)

    def forward(self, x):
        # x: (B, T, dim)
        return x + self.pe[:, : x.size(1), :]


class LandmarkProjector(nn.Module):
    """345-dim landmarks -> 1024-dim T5 embedding space."""

    def __init__(self, input_dim=345, hidden_dim=512, output_dim=1024, max_frames=150, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.ln2 = nn.LayerNorm(output_dim)
        self.pos_embed = PositionalEncoding(output_dim, max_len=max_frames)

    def forward(self, landmarks):
        # landmarks: (B, T, 345)
        x = self.fc1(landmarks)
        x = self.ln1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.pos_embed(x)
        return x  # (B, T, 1024)


def build_t5_4bit_lora(model_name="t5-large", lora_r=16, lora_alpha=32, lora_dropout=0.05):
    """Loads T5-Large in 4-bit NF4 and wraps Q/V attention projections with LoRA.

    IMPORTANT — enable_input_require_grads() ordering:
    With 4-bit NF4 quantization (bitsandbytes), the base model's weights are
    frozen/quantized and won't naturally propagate gradients back through
    inputs_embeds. We must call enable_input_require_grads() on the RAW base
    model BEFORE get_peft_model() wraps it. Calling it after (on the PeftModel)
    skips the hook registration on the inner base model, causing the projector's
    output tensor to lose its grad_fn at the T5 encoder boundary — silently
    zeroing projector gradients under gradient checkpointing.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,  # bfloat16 prevents float16 overflow
    )

    model = T5ForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # ← Must come BEFORE get_peft_model() so the hook is on the base model.
    # This registers a forward hook that forces input tensors to require grad,
    # which is necessary for inputs_embeds (our projector output) to carry
    # gradients through the quantized T5 encoder layers.
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q", "v"],  # T5 attention proj naming
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )

    model = get_peft_model(model, lora_config)
    return model


class ASLTranslationModel(nn.Module):
    """Wraps LandmarkProjector + 4-bit/LoRA T5 into a single trainable module."""

    def __init__(self, t5_model_name="t5-large", landmark_dim=345, projector_hidden=512,
                 t5_hidden=1024, max_frames=150, lora_r=16, lora_alpha=32):
        super().__init__()
        self.projector = LandmarkProjector(
            input_dim=landmark_dim,
            hidden_dim=projector_hidden,
            output_dim=t5_hidden,
            max_frames=max_frames,
        )
        self.t5 = build_t5_4bit_lora(t5_model_name, lora_r=lora_r, lora_alpha=lora_alpha)
        self.tokenizer = T5Tokenizer.from_pretrained(t5_model_name)

    def forward(self, landmarks, attention_mask, labels=None):
        inputs_embeds = self.projector(landmarks)  # (B, T, 1024)

        # Explicitly cast to bfloat16 (our configured bnb_4bit_compute_dtype).
        inputs_embeds = inputs_embeds.to(dtype=torch.bfloat16)

        # Sanity guard — catch NaN/inf before they enter T5 so the error is
        # actionable rather than a silent NaN loss.
        if torch.isnan(inputs_embeds).any() or torch.isinf(inputs_embeds).any():
            raise RuntimeError(
                "inputs_embeds contains NaN or Inf after projector. "
                "Check landmark data for corrupted .npy files or extreme values."
            )

        # T5 expects attention_mask as a LongTensor (0/1) for its
        # get_extended_attention_mask() to compute the additive mask correctly.
        attention_mask_long = attention_mask.long()

        outputs = self.t5(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask_long,
            labels=labels,
        )
        return outputs

    def generate(self, landmarks, attention_mask, max_new_tokens=32, num_beams=4):
        inputs_embeds = self.projector(landmarks)
        inputs_embeds = inputs_embeds.to(dtype=torch.bfloat16)
        generated_ids = self.t5.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask.long(),
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return texts

    def trainable_parameter_groups(self, lr_projector, lr_lora):
        """Returns param groups for the optimizer with separate LRs, matching
        the phase configs in config.yaml (projector and LoRA train at
        different rates, especially important in Phase 3)."""
        lora_params = [p for n, p in self.t5.named_parameters() if p.requires_grad]
        projector_params = list(self.projector.parameters())
        return [
            {"params": projector_params, "lr": lr_projector},
            {"params": lora_params, "lr": lr_lora},
        ]
