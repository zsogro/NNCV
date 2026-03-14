import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import re
from torchvision.models import vit_b_16

from head import LinearHead


class Model(nn.Module):

    EMBED_DIM = 768
    PATCH_SIZE = 16
    BACKBONE_IMAGE_SIZE = 288
    BACKBONE_WEIGHTS = "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth" 

    def __init__(self, in_channels=3, n_classes=19):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes

        # Build ViT-B/16 backbone architecture, then load local checkpoint weights.
        self.backbone = vit_b_16(weights=None, image_size=self.BACKBONE_IMAGE_SIZE)
        self._load_backbone_weights(self.BACKBONE_WEIGHTS)

        # ViT-B/16 embedding size
        self.embed_dim = self.EMBED_DIM
        self.patch_size = self.PATCH_SIZE

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.eval()

        self.seg_head = LinearHead(
            in_channels=[self.embed_dim],
            n_output_channels=self.n_classes,
            use_batchnorm=False,
            use_cls_token=False,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def _load_backbone_weights(self, weights_path):
        checkpoint_path = Path(weights_path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path(__file__).resolve().parent / checkpoint_path

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Backbone checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
                state_dict = checkpoint["model"]

        if not isinstance(state_dict, dict):
            raise TypeError("Checkpoint does not contain a valid state_dict")

        cleaned_state_dict = {}
        prefixes = (
            "module.",
            "backbone.",
            "teacher.backbone.",
            "student.backbone.",
        )
        for key, value in state_dict.items():
            if not torch.is_tensor(value):
                continue
            clean_key = key
            for prefix in prefixes:
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]
                    break
            cleaned_state_dict[clean_key] = value

        remapped_state_dict = self._remap_dinov3_to_torchvision(cleaned_state_dict)

        expected_state_dict = self.backbone.state_dict()
        compatible_state_dict = {
            key: value
            for key, value in remapped_state_dict.items()
            if key in expected_state_dict and expected_state_dict[key].shape == value.shape
        }

        # DINOv3 uses RoPE, while torchvision ViT uses absolute positional embeddings.
        # Keep positional embeddings zeroed when no compatible checkpoint tensor is available.
        if "encoder.pos_embedding" not in compatible_state_dict:
            compatible_state_dict["encoder.pos_embedding"] = torch.zeros_like(
                expected_state_dict["encoder.pos_embedding"]
            )

        load_info = self.backbone.load_state_dict(compatible_state_dict, strict=False)
        if len(load_info.missing_keys) == len(self.backbone.state_dict()):
            raise RuntimeError(
                "No backbone weights matched vit_b_16. "
                "Verify that the checkpoint matches torchvision VisionTransformer naming."
            )

    def _remap_dinov3_to_torchvision(self, state_dict):
        remapped = {}

        # Stem / tokens
        if "cls_token" in state_dict:
            remapped["class_token"] = state_dict["cls_token"]
        if "patch_embed.proj.weight" in state_dict:
            remapped["conv_proj.weight"] = state_dict["patch_embed.proj.weight"]
        if "patch_embed.proj.bias" in state_dict:
            remapped["conv_proj.bias"] = state_dict["patch_embed.proj.bias"]

        # Transformer blocks
        block_pattern = re.compile(r"^blocks\.(\d+)\.(.+)$")
        for key, value in state_dict.items():
            match = block_pattern.match(key)
            if not match:
                continue

            block_idx, suffix = match.groups()
            prefix = f"encoder.layers.encoder_layer_{block_idx}."

            mapping = {
                "norm1.weight": "ln_1.weight",
                "norm1.bias": "ln_1.bias",
                "attn.qkv.weight": "self_attention.in_proj_weight",
                "attn.qkv.bias": "self_attention.in_proj_bias",
                "attn.proj.weight": "self_attention.out_proj.weight",
                "attn.proj.bias": "self_attention.out_proj.bias",
                "norm2.weight": "ln_2.weight",
                "norm2.bias": "ln_2.bias",
                "mlp.fc1.weight": "mlp.0.weight",
                "mlp.fc1.bias": "mlp.0.bias",
                "mlp.fc2.weight": "mlp.3.weight",
                "mlp.fc2.bias": "mlp.3.bias",
            }

            target_suffix = mapping.get(suffix)
            if target_suffix is not None:
                remapped[prefix + target_suffix] = value

        # Final norm
        if "norm.weight" in state_dict:
            remapped["encoder.ln.weight"] = state_dict["norm.weight"]
        if "norm.bias" in state_dict:
            remapped["encoder.ln.bias"] = state_dict["norm.bias"]

        return remapped

    def _forward_backbone_patch_tokens(self, x):
        # Mirror torchvision VisionTransformer forward path and keep patch tokens.
        tokens = self.backbone._process_input(x)
        batch_size = tokens.shape[0]
        class_token = self.backbone.class_token.expand(batch_size, -1, -1)
        tokens = torch.cat((class_token, tokens), dim=1)
        tokens = self.backbone.encoder(tokens)
        return tokens[:, 1:, :]

    def forward(self, x):

        # Check if the input tensor has the expected number of channels
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, but got {x.shape[1]}")
        B, _, H, W = x.shape

        # Extract patch tokens
        with torch.no_grad():
            patch_tokens = self._forward_backbone_patch_tokens(x)

        # Convert tokens -> feature map
        n_patches = patch_tokens.shape[1]
        patch_height = H // self.patch_size
        patch_width = W // self.patch_size
        if patch_height * patch_width != n_patches:
            raise ValueError(
                f"Expected {patch_height * patch_width} patch tokens for input size {(H, W)}, got {n_patches}"
            )

        feat_map = patch_tokens.permute(0, 2, 1).reshape(
            B, self.embed_dim, patch_height, patch_width
        )

        # Segmentation head
        logits = self.seg_head([feat_map])

        # Upsample to original resolution
        logits = F.interpolate(
            logits,
            size=(H, W),
            mode="bilinear",
            align_corners=False
        )

        return logits
