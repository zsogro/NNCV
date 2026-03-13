import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

try:
    from torchvision.models import vit_s_16
except ImportError:
    vit_s_16 = None

from head import LinearHead


class Model(nn.Module):

    EMBED_DIM = 384
    PATCH_SIZE = 16
    BACKBONE_WEIGHTS = "dinov3_vits16_pretrain_lvd1689m-08c60483.pth" 

    def __init__(self, in_channels=3, n_classes=19):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes

        if vit_s_16 is None:
            raise ImportError(
                "torchvision is required for vit_s_16. Install it with: pip install torchvision"
            )

        # Build ViT-S/16 backbone architecture, then load local checkpoint weights.
        self.backbone = vit_s_16(weights=None)
        self._load_backbone_weights(self.BACKBONE_WEIGHTS)

        # ViT-S/16 embedding size
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

        load_info = self.backbone.load_state_dict(cleaned_state_dict, strict=False)
        if len(load_info.missing_keys) == len(self.backbone.state_dict()):
            raise RuntimeError(
                "No backbone weights matched vit_s_16. "
                "Verify that the checkpoint matches torchvision VisionTransformer naming."
            )

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
