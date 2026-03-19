import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from head import LinearHead,MLPHead


class Model(nn.Module):

    EMBED_DIM = 768
    PATCH_SIZE = 16
    RESOLUTION = 512
    PRETRAINED_BACKBONE_WEIGHTS = "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
    BACKBONE_REPO = "dinov3"

    def __init__(
        self,
        in_channels=3,
        n_classes=19,
        load_backbone_for_training=True,
        head_hidden_channels=512,
        head_num_layers=3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.load_backbone_for_training = load_backbone_for_training

        # Build DINOv3 ViT-B/16 from local hub repo and load local checkpoint weights.
        self.backbone = torch.hub.load(
            str(self.BACKBONE_REPO),
            "dinov3_vitb16",
            source="local",
            pretrained=False,
        )
        if self.load_backbone_for_training:
            self._load_backbone_weights(self.PRETRAINED_BACKBONE_WEIGHTS)

        # ViT-B/16 embedding size
        self.embed_dim = self.EMBED_DIM
        self.patch_size = self.PATCH_SIZE

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.eval()

        self.seg_head = MLPHead(
            in_channels=[self.embed_dim],
            n_output_channels=self.n_classes,
            hidden_channels=head_hidden_channels,
            num_layers=head_num_layers,
            use_batchnorm=False,
            use_cls_token=False,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def _load_backbone_weights(self, weights_path):
        checkpoint_path = weights_path
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
                state_dict = checkpoint["model"]

        if not isinstance(state_dict, dict):
            raise TypeError("Checkpoint does not contain a valid state_dict")

        self.backbone.load_state_dict(state_dict, strict=True)

    def _forward_backbone_patch_tokens(self, x):
        features = self.backbone.forward_features(x)
        return features["x_norm_patchtokens"]

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
