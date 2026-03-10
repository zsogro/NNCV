import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, in_channels=3, num_classes=19):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Load pretrained DINOv2 backbone
        # https://github.com/facebookresearch/dinov2
        self.backbone = torch.hub.load(
            'facebookresearch/dinov2',
            'dinov2_vits14'
        )
        # https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md
        # ViT-S (38M params): Patch size 14, embedding dimension 384, 6 heads, MLP FFN
        self.embed_dim = 384  # ViT-S embedding size 
        self.patch_size = 14  # ViT-S patch size

        # Simple segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(self.embed_dim, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, self.num_classes, kernel_size=1)
        )

    def forward(self, x):

        # Check if the input tensor has the expected number of channels
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, but got {x.shape[1]}")
        B, C, H, W = x.shape # batch size, channels, height, width

        # Extract patch tokens
        features = self.backbone.forward_features(x)

        patch_tokens = features["x_norm_patchtokens"]

        # Convert tokens -> feature map
        n_patches = patch_tokens.shape[1]
        patch_size = int(n_patches ** 0.5)

        feat_map = patch_tokens.permute(0, 2, 1).reshape(
            B, self.embed_dim, patch_size, patch_size
        )

        # Segmentation head
        logits = self.seg_head(feat_map)

        # Upsample to original resolution
        logits = F.interpolate(
            logits,
            size=(H, W),
            mode="bilinear",
            align_corners=False
        )

        return logits
