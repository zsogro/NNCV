import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOv2SegmentationModel(nn.Module):

    def __init__(self, num_classes=19):
        super().__init__()

        # Load pretrained DINOv2 backbone
        self.backbone = torch.hub.load(
            'facebookresearch/dinov2',
            'dinov2_vitb14'
        )

        embed_dim = 768  # ViT-B embedding size

        # Simple segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(embed_dim, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):

        B, C, H, W = x.shape

        # Extract patch tokens
        features = self.backbone.forward_features(x)

        patch_tokens = features["x_norm_patchtokens"]

        # Convert tokens -> feature map
        n_patches = patch_tokens.shape[1]
        patch_size = int(n_patches ** 0.5)

        feat_map = patch_tokens.permute(0, 2, 1).reshape(
            B, 768, patch_size, patch_size
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
