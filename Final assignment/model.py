import torch
import torch.nn as nn
import torch.nn.functional as F

from head import LinearHead


class Model(nn.Module):

    EMBED_DIM = 384
    PATCH_SIZE = 16

    def __init__(self, in_channels=3, n_classes=19):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes

        # Load pretrained DINOv3 backbone
        # https://github.com/facebookresearch/dinov3
        # Model is cached in torch.hub during Docker build for offline inference
        self.backbone = torch.hub.load(
            'facebookresearch/dinov3',
            'dinov3_vits16'
        )
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

    def _extract_patch_tokens(self, features):
        if isinstance(features, dict):
            for key in ("x_norm_patchtokens", "x_patchtokens", "patchtokens"):
                if key in features:
                    return features[key]
        if torch.is_tensor(features):
            return features
        raise KeyError("Could not find patch tokens in backbone features output")

    def forward(self, x):

        # Check if the input tensor has the expected number of channels
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, but got {x.shape[1]}")
        B, _, H, W = x.shape

        # Extract patch tokens
        with torch.no_grad():
            features = self.backbone.forward_features(x)

        patch_tokens = self._extract_patch_tokens(features)

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
