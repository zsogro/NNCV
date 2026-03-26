import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from head import MLPHead
from ood_model import OOD_Detector

class Model(nn.Module):

    EMBED_DIM = 1024 # large:1024, base:768
    PATCH_SIZE = 16
    RESOLUTION = 512
    PRETRAINED_BACKBONE_WEIGHTS = "dinov3_vitl16_pretrained_weights.pth"
    BACKBONE_REPO = "dinov3"
    OOD_DETECTOR_WEIGHTS = "ood_detector_weights.pt"

    def __init__(
        self,
        in_channels=3,
        n_classes=19,
        load_backbone_for_training=True,
        head_hidden_channels=512,
        head_num_layers=5,
        ood=False,
        ood_threshold=0.95,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.load_backbone_for_training = load_backbone_for_training
        self.ood = ood
        self.ood_threshold = ood_threshold
        # Build DINOv3 ViT from local hub repo and load local checkpoint weights.
        self.backbone = torch.hub.load(
            str(self.BACKBONE_REPO),
            "dinov3_vitl16",
            source="local",
            pretrained=False,
        )
        if self.load_backbone_for_training:
            self._load_backbone_weights(self.PRETRAINED_BACKBONE_WEIGHTS)

        # ViT embedding size
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
        if self.ood:
            self.ood_detector = OOD_Detector(
                token_dim=self.embed_dim,
                flow_dim=128,
                hidden_dim=256,
                num_flow_layers=8,
                token_sample_size=4096,
            )
            self._load_ood_detector_weights(self.OOD_DETECTOR_WEIGHTS)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def _load_backbone_weights(self, weights_path):
        print(f"Loading backbone weights from {weights_path}...")
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

    def _load_ood_detector_weights(self, weights_path):
        print(f"Loading OOD detector weights from {weights_path}...")
        if not hasattr(self, 'ood_detector') or self.ood_detector is None:
            raise RuntimeError("OOD detector not initialized. Set ood=True first.")
        
        checkpoint = torch.load(weights_path, map_location="cpu")
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            if "detector_state_dict" in checkpoint and isinstance(checkpoint["detector_state_dict"], dict):
                state_dict = checkpoint["detector_state_dict"]
            elif "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
                state_dict = checkpoint["model"]

        if not isinstance(state_dict, dict):
            raise TypeError("Checkpoint does not contain a valid state_dict")

        # Remove "ood_detector." prefix from keys if present
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("ood_detector."):
                new_key = key.replace("ood_detector.", "", 1)
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        self.ood_detector.load_state_dict(new_state_dict, strict=True)
        if isinstance(checkpoint, dict):
            threshold = checkpoint.get("threshold")
            if threshold is not None:
                self.ood_detector.threshold = float(threshold)

            id_score_mean = checkpoint.get("id_score_mean")
            id_score_std = checkpoint.get("id_score_std")
            if id_score_mean is not None and id_score_std is not None:
                if float(id_score_std) > 0:
                    self.ood_detector.set_score_calibration(
                        mean=float(id_score_mean),
                        std=float(id_score_std),
                    )

    def load_model_state_dict(self, checkpoint_path):
        """Load model state dict, filtering out OOD detector keys if not in checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint
        
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
                state_dict = checkpoint["model"]
        
        # Filter out OOD detector keys if OOD is not enabled or keys don't exist
        filtered_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("ood_detector."):
                if self.ood:
                    filtered_state_dict[key] = value
                # Skip OOD keys if OOD is disabled
            else:
                filtered_state_dict[key] = value
        
        self.load_state_dict(filtered_state_dict, strict=False)

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

        if self.ood:
            is_ood, probability = self.ood_detector.predict_ood(
                patch_tokens,
                threshold=self.ood_threshold,
            )
            return logits, is_ood, probability
        else:
            return logits
