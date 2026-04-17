import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from head import MLPHead, AllMLPDecoder
from ood_model import OOD_Detector_v1, OOD_Detector_v2

class Model(nn.Module):

    EMBED_DIM = 1024 # large:1024, base:768
    PATCH_SIZE = 16
    RESOLUTION = 1024
    PRETRAINED_BACKBONE_WEIGHTS = "dinov3_vitl16_pretrained_weights.pth"
    BACKBONE_REPO = "dinov3"
    OOD_DETECTOR_WEIGHTS = "ood_detector_weights.pt"

    def __init__(
        self,
        in_channels=3,
        n_classes=19,
        load_backbone_for_training=True,
        head_hidden_channels=256,
        head_num_layers=3,
        use_multidepth_decoder=False,
        multidepth_feature_levels=8,
        ood=False,
        ood_type=1,  # 1 for OOD_Detector_v1, 2 for OOD_Detector_v2
        ood_threshold=0.95,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.load_backbone_for_training = load_backbone_for_training
        self.head_num_layers = head_num_layers
        self.head_hidden_channels = head_hidden_channels
        self.use_multidepth_decoder = use_multidepth_decoder
        self.multidepth_feature_levels = multidepth_feature_levels
        self.ood = ood
        self.ood_type = ood_type
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

        if self.use_multidepth_decoder:
            self.multidepth_indices = self._build_multidepth_indices(self.multidepth_feature_levels)
            print(f"Using multi-depth decoder with feature levels {self.multidepth_indices}, {self.head_num_layers} head layers, and {self.head_hidden_channels} hidden channels")
            self.seg_head = AllMLPDecoder(
                in_channels=[self.embed_dim] * len(self.multidepth_indices),
                n_output_channels=self.n_classes,
                embed_channels=256,
                hidden_channels=self.head_hidden_channels,
                num_fuse_layers=self.head_num_layers,
                use_batchnorm=False,
            )
        else:
            self.multidepth_indices = None
            print(f"Using standard MLP head with {self.head_num_layers} layers and {self.head_hidden_channels} hidden channels")
            self.seg_head = MLPHead(
                in_channels=[self.embed_dim],
                n_output_channels=self.n_classes,
                hidden_channels=self.head_hidden_channels,
                num_layers=self.head_num_layers,
                use_batchnorm=False,
                use_cls_token=False,
            )
        if self.ood:
            print(f"Initializing OOD detector with threshold {self.ood_threshold}")
            if self.ood_type == 1:
                self.ood_detector = OOD_Detector_v1(
                    token_dim=self.embed_dim,
                    flow_dim=128,
                    hidden_dim=256,
                    num_flow_layers=8,
                    token_sample_size=4096,
                )
            else:
                self.ood_detector = OOD_Detector_v2(
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

    def enable_backbone_finetune(self, last_n_blocks: int) -> int:
        """Unfreeze only the last N transformer blocks of the frozen backbone.

        Returns the number of trainable backbone parameters.
        """
        if last_n_blocks < 0:
            raise ValueError("last_n_blocks must be >= 0")

        total_blocks = len(self.backbone.blocks)
        if last_n_blocks > total_blocks:
            raise ValueError(
                f"Cannot unfreeze last {last_n_blocks} blocks: backbone has {total_blocks} blocks"
            )

        # Start from a fully frozen backbone, then selectively unfreeze tail blocks.
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        if last_n_blocks == 0:
            return 0

        for block in self.backbone.blocks[-last_n_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True

        # Keep final normalization trainable when fine-tuning tail blocks.
        if hasattr(self.backbone, "norm"):
            for parameter in self.backbone.norm.parameters():
                parameter.requires_grad = True

        return sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)

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

    def _build_multidepth_indices(self, num_levels: int) -> list[int]:
        if num_levels < 2:
            raise ValueError("multidepth_feature_levels must be >= 2")
        total_blocks = len(self.backbone.blocks)
        if num_levels > total_blocks:
            raise ValueError(
                f"multidepth_feature_levels={num_levels} exceeds backbone depth={total_blocks}"
            )

        indices = []
        for i in range(num_levels):
            idx = int((i + 1) * total_blocks / num_levels) - 1
            indices.append(max(idx, 0))
        return sorted(set(indices))

    def _forward_backbone_multidepth_maps(self, x):
        if self.multidepth_indices is None:
            raise RuntimeError("Multi-depth decoder not enabled")
        return list(
            self.backbone.get_intermediate_layers(
                x,
                n=self.multidepth_indices,
                reshape=True,
                norm=True,
            )
        )

    def forward(self, x):

        # Check if the input tensor has the expected number of channels
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, but got {x.shape[1]}")
        B, _, H, W = x.shape

        # Extract patch tokens
        with torch.no_grad():
            if self.use_multidepth_decoder:
                multi_feats = self._forward_backbone_multidepth_maps(x)
                last_feat = multi_feats[-1]
                patch_tokens = last_feat.flatten(2).transpose(1, 2)
            else:
                patch_tokens = self._forward_backbone_patch_tokens(x)

        # Convert tokens -> feature map
        n_patches = patch_tokens.shape[1]
        patch_height = H // self.patch_size
        patch_width = W // self.patch_size
        if patch_height * patch_width != n_patches:
            raise ValueError(
                f"Expected {patch_height * patch_width} patch tokens for input size {(H, W)}, got {n_patches}"
            )

        if self.use_multidepth_decoder:
            # Segmentation head with multi-depth backbone features.
            logits = self.seg_head(multi_feats)
        else:
            feat_map = patch_tokens.permute(0, 2, 1).reshape(
                B, self.embed_dim, patch_height, patch_width
            )
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


if __name__ == "__main__":
    # Example usage

    model = Model(
            in_channels=3,  # RGB images
            n_classes=19,  # 19 classes in the Cityscapes dataset
            load_backbone_for_training=False,
            head_hidden_channels=512,
            head_num_layers=4,
            use_multidepth_decoder=True,
            multidepth_feature_levels=4,
            ood=False,
            ood_threshold=0.95, 
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Params: {trainable_params:,}")