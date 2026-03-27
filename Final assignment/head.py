# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearHead(nn.Module):
    """Linear layer ."""

    def __init__(
        self,
        in_channels,
        n_output_channels,
        use_batchnorm=True,
        use_cls_token=False,
        dropout=0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.channels = sum(in_channels)
        if use_cls_token:
            self.channels *= 2  # concatenate CLS to patch tokens
        self.n_output_channels = n_output_channels
        self.use_cls_token = use_cls_token
        self.batchnorm_layer = nn.SyncBatchNorm(self.channels) if use_batchnorm else nn.Identity(self.channels)
        self.conv = nn.Conv2d(self.channels, self.n_output_channels, kernel_size=1, padding=0, stride=1)
        self.dropout = nn.Dropout2d(dropout)
        nn.init.normal_(self.conv.weight, mean=0, std=0.01)
        nn.init.constant_(self.conv.bias, 0)

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """
        inputs = [
            torch.nn.functional.interpolate(
                input=x,
                size=inputs[0].shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            for x in inputs
        ]
        inputs = torch.cat(inputs, dim=1)
        return inputs

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        # accept lists (for cls token)
        inputs = list(inputs)
        for i, x in enumerate(inputs):
            if self.use_cls_token:
                assert len(x) == 2, "Missing class tokens"
                x, cls_token = x[0], x[1]
                if len(x.shape) == 2:
                    x = x[:, :, None, None]
                cls_token = cls_token[:, :, None, None].expand_as(x)
                inputs[i] = torch.cat((x, cls_token), 1)
            else:
                if len(x.shape) == 2:
                    x = x[:, :, None, None]
                inputs[i] = x
        x = self._transform_inputs(inputs)
        return x

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.dropout(output)
        output = self.batchnorm_layer(output)
        output = self.conv(output)
        return output

    def predict(self, x, rescale_to=(512, 512)):
        """
        Predict function used in evaluation.
        No dropout is used, and the output is rescaled to the ground truth
        for computing metrics.
        """
        x = self._forward_feature(x)
        x = self.batchnorm_layer(x)
        x = self.conv(x)
        x = F.interpolate(input=x, size=rescale_to, mode="bilinear")
        return x


class MLPHead(nn.Module):
    """Multi layer linear."""

    def __init__(
        self,
        in_channels,
        n_output_channels,
        hidden_channels=512,
        num_layers=3,
        use_batchnorm=True,
        use_cls_token=False,
        dropout=0.1,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("MLPHead requires at least 2 layers")

        self.in_channels = in_channels
        self.channels = sum(in_channels)
        if use_cls_token:
            self.channels *= 2  # concatenate CLS to patch tokens
        self.n_output_channels = n_output_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.use_cls_token = use_cls_token
        self.input_norm = nn.SyncBatchNorm(self.channels) if use_batchnorm else nn.Identity()
        self.dropout = nn.Dropout2d(dropout)

        layers = []
        in_dim = self.channels
        for _ in range(self.num_layers - 1):
            conv = nn.Conv2d(in_dim, self.hidden_channels, kernel_size=1, padding=0, stride=1)
            nn.init.normal_(conv.weight, mean=0, std=0.01)
            nn.init.constant_(conv.bias, 0)
            layers.append(conv)
            if use_batchnorm:
                layers.append(nn.SyncBatchNorm(self.hidden_channels))
            layers.append(nn.GELU())
            in_dim = self.hidden_channels

        self.hidden_layers = nn.Sequential(*layers)
        self.out_conv = nn.Conv2d(in_dim, self.n_output_channels, kernel_size=1, padding=0, stride=1)
        nn.init.normal_(self.out_conv.weight, mean=0, std=0.01)
        nn.init.constant_(self.out_conv.bias, 0)

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """
        inputs = [
            torch.nn.functional.interpolate(
                input=x,
                size=inputs[0].shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            for x in inputs
        ]
        inputs = torch.cat(inputs, dim=1)
        return inputs

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        # accept lists (for cls token)
        inputs = list(inputs)
        for i, x in enumerate(inputs):
            if self.use_cls_token:
                assert len(x) == 2, "Missing class tokens"
                x, cls_token = x[0], x[1]
                if len(x.shape) == 2:
                    x = x[:, :, None, None]
                cls_token = cls_token[:, :, None, None].expand_as(x)
                inputs[i] = torch.cat((x, cls_token), 1)
            else:
                if len(x.shape) == 2:
                    x = x[:, :, None, None]
                inputs[i] = x
        x = self._transform_inputs(inputs)
        return x

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.dropout(output)
        output = self.input_norm(output)
        output = self.hidden_layers(output)
        output = self.out_conv(output)
        return output

    def predict(self, x, rescale_to=(512, 512)):
        """
        Predict function used in evaluation.
        No dropout is used, and the output is rescaled to the ground truth
        for computing metrics.
        """
        x = self._forward_feature(x)
        x = self.input_norm(x)
        x = self.hidden_layers(x)
        x = self.out_conv(x)
        x = F.interpolate(input=x, size=rescale_to, mode="bilinear")
        return x


class AllMLPDecoder(nn.Module):
    """Lightweight all-MLP decoder for multi-depth feature fusion.

    Each depth feature is projected with a 1x1 MLP-style block, then all
    projected maps are concatenated and fused with another 1x1 MLP.
    """

    def __init__(
        self,
        in_channels,
        n_output_channels,
        embed_channels=256,
        hidden_channels=256,
        num_fuse_layers=2,
        use_batchnorm=False,
        dropout=0.1,
    ):
        super().__init__()
        if len(in_channels) < 2:
            raise ValueError("AllMLPDecoder expects at least 2 input feature levels")
        if num_fuse_layers < 1:
            raise ValueError("num_fuse_layers must be >= 1")

        self.in_channels = in_channels
        self.n_output_channels = n_output_channels
        self.embed_channels = embed_channels
        self.hidden_channels = hidden_channels

        self.level_mlps = nn.ModuleList()
        for in_ch in in_channels:
            layers = [nn.Conv2d(in_ch, embed_channels, kernel_size=1, stride=1, padding=0)]
            if use_batchnorm:
                layers.append(nn.SyncBatchNorm(embed_channels))
            layers.append(nn.GELU())
            self.level_mlps.append(nn.Sequential(*layers))

        fused_in = embed_channels * len(in_channels)
        fuse_layers = []
        in_dim = fused_in
        for _ in range(num_fuse_layers - 1):
            fuse_layers.append(nn.Conv2d(in_dim, hidden_channels, kernel_size=1, stride=1, padding=0))
            if use_batchnorm:
                fuse_layers.append(nn.SyncBatchNorm(hidden_channels))
            fuse_layers.append(nn.GELU())
            in_dim = hidden_channels
        self.fuse_mlp = nn.Sequential(*fuse_layers)

        self.dropout = nn.Dropout2d(dropout)
        self.out_conv = nn.Conv2d(in_dim, n_output_channels, kernel_size=1, stride=1, padding=0)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def _fuse(self, inputs):
        inputs = list(inputs)
        if len(inputs) != len(self.level_mlps):
            raise ValueError(
                f"Expected {len(self.level_mlps)} input feature maps, got {len(inputs)}"
            )

        target_size = inputs[0].shape[2:]
        projected = []
        for feat, mlp in zip(inputs, self.level_mlps):
            x = mlp(feat)
            if x.shape[2:] != target_size:
                x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
            projected.append(x)

        x = torch.cat(projected, dim=1)
        x = self.fuse_mlp(x)
        x = self.dropout(x)
        return self.out_conv(x)

    def forward(self, inputs):
        return self._fuse(inputs)

    def predict(self, x, rescale_to=(512, 512)):
        x = self._fuse(x)
        x = F.interpolate(input=x, size=rescale_to, mode="bilinear", align_corners=False)
        return x