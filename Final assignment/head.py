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