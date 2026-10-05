"""
LipNet Deep Learning Architecture in PyTorch (FR-9, FR-10).
Compatible with torch.nn.CTCLoss and GPU-accelerated tensor pipelines.
"""

import torch
import torch.nn as nn
from typing import Tuple


class LipNetTorch(nn.Module):
    """
    PyTorch implementation of LipNet:
    3D-CNN Spatiotemporal Frontend + Bidirectional GRU Sequence Model + Linear CTC Projection.
    """
    def __init__(self, vocab_size: int = 39, dropout_rate: float = 0.5):
        super().__init__()
        self.vocab_size = vocab_size

        # 3D Convolutional Layers
        # Input tensor: (Batch, Channels=1, Frames=75, Height=46, Width=96)
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_rate),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  # (B, 32, 75, 23, 48)
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_rate),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  # (B, 64, 75, 11, 24)
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_rate),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  # (B, 96, 75, 5, 12)
        )

        # Spatial feature dimension after 3 pooling operations: 96 channels * 5 height * 12 width = 5760
        self.flatten_dim = 96 * 5 * 12

        # 2-Layer Bidirectional GRU
        self.gru1 = nn.GRU(
            input_size=self.flatten_dim,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.dropout1 = nn.Dropout(dropout_rate)

        self.gru2 = nn.GRU(
            input_size=512,  # 256 * 2 due to bidirectional
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.dropout2 = nn.Dropout(dropout_rate)

        # Projection to Vocabulary Size
        self.fc = nn.Linear(512, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, Frames=75, Height=46, Width=96, Channels=1)
               or (Batch, Channels=1, Frames=75, Height=46, Width=96)
        Returns:
            Log-probabilities of shape (Batch, Frames=75, Vocab_Size)
        """
        # If input is (B, T, H, W, C), permute to (B, C, T, H, W)
        if x.ndim == 5 and x.shape[-1] == 1:
            x = x.permute(0, 4, 1, 2, 3)

        b, c, t, h, w = x.shape

        # 3D CNN Feature Extraction
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)  # (B, 96, 75, 5, 12)

        # Permute and flatten spatial dims: (B, T, C * H * W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(b, t, -1)  # (B, 75, 5760)

        # BiGRU Sequence Modeling
        x, _ = self.gru1(x)
        x = self.dropout1(x)

        x, _ = self.gru2(x)
        x = self.dropout2(x)

        # Linear Projection
        logits = self.fc(x)  # (B, 75, vocab_size)
        log_probs = nn.functional.log_softmax(logits, dim=-1)
        return log_probs
