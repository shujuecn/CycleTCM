"""
MLLM-Model.
Used for MLLM-enhanced feature classification.
"""

import torch
import torch.nn as nn

class MLLM_Adapter(nn.Module):
    def __init__(self, input_dim: int = 2560, output_dim: int = 2048):
        super().__init__()
        self.mlp1 = nn.Linear(input_dim, output_dim)
        self.ln1 = nn.LayerNorm(output_dim)
        self.relu = nn.ReLU(inplace=True)
        self.ln2 = nn.LayerNorm(output_dim)
        self.mlp2 = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        x = self.mlp1(x)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.ln2(x)
        x = self.mlp2(x)

        return x


class MLLM_Model(MLLM_Adapter):
    def __init__(self, input_dim=2560, output_dim=2048, num_classes1=8, num_classes2=5):
        super().__init__(input_dim, output_dim)
        self.classifier1 = nn.Linear(output_dim, num_classes1)
        self.classifier2 = nn.Linear(output_dim, num_classes2)

    def forward(self, x):
        x = super().forward(x)
        syndrome_pred = self.classifier1(x)
        organ_pred = self.classifier2(x)

        return syndrome_pred, organ_pred
