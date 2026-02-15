import torch
from torch import nn


class SimpleMLP(nn.Module):
    """
    A simple MLP with fixed hidden dimensions.

    Architecture:
      - If num_layers == 1: a single Linear mapping: input_dim → output_dim.
      - Otherwise: input_dim → hidden_channels → ... → hidden_channels → output_dim.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_channels: int,
        num_layers: int,
        output_dim: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, hidden_channels))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_channels, hidden_channels))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_channels, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, data):
        """Run forward on a Tensor or an object with `.features`."""
        if isinstance(data, torch.Tensor):
            features = data
        elif hasattr(data, "features"):
            features = data.features
        else:
            raise TypeError(
                "SimpleMLP.forward expected a Tensor or an object with a 'features' attribute"
            )
        out = self.model(features)
        return out.squeeze(-1) if out.shape[-1] == 1 else out
