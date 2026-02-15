import torch.nn as nn


class EncoderMLP(nn.Module):
    """
    A flexible MLP that creates a network with decreasing hidden dimensions.

    The network architecture is built as follows:
      - Linear interpolation is done between the input_dim and hidden_channels
      - When used as an encoder, the final layer outputs hidden_channels.
 

    Args:
        input_dim (int): Dimensionality of the input.
        hidden_channels (int): Base hidden dimension.
        num_layers (int): Number of layers used in the encoder (for the decreasing sizes).
        dropout (float): Dropout rate applied after each hidden layer.
    """

    def __init__(
        self,
        input_dim: int = 10,
        hidden_channels: int = 32,
        num_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        # Build a list of sizes: starting at input_dim then decreasing
        # We'll do linear interpolation between the input_dim and hidden_channels
        difference = hidden_channels - input_dim
        delta = difference // num_layers
        
        sizes = [input_dim] + [
            input_dim + delta * (i + 1) for i in range(num_layers - 1)
        ] + [hidden_channels]
            
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            # For every layer except the last one, add ReLU (and dropout if requested)
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the encoder.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Encoded representation
        """
        return self.model(x)
