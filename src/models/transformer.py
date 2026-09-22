import torch
import torch.nn as nn


class TransformerForecast(nn.Module):

    def __init__(
        self,
        input_size,
        d_model=64,
        nhead=4,
        num_layers=2,
        horizon=24,
        dropout=0.2
    ):
        super().__init__()

        # Map input features → Transformer dimension
        self.input_projection = nn.Linear(
            input_size,
            d_model
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Map Transformer representation → forecast horizon
        self.output_layer = nn.Linear(
            d_model,
            horizon
        )

    def forward(self, x):

        x = self.input_projection(x)

        x = self.transformer(x)

        # Final timestep
        x = x[:, -1, :]

        # (batch, 24)
        x = self.output_layer(x)

        # (batch, 24, 1)
        x = x.unsqueeze(-1)

        return x