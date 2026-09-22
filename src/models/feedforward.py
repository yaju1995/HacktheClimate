from torch import nn


class FeedForwardForecast(nn.Module):
    """Feed-forward model that maps input features to a forecast horizon."""

    def __init__(
        self,
        input_size,
        hidden_size=128,
        num_layers=2,
        horizon=1,
        dropout=0.2,
    ):
        super().__init__()

        if input_size <= 0:
            raise ValueError("input_size must be greater than zero.")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be greater than zero.")
        if num_layers <= 0:
            raise ValueError("num_layers must be greater than zero.")
        if horizon <= 0:
            raise ValueError("horizon must be greater than zero.")

        layers = [
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        ]

        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                ]
            )

        layers.extend(
            [
                nn.Dropout(dropout),
                nn.Linear(hidden_size, horizon),
            ]
        )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x: [batch, input_size]
        if x.ndim != 2:
            raise ValueError(
                "Expected input with shape [batch, input_size]."
            )

        # [batch, horizon, 1]
        return self.network(x).unsqueeze(-1)
