from torch import nn
import torch
from torch.utils.data import Dataset

class LSTMForecast(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        horizon,
        dropout=0.2
    ):

        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.horizon = horizon

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Map final hidden state to 24 forecast values
        self.fc = nn.Linear(
            hidden_size,
            horizon
        )

    def forward(self, x):

        # x:
        # [batch, lookback, features]

        output, (hidden, cell) = self.lstm(x)

        # Last LSTM hidden state
        last_hidden = hidden[-1]

        # Forecast next 24 hours
        out = self.fc(last_hidden)

        # [batch, horizon, 1]
        out = out.unsqueeze(-1)

        return out





class LSTMDataset(Dataset):

    def __init__(
        self,
        features,
        target,
        lookback,
        horizon
    ):

        self.features = features
        self.target = target

        self.lookback = lookback
        self.horizon = horizon

    def __len__(self):

        return (
            len(self.features)
            - self.lookback
            - self.horizon
            + 1
        )

    def __getitem__(self, index):

        X = self.features[
            index:
            index + self.lookback
        ]

        y = self.target[
            index + self.lookback:
            index + self.lookback + self.horizon
        ]

        X = torch.tensor(
            X,
            dtype=torch.float32
        )

        y = torch.tensor(
            y,
            dtype=torch.float32
        )

        return X, y