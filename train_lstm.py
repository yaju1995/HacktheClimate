"""Train and evaluate the wind-power forecasting LSTM.

Examples:
    python train_lstm.py --mode train --data Data/T1.csv --save-dir artifacts
    python train_lstm.py --mode evaluate --data Data/T1.csv --save-dir artifacts
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.dataloader import LoadForecastDataLoader
from src.models.lstm import LSTMForecast


DEFAULT_FEATURES = [
    "Wind Speed (m/s)",
    "Theoretical_Power_Curve (KWh)",
    "Wind Direction (°)",
]
DEFAULT_TARGET = "LV ActivePower (kW)"


class LSTMDataset(Dataset):
    """Create lookback/horizon windows from scaled arrays."""

    def __init__(
        self,
        features: np.ndarray,
        target: np.ndarray,
        lookback: int,
        horizon: int,
    ) -> None:
        self.features = features
        self.target = target
        self.lookback = lookback
        self.horizon = horizon
        sample_count = len(features) - lookback - horizon + 1
        if sample_count <= 0:
            raise ValueError(
                "The split is too short for the requested lookback and horizon."
            )

    def __len__(self) -> int:
        return len(self.features) - self.lookback - self.horizon + 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        feature_window = self.features[index : index + self.lookback]
        target_window = self.target[
            index + self.lookback : index + self.lookback + self.horizon
        ]
        return (
            torch.as_tensor(feature_window, dtype=torch.float32),
            torch.as_tensor(target_window, dtype=torch.float32),
        )


def make_loader(
    csv_path: Path,
    lookback: int,
    horizon: int,
) -> LoadForecastDataLoader:
    loader = LoadForecastDataLoader(
        csv_path=csv_path,
        datetime_col="Date/Time",
        datetime_format="%d %m %Y %H:%M",
        features=DEFAULT_FEATURES,
        target=DEFAULT_TARGET,
        lookback=lookback,
        horizon=horizon,
        expected_frequency="1h",
    )
    loader.summary()
    return loader


def split_and_scale(
    loader: LoadForecastDataLoader,
) -> tuple[
    StandardScaler,
    StandardScaler,
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    dataframe = loader.get_dataframe()
    train_end = int(len(dataframe) * 0.70)
    val_end = int(len(dataframe) * 0.85)
    train_df = dataframe.iloc[:train_end]
    val_df = dataframe.iloc[train_end:val_end]
    test_df = dataframe.iloc[val_end:]

    feature_scaler = StandardScaler().fit(train_df[loader.features])
    target_scaler = StandardScaler().fit(train_df[loader.target])

    feature_arrays = (
        feature_scaler.transform(train_df[loader.features]),
        feature_scaler.transform(val_df[loader.features]),
        feature_scaler.transform(test_df[loader.features]),
    )
    target_arrays = (
        target_scaler.transform(train_df[loader.target]),
        target_scaler.transform(val_df[loader.target]),
        target_scaler.transform(test_df[loader.target]),
    )
    return feature_scaler, target_scaler, feature_arrays, target_arrays


def make_dataloaders(
    loader: LoadForecastDataLoader,
    feature_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    target_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    datasets = [
        LSTMDataset(features, target, loader.lookback, loader.horizon)
        for features, target in zip(feature_arrays, target_arrays)
    ]
    return (
        DataLoader(datasets[0], batch_size=batch_size, shuffle=True),
        DataLoader(datasets[1], batch_size=batch_size, shuffle=False),
        DataLoader(datasets[2], batch_size=batch_size, shuffle=False),
    )


def make_test_loader(
    loader: LoadForecastDataLoader,
    feature_scaler: StandardScaler,
    target_scaler: StandardScaler,
    batch_size: int,
) -> DataLoader:
    dataframe = loader.get_dataframe()
    val_end = int(len(dataframe) * 0.85)
    test_df = dataframe.iloc[val_end:]
    test_features = feature_scaler.transform(test_df[loader.features].to_numpy())
    test_target = target_scaler.transform(test_df[loader.target].to_numpy())
    return DataLoader(
        LSTMDataset(
            test_features,
            test_target,
            loader.lookback,
            loader.horizon,
        ),
        batch_size=batch_size,
        shuffle=False,
    )


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for features, targets in dataloader:
        features, targets = features.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = criterion(model(features), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for features, targets in dataloader:
            loss = criterion(
                model(features.to(device)),
                targets.to(device),
            )
            total_loss += loss.item()
    return total_loss / len(dataloader)


def scaler_state(scaler: StandardScaler) -> dict[str, list[float]]:
    return {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "var": scaler.var_.tolist(),
        "n_samples_seen": np.asarray(scaler.n_samples_seen_).tolist(),
    }


def restore_scaler(state: dict[str, Any]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(state["mean"], dtype=np.float64)
    scaler.scale_ = np.asarray(state["scale"], dtype=np.float64)
    scaler.var_ = np.asarray(state["var"], dtype=np.float64)
    scaler.n_features_in_ = len(scaler.mean_)
    scaler.n_samples_seen_ = np.asarray(state["n_samples_seen"])
    return scaler


def save_checkpoint(
    path: Path,
    model: LSTMForecast,
    feature_scaler: StandardScaler,
    target_scaler: StandardScaler,
    config: dict[str, Any],
    best_val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_scaler": scaler_state(feature_scaler),
            "target_scaler": scaler_state(target_scaler),
            "config": config,
            "best_val_loss": best_val_loss,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[LSTMForecast, StandardScaler, StandardScaler, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = LSTMForecast(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        horizon=config["horizon"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return (
        model,
        restore_scaler(checkpoint["feature_scaler"]),
        restore_scaler(checkpoint["target_scaler"]),
        config,
    )


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    target_scaler: StandardScaler,
    device: torch.device,
) -> tuple[float, float]:
    predictions: list[np.ndarray] = []
    actuals: list[np.ndarray] = []
    with torch.no_grad():
        for features, targets in dataloader:
            predictions.append(model(features.to(device)).cpu().numpy())
            actuals.append(targets.numpy())

    predictions_array = np.concatenate(predictions)
    actuals_array = np.concatenate(actuals)
    predictions_original = target_scaler.inverse_transform(
        predictions_array.reshape(-1, 1)
    ).reshape(predictions_array.shape)
    actuals_original = target_scaler.inverse_transform(
        actuals_array.reshape(-1, 1)
    ).reshape(actuals_array.shape)
    mae = mean_absolute_error(actuals_original.ravel(), predictions_original.ravel())
    rmse = np.sqrt(mean_squared_error(actuals_original.ravel(), predictions_original.ravel()))
    print(f"Predictions: {predictions_array.shape}")
    print(f"Actual:      {actuals_array.shape}")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    return float(mae), float(rmse)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "evaluate"), default="train")
    parser.add_argument("--data", type=Path, default=Path("Data/T1.csv"))
    parser.add_argument("--save-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lookback", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("--epochs and --batch-size must be positive.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = args.save_dir / "best_lstm.pt"

    if args.mode == "train":
        loader = make_loader(args.data, args.lookback, args.horizon)
        feature_scaler, target_scaler, feature_arrays, target_arrays = split_and_scale(loader)
        train_loader, val_loader, test_loader = make_dataloaders(
            loader, feature_arrays, target_arrays, args.batch_size
        )
        model_config = {
            "input_size": len(loader.features),
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "horizon": loader.horizon,
            "dropout": args.dropout,
            "lookback": loader.lookback,
            "features": loader.features,
            "target": loader.target,
        }
        model = LSTMForecast(
            input_size=model_config["input_size"],
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            horizon=loader.horizon,
            dropout=args.dropout,
        ).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        best_val_loss = float("inf")

        print(f"Using device: {device}")
        for epoch in range(args.epochs):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss = validate(model, val_loader, criterion, device)
            print(
                f"Epoch {epoch + 1:02d}/{args.epochs} "
                f"| Train: {train_loss:.5f} | Val: {val_loss:.5f}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    checkpoint_path,
                    model,
                    feature_scaler,
                    target_scaler,
                    model_config,
                    best_val_loss,
                )
        print(f"Saved best model to {checkpoint_path}")
    else:
        model, feature_scaler, target_scaler, config = load_checkpoint(
            checkpoint_path, device
        )
        loader = make_loader(
            args.data,
            config["lookback"],
            config["horizon"],
        )
        test_loader = make_test_loader(
            loader,
            feature_scaler,
            target_scaler,
            args.batch_size,
        )
        evaluate(model, test_loader, target_scaler, device)


if __name__ == "__main__":
    main()
