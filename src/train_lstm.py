
"""
train_lstm.py

Train an LSTM regression model on chronologically ordered bike-share data.
The module exposes helpers so notebooks can import and reuse them without
duplicating data-loading or training logic.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


def load_dataframe(path: str | Path, target: str = "cnt") -> pd.DataFrame:
    """Load processed CSV with a timestamp column and sort chronologically."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {path}")
    return df.sort_values("timestamp").reset_index(drop=True)


def default_feature_columns(df: pd.DataFrame, target: str) -> List[str]:
    """Use all numeric columns except timestamp + target when feature list omitted."""
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in numeric if col != target]


def build_sequences(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target: str,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert dataframe into overlapping sequences for LSTM consumption."""
    feature_values = df[feature_cols].values.astype(np.float32)
    targets = df[target].values.astype(np.float32)

    sequences: List[np.ndarray] = []
    y: List[float] = []
    total = len(df)
    for start in range(total - seq_len):
        end = start + seq_len
        sequences.append(feature_values[start:end])
        y.append(targets[end])

    if not sequences:
        raise ValueError("Not enough rows to create sequences; reduce seq_len.")

    return np.stack(sequences), np.array(y, dtype=np.float32)


def chronological_split(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float,
    val_frac: float,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Split sequences chronologically into train/val/test chunks."""
    n = len(X)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    train = (X[:train_end], y[:train_end])
    val = (X[train_end:val_end], y[train_end:val_end])
    test = (X[val_end:], y[val_end:])
    if len(val[0]) == 0 or len(test[0]) == 0:
        raise ValueError("Increase dataset size or adjust fractions; val/test are empty.")
    return train, val, test


class SequenceDataset(Dataset):
    """Torch Dataset wrapping pre-built sequence arrays."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class LSTMRegressor(nn.Module):
    """Minimal LSTM regressor with optional dropout."""

    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, max(1, hidden_size // 2)),
            nn.ReLU(),
            nn.Linear(max(1, hidden_size // 2), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        return self.head(last_hidden)


@dataclass
class TrainHistoryEntry:
    epoch: int
    train_loss: float
    val_loss: float
    val_rmse: float
    val_mae: float


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_squared_error(y_true, y_pred) ** 0.5)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device):
    """Return MSE loss along with stacked predictions/targets."""
    criterion = nn.MSELoss()
    model.eval()
    losses: List[float] = []
    preds: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    with torch.no_grad():
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            losses.append(loss.item())
            preds.append(outputs.cpu().numpy().ravel())
            targets.append(batch_y.cpu().numpy().ravel())
    pred_arr = np.concatenate(preds)
    target_arr = np.concatenate(targets)
    return float(np.mean(losses)), pred_arr, target_arr


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
):
    """Train the LSTM while tracking validation metrics."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: List[TrainHistoryEntry] = []
    best_state = None
    best_rmse = float("inf")

    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: List[float] = []
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        val_loss, val_preds, val_targets = evaluate_model(model, val_loader, device)
        val_rmse = rmse(val_targets, val_preds)
        val_mae = mae(val_targets, val_preds)
        entry = TrainHistoryEntry(
            epoch=epoch,
            train_loss=float(np.mean(train_losses)),
            val_loss=val_loss,
            val_rmse=val_rmse,
            val_mae=val_mae,
        )
        history.append(entry)

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_state = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "val_rmse": val_rmse,
                "val_mae": val_mae,
            }

    if best_state is None:
        best_state = {
            "epoch": epochs,
            "state_dict": model.state_dict(),
            "val_rmse": history[-1].val_rmse,
            "val_mae": history[-1].val_mae,
        }
    return history, best_state


def prepare_dataloaders(
    df: pd.DataFrame,
    target: str,
    seq_len: int,
    feature_cols: Sequence[str] | None,
    batch_size: int,
    train_frac: float,
    val_frac: float,
):
    """Build standardized sequences and wrap them into DataLoaders."""
    if feature_cols is None:
        feature_cols = default_feature_columns(df, target)

    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])

    X, y = build_sequences(df_scaled, feature_cols, target, seq_len)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = chronological_split(X, y, train_frac, val_frac)

    loaders = tuple(
        DataLoader(SequenceDataset(X_split, y_split), batch_size=batch_size, shuffle=shuffle)
        for (X_split, y_split), shuffle in zip(
            [(X_train, y_train), (X_val, y_val), (X_test, y_test)],
            [True, False, False],
        )
    )
    train_loader, val_loader, test_loader = loaders
    return train_loader, val_loader, test_loader, scaler


def run_experiment(
    data_path: str | Path,
    output_dir: str | Path,
    target: str = "cnt",
    seq_len: int = 24,
    feature_cols: Sequence[str] | None = None,
    batch_size: int = 128,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    epochs: int = 30,
    lr: float = 1e-3,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    use_gpu: bool = False,
):
    """High-level convenience wrapper used by notebooks/CLI."""
    df = load_dataframe(data_path, target=target)
    train_loader, val_loader, test_loader, scaler = prepare_dataloaders(
        df=df,
        target=target,
        seq_len=seq_len,
        feature_cols=feature_cols,
        batch_size=batch_size,
        train_frac=train_frac,
        val_frac=val_frac,
    )

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    model = LSTMRegressor(
        n_features=train_loader.dataset.X.shape[-1],  # type: ignore[attr-defined]
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )

    history, best_state = train_model(model, train_loader, val_loader, device, epochs, lr)
    model.load_state_dict(best_state["state_dict"])

    _, test_preds, test_targets = evaluate_model(model, test_loader, device)
    test_metrics = {
        "rmse": rmse(test_targets, test_preds),
        "mae": mae(test_targets, test_preds),
    }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    history_path = output_path / "lstm_train_history.json"
    with history_path.open("w") as f:
        json.dump([asdict(entry) for entry in history], f, indent=2)

    preds_df = pd.DataFrame({"y_true": test_targets, "y_pred": test_preds})
    preds_df.to_csv(output_path / "lstm_test_predictions.csv", index=False)

    metrics_path = output_path / "lstm_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(
            {
                "test": test_metrics,
                "best_val": {"rmse": best_state["val_rmse"], "mae": best_state["val_mae"]},
                "device": str(device),
            },
            f,
            indent=2,
        )

    torch.save({"model_state": model.state_dict(), "scaler": scaler}, output_path / "lstm_best.pt")

    return {
        "history_path": str(history_path),
        "predictions_path": str(output_path / "lstm_test_predictions.csv"),
        "metrics_path": str(metrics_path),
        "device": str(device),
        "history": history,
        "test_metrics": test_metrics,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train an LSTM regressor on bike-share demand.")
    parser.add_argument("--input", required=True, help="Path to processed CSV (output of prepare.py)")
    parser.add_argument("--output-dir", default="results", help="Directory to store history/metrics")
    parser.add_argument("--target", default="cnt")
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--use-gpu", action="store_true", help="Enable if CUDA device is available.")
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_experiment(
        data_path=args.input,
        output_dir=args.output_dir,
        target=args.target,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        epochs=args.epochs,
        lr=args.lr,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        use_gpu=args.use_gpu,
    )
    print(f"LSTM training complete. Metrics saved to {result['metrics_path']}")


if __name__ == "__main__":
    main()
