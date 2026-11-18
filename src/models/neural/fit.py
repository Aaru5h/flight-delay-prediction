"""Training loop, chronological early stopping, prediction and checkpoints."""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.models.neural.networks import FlightDelayNet
from src.models.neural.preprocess import TabularPreprocessor, embedding_dim

Arrays = dict[str, np.ndarray]
INTEGER_KEYS = {"cat", "wx_len", "dep_len", "dep_carrier"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def take(arrays: Arrays, rows: np.ndarray) -> Arrays:
    return {key: value[rows] for key, value in arrays.items()}


def _to_tensors(arrays: Arrays, rows: np.ndarray | slice, device: torch.device) -> dict[str, torch.Tensor]:
    batch = {}
    for key, value in arrays.items():
        part = np.asarray(value[rows])
        dtype = torch.long if key in INTEGER_KEYS else torch.float32
        batch[key] = torch.as_tensor(part, dtype=dtype, device=device)
    return batch


def _rows(arrays: Arrays) -> int:
    sizes = {len(value) for value in arrays.values()}
    if len(sizes) != 1:
        raise ValueError("All input arrays must have the same number of rows")
    return sizes.pop()


def build_network(
    preprocessor: TabularPreprocessor, candidate: dict, settings: dict, prevalence: float,
    sequence_dims: dict | None = None, *, seed: int | None = None,
) -> tuple[FlightDelayNet, dict]:
    """Return the network and the keyword arguments needed to rebuild it.

    Pass ``seed`` to make the initial weights reproducible.
    """
    if seed is not None:
        set_seed(seed)
    cardinalities = preprocessor.cardinalities
    prevalence = min(max(prevalence, 1e-4), 1 - 1e-4)
    kwargs = {
        "cardinalities": cardinalities,
        "embedding_dims": [
            embedding_dim(size, settings["preprocessing"]["max_embedding_dim"]) for size in cardinalities
        ],
        "numeric_dim": preprocessor.numeric_dim,
        "hidden_layers": list(candidate["hidden_layers"]),
        "dropout": candidate["dropout"],
        "numeric_encoding": candidate["numeric_encoding"],
        "periodic_frequencies": candidate["periodic_frequencies"],
        "periodic_sigma": candidate["periodic_sigma"],
        "output_bias": math.log(prevalence / (1 - prevalence)),
        **(sequence_dims or {}),
    }
    return FlightDelayNet(**kwargs), kwargs


def _learning_rate_factor(epoch_progress: float, warmup: int, total: int) -> float:
    """Linear warmup, then cosine decay over the fixed maximum epoch budget.

    The schedule depends only on the epoch position, so a refit that stops
    after the selected number of epochs follows exactly the same prefix.
    """
    if warmup and epoch_progress < warmup:
        return max(epoch_progress / warmup, 1e-3)
    span = max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * min((epoch_progress - warmup) / span, 1.0)))


@dataclass
class TrainResult:
    model: FlightDelayNet
    epochs_run: int
    best_epoch: int
    holdout_losses: list[float] = field(default_factory=list)
    train_losses: list[float] = field(default_factory=list)


def mean_log_loss(model: FlightDelayNet, arrays: Arrays, y: np.ndarray, batch_size: int, device) -> float:
    logits = predict_logits(model, arrays, batch_size, device)
    target = torch.as_tensor(y, dtype=torch.float32)
    loss = nn.functional.binary_cross_entropy_with_logits(torch.as_tensor(logits), target)
    return float(loss)


def train_network(
    model: FlightDelayNet,
    arrays: Arrays,
    y: np.ndarray,
    candidate: dict,
    training: dict,
    seed: int,
    *,
    holdout: tuple[Arrays, np.ndarray] | None = None,
    epochs: int | None = None,
    device: str = "cpu",
) -> TrainResult:
    """Train with AdamW. With a holdout, stop early and restore the best epoch.

    Without a holdout, train for exactly ``epochs`` epochs (the refit path).
    """
    if (holdout is None) == (epochs is None):
        raise ValueError("Pass either a holdout for early stopping or a fixed epoch count")
    n = _rows(arrays)
    if n != len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError("Training labels must be binary and aligned with the inputs")
    target = torch.device(device)
    set_seed(seed)
    model.to(target)
    max_epochs = training["max_epochs"]
    run_epochs = epochs if epochs is not None else max_epochs
    if not 1 <= run_epochs <= max_epochs:
        raise ValueError("Epoch count must be between 1 and max_epochs")
    positives = float(y.sum())
    pos_weight = None
    if candidate["positive_weighting"] == "balanced" and 0 < positives < n:
        pos_weight = torch.tensor((n - positives) / positives, device=target)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=candidate["learning_rate"], weight_decay=candidate["weight_decay"],
    )
    batch_size = int(candidate["batch_size"])
    steps_per_epoch = math.ceil(n / batch_size)
    generator = torch.Generator().manual_seed(seed)
    labels = torch.as_tensor(y, dtype=torch.float32)

    best_loss, best_epoch, best_state, stale = math.inf, 0, None, 0
    result = TrainResult(model, 0, 0)
    for epoch in range(run_epochs):
        model.train()
        order = torch.randperm(n, generator=generator).numpy()
        running = 0.0
        for step in range(steps_per_epoch):
            rows = order[step * batch_size:(step + 1) * batch_size]
            factor = _learning_rate_factor(epoch + step / steps_per_epoch, training["warmup_epochs"], max_epochs)
            for group in optimizer.param_groups:
                group["lr"] = candidate["learning_rate"] * factor
            batch = _to_tensors(arrays, rows, target)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch), labels[rows].to(target))
            loss.backward()
            optimizer.step()
            running += loss.item() * len(rows)
        result.train_losses.append(running / n)
        result.epochs_run = epoch + 1
        if holdout is not None:
            holdout_loss = mean_log_loss(model, holdout[0], holdout[1], training["eval_batch_size"], target)
            result.holdout_losses.append(holdout_loss)
            if holdout_loss < best_loss - 1e-6:
                best_loss, best_epoch, stale = holdout_loss, epoch + 1, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
                if stale >= training["patience_epochs"]:
                    break
    if holdout is not None:
        if best_state is None:
            raise ValueError("The early-stopping holdout loss was never finite")
        model.load_state_dict(best_state)
        result.best_epoch = best_epoch
    else:
        result.best_epoch = run_epochs
    model.eval()
    return result


@torch.no_grad()
def predict_logits(model: FlightDelayNet, arrays: Arrays, batch_size: int, device="cpu") -> np.ndarray:
    model.eval()
    target = torch.device(device)
    n = _rows(arrays)
    outputs = [
        model(_to_tensors(arrays, slice(start, start + batch_size), target)).cpu().numpy()
        for start in range(0, n, batch_size)
    ]
    return np.concatenate(outputs).astype(np.float64) if outputs else np.zeros(0)


def predict_proba(model: FlightDelayNet, arrays: Arrays, batch_size: int, device="cpu") -> np.ndarray:
    logits = predict_logits(model, arrays, batch_size, device)
    probabilities = 1 / (1 + np.exp(-logits))
    if not (np.isfinite(probabilities) & (probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("Model produced invalid probabilities")
    return probabilities


def save_checkpoint(path: Path, model: FlightDelayNet, network_kwargs: dict, preprocessor: TabularPreprocessor,
                    *, sequence_state: dict | None = None, metadata: dict | None = None) -> None:
    torch.save({
        "format": "flight-delay-neural-1",
        "state_dict": model.state_dict(),
        "network_kwargs": network_kwargs,
        "preprocessor": preprocessor.state(),
        "sequence": sequence_state,
        "metadata": metadata or {},
    }, path)


class NeuralScorer:
    """Reload a saved checkpoint and score feature rows (and sequences, if used)."""

    def __init__(self, path: Path, device: str = "cpu"):
        payload = torch.load(path, map_location=device, weights_only=False)
        if payload.get("format") != "flight-delay-neural-1":
            raise ValueError(f"{path} is not a flight-delay neural checkpoint")
        self.preprocessor = TabularPreprocessor.from_state(payload["preprocessor"])
        self.network_kwargs = payload["network_kwargs"]
        self.sequence_state = payload["sequence"]
        self.metadata = payload["metadata"]
        self.model = FlightDelayNet(**self.network_kwargs)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.device = device

    @property
    def allowlist(self) -> list[str]:
        return self.preprocessor.allowlist

    def predict_proba(self, frame, sequence_arrays: Arrays | None = None, batch_size: int = 8192) -> np.ndarray:
        codes, numeric = self.preprocessor.transform(frame)
        arrays = {"cat": codes, "num": numeric}
        if self.sequence_state is not None:
            if sequence_arrays is None:
                raise ValueError("This checkpoint needs sequence inputs")
            from src.models.neural.sequences import SequencePreprocessor
            arrays.update(SequencePreprocessor.from_state(self.sequence_state).transform(sequence_arrays))
        return predict_proba(self.model, arrays, batch_size, self.device)
