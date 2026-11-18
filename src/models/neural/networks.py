"""Network definitions: the embedding MLP and its optional sequence encoders."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class PeriodicEncoder(nn.Module):
    """Periodic embeddings for numeric inputs (Gorishniy et al., 2022).

    Each feature x becomes [cos(2*pi*c*x), sin(2*pi*c*x)] over learned
    frequencies c, then a per-feature linear layer and ReLU.
    """

    def __init__(self, n_features: int, n_frequencies: int, sigma: float, out_dim: int = 8):
        super().__init__()
        self.frequencies = nn.Parameter(torch.randn(n_features, n_frequencies) * sigma)
        bound = 1 / math.sqrt(2 * n_frequencies)
        self.weight = nn.Parameter(torch.empty(n_features, 2 * n_frequencies, out_dim).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.empty(n_features, out_dim).uniform_(-bound, bound))
        self.out_features = n_features * out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        angles = 2 * math.pi * x.unsqueeze(-1) * self.frequencies
        periodic = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
        encoded = torch.einsum("bfk,fkd->bfd", periodic, self.weight) + self.bias
        return torch.relu(encoded).flatten(1)


class SequenceEncoder(nn.Module):
    """A GRU over right-padded event sequences; empty sequences give zeros."""

    def __init__(self, input_dim: int, hidden: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, batch_first=True)
        self.out_features = hidden + 1

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        present = lengths > 0
        packed = pack_padded_sequence(
            x, lengths.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        state = hidden[-1] * present.unsqueeze(1).to(hidden.dtype)
        return torch.cat([state, present.unsqueeze(1).to(hidden.dtype)], dim=1)


class FlightDelayNet(nn.Module):
    """Embeddings plus numerics into an MLP; GRU encoders are added for Step 7."""

    def __init__(
        self,
        cardinalities: list[int],
        embedding_dims: list[int],
        numeric_dim: int,
        hidden_layers: list[int],
        dropout: float,
        numeric_encoding: str = "linear",
        periodic_frequencies: int = 8,
        periodic_sigma: float = 1.0,
        output_bias: float = 0.0,
        weather_dim: int | None = None,
        departure_dim: int | None = None,
        departure_carrier_index: int | None = None,
        gru_hidden: int | None = None,
    ):
        super().__init__()
        if len(cardinalities) != len(embedding_dims):
            raise ValueError("Each categorical input needs one embedding size")
        self.embeddings = nn.ModuleList(
            nn.Embedding(cardinality, dim) for cardinality, dim in zip(cardinalities, embedding_dims)
        )
        if numeric_encoding == "periodic" and numeric_dim:
            self.numeric = PeriodicEncoder(numeric_dim, periodic_frequencies, periodic_sigma)
            numeric_out = self.numeric.out_features
        elif numeric_encoding in {"linear", "periodic"}:
            self.numeric = nn.Identity()
            numeric_out = numeric_dim
        else:
            raise ValueError(f"Unknown numeric encoding {numeric_encoding!r}")
        width = sum(embedding_dims) + numeric_out

        self.weather = None
        if weather_dim:
            if not gru_hidden:
                raise ValueError("Sequence encoders need gru_hidden")
            self.weather = SequenceEncoder(weather_dim, gru_hidden)
            width += self.weather.out_features
        self.departures = None
        self.departure_carrier_index = departure_carrier_index
        if departure_dim:
            if not gru_hidden or departure_carrier_index is None:
                raise ValueError("Departure sequences need gru_hidden and the carrier embedding")
            carrier_dim = embedding_dims[departure_carrier_index]
            self.departures = SequenceEncoder(departure_dim + carrier_dim, gru_hidden)
            width += self.departures.out_features

        layers: list[nn.Module] = []
        for size in hidden_layers:
            layers += [nn.Linear(width, size), nn.ReLU(), nn.Dropout(dropout)]
            width = size
        self.mlp = nn.Sequential(*layers)
        self.output = nn.Linear(width, 1)
        with torch.no_grad():
            self.output.bias.fill_(output_bias)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [embedding(batch["cat"][:, i]) for i, embedding in enumerate(self.embeddings)]
        parts.append(self.numeric(batch["num"]))
        if self.weather is not None:
            parts.append(self.weather(batch["wx"], batch["wx_len"]))
        if self.departures is not None:
            carrier = self.embeddings[self.departure_carrier_index](batch["dep_carrier"])
            parts.append(self.departures(torch.cat([batch["dep"], carrier], dim=-1), batch["dep_len"]))
        return self.output(self.mlp(torch.cat(parts, dim=1))).squeeze(1)
