"""Recurrent next-word prediction models: Simple RNN, Bidirectional RNN, LSTM, GRU."""
from __future__ import annotations

import torch
import torch.nn as nn

MODEL_SPECS = {
    # name        : (cell,  bidirectional)
    "SimpleRNN": ("rnn", False),
    "BiRNN": ("rnn", True),
    "LSTM": ("lstm", False),
    "GRU": ("gru", False),
}


class RecurrentLM(nn.Module):
    """Embedding -> recurrent encoder over the context window -> linear softmax layer.

    The model reads a context of `seq_len` previous tokens and outputs logits for the
    next token (many-to-one). For the bidirectional variant the forward and backward
    final hidden states are concatenated; it only ever sees the *past* context window,
    so it remains a valid autoregressive model (the target is outside the window).
    """

    def __init__(self, vocab_size: int, cell: str = "lstm", bidirectional: bool = False,
                 embed_dim: int = 128, hidden_dim: int = 256, num_layers: int = 1,
                 dropout: float = 0.3):
        super().__init__()
        self.cell, self.bidirectional = cell, bidirectional
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[cell]
        extra = {"nonlinearity": "tanh"} if cell == "rnn" else {}
        self.rnn = rnn_cls(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                           bidirectional=bidirectional,
                           dropout=dropout if num_layers > 1 else 0.0, **extra)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.dropout(self.embedding(x))          # (B, T, E)
        _, h = self.rnn(emb)
        if isinstance(h, tuple):                       # LSTM returns (h_n, c_n)
            h = h[0]
        feat = torch.cat([h[-2], h[-1]], dim=1) if self.bidirectional else h[-1]
        return self.fc(self.dropout(feat))             # (B, V)


def build_model(name: str, vocab_size: int, **kwargs) -> RecurrentLM:
    cell, bi = MODEL_SPECS[name]
    return RecurrentLM(vocab_size, cell=cell, bidirectional=bi, **kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
