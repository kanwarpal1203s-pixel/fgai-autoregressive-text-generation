"""Training and quantitative evaluation (loss, perplexity, top-1 / top-5 accuracy, time)."""
from __future__ import annotations

import copy
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor, batch_size: int = 1024,
             device: str = "cpu") -> dict:
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    total_loss, correct1, correct5, n = 0.0, 0, 0, 0
    for i in range(0, len(y), batch_size):
        xb, yb = X[i:i + batch_size].to(device), y[i:i + batch_size].to(device)
        logits = model(xb)
        total_loss += crit(logits, yb).item()
        top5 = logits.topk(5, dim=1).indices
        correct1 += (top5[:, 0] == yb).sum().item()
        correct5 += (top5 == yb[:, None]).any(dim=1).sum().item()
        n += len(yb)
    loss = total_loss / n
    return {"loss": loss, "perplexity": math.exp(loss), "accuracy": correct1 / n,
            "top5_accuracy": correct5 / n}


def train_model(model: nn.Module, splits: dict, epochs: int = 20, batch_size: int = 256,
                lr: float = 2e-3, clip: float = 5.0, patience: int = 3, device: str = "cpu",
                seed: int = 42, verbose: bool = True) -> dict:
    """Mini-batch Adam training with gradient clipping and early stopping on val loss.

    The weights from the epoch with the lowest validation loss are restored at the end.
    """
    Xtr, ytr = splits["train"]
    Xva, yva = splits["val"]
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)
    crit = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(seed)

    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_ppl", "val_acc",
                               "epoch_time", "lr")}
    best_loss, best_state, best_epoch, bad = float("inf"), None, 0, 0
    t_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        perm = torch.randperm(len(ytr), generator=gen)
        run_loss, run_correct, n = 0.0, 0, 0
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xtr[idx].to(device), ytr[idx].to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            run_loss += loss.item() * len(yb)
            run_correct += (logits.argmax(1) == yb).sum().item()
            n += len(yb)
        epoch_time = time.perf_counter() - t0
        val = evaluate(model, Xva, yva, device=device)
        sched.step(val["loss"])

        history["train_loss"].append(run_loss / n)
        history["train_acc"].append(run_correct / n)
        history["val_loss"].append(val["loss"])
        history["val_ppl"].append(val["perplexity"])
        history["val_acc"].append(val["accuracy"])
        history["epoch_time"].append(epoch_time)
        history["lr"].append(opt.param_groups[0]["lr"])
        if verbose:
            print(f"  epoch {epoch:2d} | train loss {run_loss / n:.4f} acc {run_correct / n:.4f}"
                  f" | val loss {val['loss']:.4f} ppl {val['perplexity']:8.2f}"
                  f" acc {val['accuracy']:.4f} | {epoch_time:5.1f}s", flush=True)

        if val["loss"] < best_loss - 1e-4:
            best_loss, best_epoch, bad = val["loss"], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  early stopping (best epoch {best_epoch})")
                break
    total_time = time.perf_counter() - t_start
    model.load_state_dict(best_state)
    return {"history": history, "best_epoch": best_epoch, "epochs_run": len(history["val_loss"]),
            "total_train_time": total_time,
            "time_per_epoch": float(np.mean(history["epoch_time"]))}
