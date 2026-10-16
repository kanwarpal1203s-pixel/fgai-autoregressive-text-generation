"""End-to-end experiment helpers: train all models, generate text, tabulate and plot."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from .data import PUNCT
from .generate import CorpusReference, generate, text_metrics
from .models import MODEL_SPECS, build_model, count_parameters
from .train import evaluate, set_seed, train_model

CONFIG = {
    "seq_len": 20,          # context window (previous tokens) per training example
    "min_freq": 2,          # words seen < 2 times in train split -> <unk>
    "embed_dim": 128,
    "hidden_dim": 256,
    "num_layers": 1,
    "dropout": 0.3,
    "batch_size": 256,
    "lr": 2e-3,
    "max_epochs": 20,
    "patience": 3,          # early stopping on validation loss
    "clip": 5.0,
    "seed": 42,
}

SEEDS = [
    "It is a truth universally acknowledged, that a single man in possession of a good fortune",
    "Elizabeth looked at Mr. Darcy and",
    "My dear Mr. Bennet, said his lady to him one day,",
    "She was determined to",
    "The ball at Netherfield was",
]
TEMPERATURES = [0.2, 0.5, 1.0, 1.5]
COLORS = {"SimpleRNN": "#1f77b4", "BiRNN": "#ff7f0e", "LSTM": "#2ca02c", "GRU": "#d62728"}


def train_all(data: dict, cfg: dict = CONFIG, model_dir: str | Path = "models",
              device: str = "cpu", names=tuple(MODEL_SPECS)):
    """Train every model under identical conditions; return models + results table."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    V = len(data["vocab"])
    trained, rows, histories = {}, [], {}
    for name in names:
        print(f"\n=== Training {name} ===", flush=True)
        set_seed(cfg["seed"])  # same initial RNG state + same batch order for all models
        model = build_model(name, V, embed_dim=cfg["embed_dim"], hidden_dim=cfg["hidden_dim"],
                            num_layers=cfg["num_layers"], dropout=cfg["dropout"])
        res = train_model(model, data["splits"], epochs=cfg["max_epochs"],
                          batch_size=cfg["batch_size"], lr=cfg["lr"], clip=cfg["clip"],
                          patience=cfg["patience"], device=device, seed=cfg["seed"])
        test = evaluate(model, *data["splits"]["test"], device=device)
        val = evaluate(model, *data["splits"]["val"], device=device)
        train_m = evaluate(model, *data["splits"]["train"], device=device)
        torch.save({"state_dict": model.state_dict(), "config": cfg, "name": name,
                    "vocab": data["vocab"].itos}, model_dir / f"{name}.pt")
        histories[name] = res["history"]
        rows.append({
            "model": name, "parameters": count_parameters(model),
            "best_epoch": res["best_epoch"], "epochs_run": res["epochs_run"],
            "train_loss": train_m["loss"], "train_ppl": train_m["perplexity"],
            "train_acc": train_m["accuracy"],
            "val_loss": val["loss"], "val_ppl": val["perplexity"], "val_acc": val["accuracy"],
            "test_loss": test["loss"], "test_ppl": test["perplexity"],
            "test_acc": test["accuracy"], "test_top5_acc": test["top5_accuracy"],
            "time_per_epoch_s": res["time_per_epoch"], "total_train_time_s": res["total_train_time"],
        })
        trained[name] = model
        print(f"  TEST loss {test['loss']:.4f} | ppl {test['perplexity']:.2f} | "
              f"acc {test['accuracy']:.4f} | top5 {test['top5_accuracy']:.4f}")
    return trained, pd.DataFrame(rows).set_index("model"), histories


def generate_all(trained: dict, data: dict, seeds=SEEDS, temperatures=TEMPERATURES,
                 n_words: int = 50, samples_per_seed: int = 3, seed: int = 123):
    """Generate from the same seeds/temperatures for every model; compute text metrics."""
    ref = CorpusReference(data["train_tokens"])
    samples, rows = [], []
    for name, model in trained.items():
        for T in temperatures:
            g = torch.Generator().manual_seed(seed)  # identical sampling RNG per (model, T)
            for s_idx, s in enumerate(seeds):
                for k in range(samples_per_seed):
                    out = generate(model, data["vocab"], s, n_words=n_words, temperature=T,
                                   seq_len=data["seq_len"], generator=g)
                    m = text_metrics(out["generated_tokens"], out["seed_tokens"], ref)
                    samples.append({"model": name, "temperature": T, "seed_id": s_idx,
                                    "sample": k, "seed": s, "text": out["text"],
                                    "continuation": out["continuation"], **m})
                    rows.append({"model": name, "temperature": T, **m})
    gen_df = pd.DataFrame(rows).groupby(["model", "temperature"]).mean()
    return pd.DataFrame(samples), gen_df


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _save(fig, out_dir, name):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / name, dpi=130)
    return fig


def plot_dataset(data, stats, out_dir):
    words = [t for t in data["tokens"] if t not in PUNCT]
    cnt = Counter(words)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    top = cnt.most_common(25)
    ax[0].bar([w for w, _ in top], [c for _, c in top], color="#4c72b0")
    ax[0].set_title("25 most frequent words")
    ax[0].tick_params(axis="x", rotation=70)
    freqs = sorted(cnt.values(), reverse=True)
    ax[1].loglog(range(1, len(freqs) + 1), freqs, ".", ms=3)
    ax[1].set_title("Zipf plot (rank vs frequency)")
    ax[1].set_xlabel("rank")
    ax[1].set_ylabel("frequency")
    ax[2].hist([x for x in stats["sentence_lengths"] if x <= 100], bins=50, color="#55a868")
    ax[2].set_title("Sentence length distribution (tokens)")
    ax[2].set_xlabel("tokens per sentence")
    return _save(fig, out_dir, "dataset_overview.png")


def plot_history(histories, out_dir):
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.5))
    for name, h in histories.items():
        ep = range(1, len(h["val_loss"]) + 1)
        c = COLORS[name]
        ax[0].plot(ep, h["train_loss"], "--", color=c, label=f"{name} train")
        ax[0].plot(ep, h["val_loss"], "-", color=c, label=f"{name} val")
        ax[1].plot(ep, h["val_ppl"], "-o", ms=3, color=c, label=name)
        ax[2].plot(ep, h["train_acc"], "--", color=c)
        ax[2].plot(ep, h["val_acc"], "-o", ms=3, color=c, label=name)
        ax[3].plot(ep, h["epoch_time"], "-o", ms=3, color=c, label=name)
    for a, t in zip(ax, ["Cross-entropy loss (dashed=train, solid=val)", "Validation perplexity",
                         "Accuracy (dashed=train, solid=val)", "Seconds per epoch"]):
        a.set_title(t)
        a.set_xlabel("epoch")
        a.grid(alpha=.3)
    ax[0].legend(fontsize=7, ncol=2)
    ax[1].legend()
    ax[1].set_ylim(top=min(ax[1].get_ylim()[1], 600))
    return _save(fig, out_dir, "training_curves.png")


def plot_comparison(results, out_dir):
    cols = [("test_ppl", "Test perplexity (lower=better)"),
            ("test_acc", "Test top-1 accuracy"),
            ("test_top5_acc", "Test top-5 accuracy"),
            ("time_per_epoch_s", "Training time per epoch (s)"),
            ("total_train_time_s", "Total training time (s)")]
    fig, ax = plt.subplots(1, len(cols), figsize=(22, 4.2))
    for a, (c, t) in zip(ax, cols):
        vals = results[c]
        bars = a.bar(results.index, vals, color=[COLORS[m] for m in results.index])
        a.bar_label(bars, fmt="%.3f" if vals.max() < 1 else "%.1f", fontsize=8)
        a.set_title(t)
        a.tick_params(axis="x", rotation=20)
    return _save(fig, out_dir, "model_comparison.png")


def plot_temperature(gen_df, out_dir):
    cols = [("distinct_2", "Distinct-2 (diversity, higher=more diverse)"),
            ("rep_3", "Rep-3 (repeated trigram rate)"),
            ("bigram_fluency", "Bigram fluency (local coherence)"),
            ("copy_rate_4", "4-gram copy rate (memorisation)")]
    fig, ax = plt.subplots(1, len(cols), figsize=(21, 4.3))
    for a, (c, t) in zip(ax, cols):
        for name in gen_df.index.get_level_values(0).unique():
            s = gen_df.loc[name][c]
            a.plot(s.index, s.values, "-o", color=COLORS[name], label=name)
        a.set_title(t)
        a.set_xlabel("temperature")
        a.grid(alpha=.3)
    ax[0].legend()
    return _save(fig, out_dir, "temperature_analysis.png")


# --------------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------------- #
def write_samples_markdown(samples: pd.DataFrame, path, seeds=SEEDS):
    """One sample (k=0) per model/temperature/seed, grouped by seed for side-by-side reading."""
    lines = ["# Generated text samples", "",
             "Each continuation is 50 generated tokens. Same seed, same sampling RNG for every "
             "model. Sample #0 of 3 is shown; all samples are in `generated_samples.csv`.", ""]
    first = samples[samples["sample"] == 0]
    for s_idx, s in enumerate(seeds):
        lines += [f"## Seed {s_idx + 1}: \"{s}\"", ""]
        for T in sorted(first["temperature"].unique()):
            lines += [f"### Temperature {T}", ""]
            for _, r in first[(first.seed_id == s_idx) & (first.temperature == T)].iterrows():
                lines += [f"- **{r.model}**: {r.continuation}"]
            lines += [""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")
