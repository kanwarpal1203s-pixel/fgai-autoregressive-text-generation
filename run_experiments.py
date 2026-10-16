"""Run the full experiment from the command line.

    python run_experiments.py            # full run (same as the notebook)
    python run_experiments.py --quick    # 2-epoch smoke test, writes to results_quick/

Outputs: results/ (CSV/JSON metrics, plots, generated samples) and models/ (checkpoints).
"""
import argparse
import copy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import torch  # noqa: E402

from textgen.data import dataset_statistics, prepare_data  # noqa: E402
from textgen.experiment import (CONFIG, generate_all, plot_comparison, plot_dataset,  # noqa: E402
                                plot_history, plot_temperature, save_json, train_all,
                                write_samples_markdown)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    cfg = copy.deepcopy(CONFIG)
    out, mdir = Path("results"), Path("models")
    if args.quick:
        cfg["max_epochs"] = 2
        out, mdir = Path("results_quick"), Path("models_quick")
    plots = out / "plots"
    out.mkdir(parents=True, exist_ok=True)

    data = prepare_data("data/pride_and_prejudice.txt", cfg["seq_len"], cfg["min_freq"])
    stats = dataset_statistics(data)
    plot_dataset(data, stats, plots)
    save_json({k: v for k, v in stats.items() if k != "sentence_lengths"},
              out / "dataset_stats.json")

    trained, results, histories = train_all(data, cfg, mdir)
    results.to_csv(out / "metrics.csv")
    save_json(histories, out / "training_history.json")
    save_json(cfg, out / "config.json")
    plot_history(histories, plots)
    plot_comparison(results, plots)
    print(results.round(4).to_string())

    samples, gen_df = generate_all(trained, data, samples_per_seed=1 if args.quick else 3)
    samples.to_csv(out / "generated_samples.csv", index=False)
    gen_df.to_csv(out / "generation_metrics.csv")
    write_samples_markdown(samples, out / "generated_samples.md")
    plot_temperature(gen_df, plots)
    print(gen_df.round(3).to_string())


if __name__ == "__main__":
    main()
