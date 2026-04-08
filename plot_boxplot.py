import json
import matplotlib.pyplot as plt
import wandb
from evaluate_sem_sim import compute_metrics


def get_fold_values(sweep_id, metrics):
    """Get per-fold values from a W&B sweep."""
    api = wandb.Api()
    entity = "ferzcam"
    project = "indigena"
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    metrics_summary = {metric: [] for metric in metrics}

    for run in sweep.runs:
        raw = run.summary._json_dict
        run_data = json.loads(raw) if isinstance(raw, str) else raw
        for metric in metrics:
            if metric in run_data:
                metrics_summary[metric].append(run_data[metric])

    for metric, values in metrics_summary.items():
        assert len(values) == 10, f"Expected 10 values for '{metric}', got {len(values)}"

    return metrics_summary


def get_lin_bma_fold_values(root_dir="data/baseline_results"):
    """Get per-fold values from Lin-BMA result files."""
    metrics_to_extract = ["mr", "mrr", "hits@1", "hits@3", "hits@10", "hits@100", "auc"]
    metrics_summary = {m: [] for m in metrics_to_extract}

    for fold in range(10):
        input_file = f"{root_dir}/resnik_lin_bma_fold{fold}_results.txt"
        _, macro_metrics = compute_metrics(input_file, output_ranks=True)
        for metric in metrics_to_extract:
            metrics_summary[metric].append(macro_metrics[metric])

    return metrics_summary


def main():
    # ConvKB-D inductive graph4
    convkbd_metrics = [
        "test_imac_bma_mr", "test_imac_bma_mrr",
        "test_imac_bma_hits@1", "test_imac_bma_hits@3",
        "test_imac_bma_hits@10", "test_imac_bma_hits@100",
        "test_imac_bma_auc",
    ]
    convkbd_values = get_fold_values("sung7r49", convkbd_metrics)

    # Lin-BMA
    lin_bma_values = get_lin_bma_fold_values()

    # Map metric names for comparison
    metric_pairs = [
        ("mr", "test_imac_bma_mr", "Mean Rank"),
        ("mrr", "test_imac_bma_mrr", "MRR"),
        ("hits@1", "test_imac_bma_hits@1", "Hits@1"),
        ("hits@3", "test_imac_bma_hits@3", "Hits@3"),
        ("hits@10", "test_imac_bma_hits@10", "Hits@10"),
        ("hits@100", "test_imac_bma_hits@100", "Hits@100"),
        ("auc", "test_imac_bma_auc", "AUC"),
    ]

    n_metrics = len(metric_pairs)
    ncols = 4
    nrows = (n_metrics + ncols - 1) // ncols
    _, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = axes.flatten()

    for i, (lin_key, convkbd_key, title) in enumerate(metric_pairs):
        ax = axes[i]
        data = [lin_bma_values[lin_key], convkbd_values[convkbd_key]]
        bp = ax.boxplot(data, labels=["Lin-BMA", "ConvKB-D"], patch_artist=True,
                        widths=0.5)
        bp["boxes"][0].set_facecolor("#4C72B0")
        bp["boxes"][1].set_facecolor("#DD8452")
        ax.set_title(title, fontsize=14)
        ax.set_ylabel("Score")
        ax.grid(axis="y", linestyle="--", alpha=0.7)

    for j in range(n_metrics, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Lin-BMA vs ConvKB-D (Inductive, Graph 4)", fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig("boxplot_lin_bma_vs_convkbd.png", dpi=300, bbox_inches="tight")
    plt.savefig("boxplot_lin_bma_vs_convkbd.pdf", bbox_inches="tight")
    print("Saved boxplot_lin_bma_vs_convkbd.png and .pdf")


if __name__ == "__main__":
    main()
