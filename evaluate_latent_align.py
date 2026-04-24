import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import MFT_dataset
import model_trainer_variational as model_trainer
import utils


FT_LABELS = ["Fx", "Ty", "Tz"]
FLOWER_LABELS = ["Position", "Velocity"]


def get_flower_labels(flower_prediction_target: str):
    if flower_prediction_target == "position":
        return ["Position"]
    return FLOWER_LABELS


def select_flower_channels(array, flower_prediction_target: str):
    if flower_prediction_target == "position":
        return array[..., :1]
    return array


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot <= 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def r2_score_flat(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return r2_score_np(y_true.reshape(-1), y_pred.reshape(-1))


def per_channel_r2(y_true: np.ndarray, y_pred: np.ndarray) -> list:
    return [r2_score_np(y_true[..., i], y_pred[..., i]) for i in range(y_true.shape[-1])]


def safe_sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def to_serializable(value):
    if isinstance(value, dict):
        return {str(key): to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def gather_loader_outputs(model, loader, mass_label: float):
    flower_true = []
    flower_true_mean = []
    flower_recon = []
    flower_recon_mean = []
    spike_true = []
    spike_recon = []
    cross_spike_recon = []
    ft_true = []
    ft_pred = []
    mass_logits = []
    moth_ids = []
    split_names = []
    flower_prediction_target = getattr(model, "flower_prediction_target", "position")

    with torch.no_grad():
        for batch in loader:
            covariates = batch[0]
            ft_mean = batch[2]
            spikes = batch[3]
            moth_id = batch[6]
            mass = batch[7]

            outputs = model.forward(covariates, spikes, mass, moth_id)

            flower_true.append(select_flower_channels(covariates.detach().cpu(), flower_prediction_target))
            flower_true_mean.append(select_flower_channels(covariates.mean(dim=1).detach().cpu(), flower_prediction_target))
            flower_recon.append(outputs["flower_recon"].detach().cpu())
            flower_recon_mean.append(outputs["flower_recon_mean"].detach().cpu())
            spike_true.append(spikes.detach().cpu())
            spike_recon.append(outputs["spike_recon"].detach().cpu())
            cross_spike_recon.append(outputs["cross_spike_recon"].detach().cpu())
            ft_true.append(ft_mean.detach().cpu())
            ft_pred.append(outputs["ft_pred"].detach().cpu())
            mass_logits.append(
                outputs["spike_latent"][:, model.shared_latent_dim : model.shared_latent_dim + 1].detach().cpu()
            )
            moth_ids.append(moth_id.detach().cpu().view(-1))
            split_names.extend(["high" if mass_label > 0.5 else "low"] * covariates.shape[0])

    num_items = len(split_names)
    return {
        "flower_true": torch.cat(flower_true, dim=0).numpy(),
        "flower_true_mean": torch.cat(flower_true_mean, dim=0).numpy(),
        "flower_recon": torch.cat(flower_recon, dim=0).numpy(),
        "flower_recon_mean": torch.cat(flower_recon_mean, dim=0).numpy(),
        "spike_true": torch.cat(spike_true, dim=0).numpy(),
        "spike_recon": torch.cat(spike_recon, dim=0).numpy(),
        "cross_spike_recon": torch.cat(cross_spike_recon, dim=0).numpy(),
        "ft_true": torch.cat(ft_true, dim=0).numpy(),
        "ft_pred": torch.cat(ft_pred, dim=0).numpy(),
        "mass_logits": torch.cat(mass_logits, dim=0).numpy(),
        "mass_labels": np.full((num_items, 1), mass_label, dtype=np.float32),
        "moth_ids": torch.cat(moth_ids, dim=0).numpy().astype(np.int64),
        "split_names": split_names,
    }


def combine_outputs(low_data, high_data):
    combined = {}
    for key in (
        "flower_true",
        "flower_true_mean",
        "flower_recon",
        "flower_recon_mean",
        "spike_true",
        "spike_recon",
        "cross_spike_recon",
        "ft_true",
        "ft_pred",
        "mass_logits",
        "mass_labels",
        "moth_ids",
    ):
        combined[key] = np.concatenate([low_data[key], high_data[key]], axis=0)
    combined["split_names"] = low_data["split_names"] + high_data["split_names"]
    return combined


def compute_metrics(data, mask=None, flower_recon_mode: str = "mean", flower_labels=None):
    if flower_labels is None:
        flower_labels = FLOWER_LABELS
    total_items = data["mass_labels"].shape[0]
    if mask is None:
        mask = np.ones(total_items, dtype=bool)
    else:
        mask = np.asarray(mask).reshape(-1).astype(bool)

    num_items = int(mask.sum())
    if num_items == 0:
        return {
            "num_samples": 0,
            "flower_recon_mode": flower_recon_mode,
            "mass_accuracy": None,
            "mass_probability_mean": None,
            "spike_r2": None,
            "cross_spike_r2": None,
            "flower_r2": None,
            "flower_mean_r2": None,
            "ft_r2": None,
            "ft_r2_per_channel": [None] * len(FT_LABELS),
            "flower_r2_per_channel": [None] * len(flower_labels),
            "flower_mean_r2_per_channel": [None] * len(flower_labels),
        }

    mass_probs = safe_sigmoid(data["mass_logits"][mask].reshape(-1))
    mass_true = data["mass_labels"][mask].reshape(-1)
    mass_pred = (mass_probs >= 0.5).astype(np.float32)

    ft_true = data["ft_true"][mask]
    ft_pred = data["ft_pred"][mask]
    flower_true = data["flower_true"][mask]
    flower_pred = data["flower_recon"][mask]
    flower_true_mean = data["flower_true_mean"][mask]
    flower_pred_mean = data["flower_recon_mean"][mask]
    flower_mean_r2 = r2_score_flat(flower_true_mean, flower_pred_mean)
    flower_mean_r2_per_channel = per_channel_r2(flower_true_mean, flower_pred_mean)
    flower_r2 = flower_mean_r2
    flower_r2_per_channel = flower_mean_r2_per_channel

    return {
        "num_samples": num_items,
        "flower_recon_mode": flower_recon_mode,
        "mass_accuracy": float(np.mean(mass_pred == mass_true)),
        "mass_probability_mean": float(np.mean(mass_probs)),
        "spike_r2": r2_score_flat(data["spike_true"][mask], data["spike_recon"][mask]),
        "cross_spike_r2": r2_score_flat(data["spike_true"][mask], data["cross_spike_recon"][mask]),
        "flower_r2": flower_r2,
        "flower_mean_r2": flower_mean_r2,
        "ft_r2": r2_score_flat(ft_true, ft_pred),
        "ft_r2_per_channel": per_channel_r2(ft_true, ft_pred),
        "flower_r2_per_channel": flower_r2_per_channel,
        "flower_mean_r2_per_channel": flower_mean_r2_per_channel,
    }


def save_mass_plot(output_dir: Path, mass_probs: np.ndarray, mass_labels: np.ndarray, accuracy: float):
    plt.figure(figsize=(6, 4), dpi=200)
    low_mask = mass_labels.reshape(-1) == 0
    high_mask = ~low_mask
    plt.hist(mass_probs[low_mask], bins=30, alpha=0.6, label="Low")
    plt.hist(mass_probs[high_mask], bins=30, alpha=0.6, label="High")
    plt.xlabel("Predicted mass probability")
    plt.ylabel("Count")
    plt.title(f"Mass prediction accuracy = {accuracy:.3f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "mass_prediction_hist.png")
    plt.close()


def save_spike_examples(
    output_dir: Path,
    spike_true: np.ndarray,
    spike_recon: np.ndarray,
    split_names,
    filename: str,
    recon_label: str,
):
    example_indices = [0, 1, len(split_names) // 2, min(len(split_names) // 2 + 1, len(split_names) - 1)]
    fig, axes = plt.subplots(len(example_indices), 2, figsize=(10, 10), dpi=200)
    for row, idx in enumerate(example_indices):
        axes[row, 0].imshow(spike_true[idx].T, aspect="auto", interpolation="none")
        axes[row, 0].set_title(f"{split_names[idx]} true spike #{idx}")
        axes[row, 1].imshow(spike_recon[idx].T, aspect="auto", interpolation="none")
        axes[row, 1].set_title(f"{split_names[idx]} {recon_label} spike #{idx}")
    plt.tight_layout()
    plt.savefig(output_dir / filename)
    plt.close(fig)


def save_spike_examples_by_moth(
    output_dir: Path,
    spike_true: np.ndarray,
    spike_recon: np.ndarray,
    cross_spike_recon: np.ndarray,
    split_names,
    moth_ids: np.ndarray,
    max_examples_per_moth: int = 4,
):
    by_moth_dir = output_dir / "spike_recon_by_moth"
    by_moth_dir.mkdir(parents=True, exist_ok=True)

    unique_moths = sorted(np.unique(moth_ids).astype(int).tolist())
    for moth_id in unique_moths:
        moth_indices = np.where(moth_ids == moth_id)[0]
        if moth_indices.size == 0:
            continue

        low_indices = [idx for idx in moth_indices if split_names[idx] == "low"]
        high_indices = [idx for idx in moth_indices if split_names[idx] == "high"]
        example_indices = low_indices[:2] + high_indices[:2]
        if len(example_indices) < max_examples_per_moth:
            for idx in moth_indices.tolist():
                if idx not in example_indices:
                    example_indices.append(idx)
                if len(example_indices) >= max_examples_per_moth:
                    break
        if not example_indices:
            continue

        fig, axes = plt.subplots(len(example_indices), 3, figsize=(12, 2.8 * len(example_indices)), dpi=200)
        axes = np.atleast_2d(axes)
        for row, idx in enumerate(example_indices):
            axes[row, 0].imshow(spike_true[idx].T, aspect="auto", interpolation="none")
            axes[row, 0].set_title(f"moth {moth_id} {split_names[idx]} true #{idx}")
            axes[row, 1].imshow(spike_recon[idx].T, aspect="auto", interpolation="none")
            axes[row, 1].set_title(f"moth {moth_id} {split_names[idx]} recon #{idx}")
            axes[row, 2].imshow(cross_spike_recon[idx].T, aspect="auto", interpolation="none")
            axes[row, 2].set_title(f"moth {moth_id} {split_names[idx]} cross #{idx}")
            axes[row, 0].set_ylabel("Muscle")
            for col in range(3):
                axes[row, col].set_xlabel("Timestep")

        plt.tight_layout()
        plt.savefig(by_moth_dir / f"moth_{moth_id}_spike_examples.png")
        plt.close(fig)


def save_flower_mean_scatter(
    output_dir: Path,
    flower_true_mean: np.ndarray,
    flower_pred_mean: np.ndarray,
    filename: str,
    title_prefix: str,
    r2_per_channel,
    flower_labels,
):
    fig, axes = plt.subplots(1, len(flower_labels), figsize=(5 * len(flower_labels), 4), dpi=200)
    axes = np.atleast_1d(axes)
    for col, label in enumerate(flower_labels):
        ax = axes[col]
        true_vals = flower_true_mean[:, col]
        pred_vals = flower_pred_mean[:, col]
        lo = min(true_vals.min(), pred_vals.min())
        hi = max(true_vals.max(), pred_vals.max())
        ax.scatter(true_vals, pred_vals, s=10, alpha=0.5)
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
        ax.set_xlabel("True")
        ax.set_ylabel("Pred")
        ax.set_title(f"{title_prefix} {label} R2={r2_per_channel[col]:.3f}")
    plt.tight_layout()
    plt.savefig(output_dir / filename)
    plt.close(fig)


def save_ft_scatter(
    output_dir: Path,
    ft_true: np.ndarray,
    ft_pred: np.ndarray,
    ft_r2_per_channel,
    filename: str,
    title_prefix: str,
    labels=FT_LABELS,
):
    n_channels = ft_true.shape[1]
    ncols = min(3, n_channels)
    nrows = int(np.ceil(n_channels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), dpi=200)
    axes = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes[:n_channels]):
        lo = min(ft_true[:, i].min(), ft_pred[:, i].min())
        hi = max(ft_true[:, i].max(), ft_pred[:, i].max())
        ax.scatter(ft_true[:, i], ft_pred[:, i], s=8, alpha=0.5)
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
        ax.set_title(f"{title_prefix} {labels[i]} R2={ft_r2_per_channel[i]:.3f}")
        ax.set_xlabel("True")
        ax.set_ylabel("Pred")
    for ax in axes[n_channels:]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / filename)
    plt.close(fig)


def save_by_moth_summary(output_dir: Path, by_moth_metrics: dict):
    moth_ids = sorted(int(moth_id) for moth_id in by_moth_metrics.keys())
    if not moth_ids:
        return

    metric_specs = [
        ("mass_accuracy", "Mass Accuracy"),
        ("spike_r2", "Spike R2"),
        ("cross_spike_r2", "Cross Spike R2"),
        ("flower_r2", "Flower R2"),
        ("ft_r2", "FT R2"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 14), dpi=200)
    x = np.arange(len(moth_ids))
    width = 0.35

    for ax, (metric_name, title) in zip(axes.flat, metric_specs):
        low_vals = []
        high_vals = []
        for moth_id in moth_ids:
            moth_metrics = by_moth_metrics[str(moth_id)]
            low_val = moth_metrics["low"].get(metric_name)
            high_val = moth_metrics["high"].get(metric_name)
            low_vals.append(np.nan if low_val is None else low_val)
            high_vals.append(np.nan if high_val is None else high_val)

        ax.bar(x - width / 2, low_vals, width=width, label="Low")
        ax.bar(x + width / 2, high_vals, width=width, label="High")
        ax.set_xticks(x)
        ax.set_xticklabels([str(moth_id) for moth_id in moth_ids])
        ax.set_title(title)
        ax.set_xlabel("Moth ID")
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "metrics_by_moth.png")
    plt.close(fig)


def main():
    args = utils.get_parser().parse_args()
    if isinstance(args.device, str) and args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"

    high_train_loader, high_test_loader, low_train_loader, low_test_loader = MFT_dataset.get_MFT(
        filepath_high=args.filepath_high,
        filepath_low=args.filepath_low,
        batch_size=args.batch_size,
        convolution=args.convolution,
        device=args.device,
        split=args.split,
        evaluate=True,
        data_seed=args.data_seed,
        ft_norm_mode=args.ft_norm_mode,
        use_filtered_data=True,
    )

    covariate_data, _, ft_data, muscle_data, _, _, _, _ = next(iter(high_train_loader))
    args.covariate_dims = [covariate_data.shape[-2], covariate_data.shape[-1]]
    args.ft_dims = [ft_data.shape[-1]]
    args.spike_dims = [muscle_data.shape[-2], muscle_data.shape[-1]]
    args.moth_number = 10

    model, _ = model_trainer.get_model_and_optimizer(args)
    model = model_trainer.load_checkpoint(args.save_path, model, args.device)
    model.eval()
    flower_recon_mode = "mean"
    flower_prediction_target = getattr(model, "flower_prediction_target", getattr(args, "flower_prediction_target", "position"))
    flower_labels = get_flower_labels(flower_prediction_target)

    low_data = gather_loader_outputs(model, low_test_loader, mass_label=0.0)
    high_data = gather_loader_outputs(model, high_test_loader, mass_label=1.0)
    combined = combine_outputs(low_data, high_data)

    mass_probs = safe_sigmoid(combined["mass_logits"].reshape(-1))
    mass_true = combined["mass_labels"].reshape(-1)
    low_mask = mass_true == 0
    high_mask = mass_true == 1

    by_moth_metrics = {}
    unique_moths = sorted(np.unique(combined["moth_ids"]).astype(int).tolist())
    for moth_id in unique_moths:
        moth_mask = combined["moth_ids"].reshape(-1) == moth_id
        by_moth_metrics[str(moth_id)] = {
            "overall": compute_metrics(combined, moth_mask, flower_recon_mode, flower_labels),
            "low": compute_metrics(combined, moth_mask & low_mask, flower_recon_mode, flower_labels),
            "high": compute_metrics(combined, moth_mask & high_mask, flower_recon_mode, flower_labels),
        }

    metrics = {
        "flower_recon_mode": flower_recon_mode,
        "flower_prediction_target": flower_prediction_target,
        "flower_labels": flower_labels,
        "overall": compute_metrics(combined, flower_recon_mode=flower_recon_mode, flower_labels=flower_labels),
        "by_mass": {
            "low": compute_metrics(combined, low_mask, flower_recon_mode, flower_labels),
            "high": compute_metrics(combined, high_mask, flower_recon_mode, flower_labels),
        },
        "by_moth": by_moth_metrics,
        "num_test_samples": int(combined["ft_true"].shape[0]),
    }

    output_dir = Path(args.save_path) / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(to_serializable(metrics), f, indent=2)

    save_mass_plot(output_dir, mass_probs, mass_true, metrics["overall"]["mass_accuracy"])
    save_spike_examples(
        output_dir,
        combined["spike_true"],
        combined["spike_recon"],
        combined["split_names"],
        "spike_reconstruction_examples.png",
        "recon",
    )
    save_spike_examples(
        output_dir,
        combined["spike_true"],
        combined["cross_spike_recon"],
        combined["split_names"],
        "cross_spike_reconstruction_examples.png",
        "cross-recon",
    )
    save_spike_examples_by_moth(
        output_dir,
        combined["spike_true"],
        combined["spike_recon"],
        combined["cross_spike_recon"],
        combined["split_names"],
        combined["moth_ids"],
    )
    save_flower_mean_scatter(
        output_dir,
        combined["flower_true_mean"],
        combined["flower_recon_mean"],
        "flower_mean_prediction_scatter.png",
        "Overall Mean Flower Recon",
        metrics["overall"]["flower_mean_r2_per_channel"],
        flower_labels,
    )
    save_ft_scatter(
        output_dir,
        combined["ft_true"],
        combined["ft_pred"],
        metrics["overall"]["ft_r2_per_channel"],
        "ft_prediction_scatter.png",
        "Overall",
    )
    save_ft_scatter(
        output_dir,
        combined["ft_true"][low_mask],
        combined["ft_pred"][low_mask],
        metrics["by_mass"]["low"]["ft_r2_per_channel"],
        "ft_prediction_scatter_low.png",
        "Low",
    )
    save_ft_scatter(
        output_dir,
        combined["ft_true"][high_mask],
        combined["ft_pred"][high_mask],
        metrics["by_mass"]["high"]["ft_r2_per_channel"],
        "ft_prediction_scatter_high.png",
        "High",
    )
    save_by_moth_summary(output_dir, metrics["by_moth"])

    print(json.dumps(to_serializable(metrics), indent=2))
    print(f"Saved analysis to {output_dir}")


if __name__ == "__main__":
    main()
