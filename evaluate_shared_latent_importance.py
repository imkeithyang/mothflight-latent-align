import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import MFT_dataset
import model_trainer_variational as model_trainer
import utils


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


def mse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def per_channel_mse(y_true: np.ndarray, y_pred: np.ndarray) -> list:
    return [mse_np(y_true[..., i], y_pred[..., i]) for i in range(y_true.shape[-1])]


def to_serializable(value):
    if isinstance(value, dict):
        return {str(key): to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def load_runtime_args():
    parser = utils.get_parser()
    cli_args = parser.parse_args()
    default_args = parser.parse_args([])

    config_path = Path(cli_args.save_path) / "config.json"
    if config_path.exists():
        args = utils.load_config(cli_args.save_path)
        for key, default_value in vars(default_args).items():
            cli_value = getattr(cli_args, key)
            if cli_value != default_value:
                setattr(args, key, cli_value)
    else:
        args = cli_args

    if isinstance(args.device, str) and args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    return args


def gather_loader_outputs(model, loader, mass_label: float):
    shared_dim = model.shared_latent_dim
    ablated_recons = {dim: [] for dim in range(shared_dim)}
    spike_true = []
    spike_recon = []
    all_shared_zero_recon = []
    split_names = []

    with torch.no_grad():
        for batch in loader:
            spikes = batch[3].to(model.device)
            moth_id = batch[6].to(model.device)
            mass = batch[7].to(model.device).float().view(spikes.shape[0], -1)
            cond = model._build_condition(mass, moth_id)
            spike_latent = model._encode_spike(spikes, cond)

            spike_recon.append(model.spike_decoder(spike_latent, cond).detach().cpu())

            all_shared_zero = spike_latent.clone()
            all_shared_zero[:, :shared_dim] = 0.0
            all_shared_zero_recon.append(model.spike_decoder(all_shared_zero, cond).detach().cpu())

            for dim in range(shared_dim):
                ablated_latent = spike_latent.clone()
                ablated_latent[:, dim] = 0.0
                ablated_recons[dim].append(model.spike_decoder(ablated_latent, cond).detach().cpu())

            spike_true.append(spikes.detach().cpu())
            split_names.extend(["high" if mass_label > 0.5 else "low"] * spikes.shape[0])

    num_items = len(split_names)
    return {
        "spike_true": torch.cat(spike_true, dim=0).numpy(),
        "spike_recon": torch.cat(spike_recon, dim=0).numpy(),
        "all_shared_zero_recon": torch.cat(all_shared_zero_recon, dim=0).numpy(),
        "ablated_recons": {
            str(dim): torch.cat(chunks, dim=0).numpy() for dim, chunks in ablated_recons.items()
        },
        "mass_labels": np.full((num_items,), mass_label, dtype=np.float32),
        "split_names": split_names,
    }


def combine_outputs(low_data, high_data):
    return {
        "spike_true": np.concatenate([low_data["spike_true"], high_data["spike_true"]], axis=0),
        "spike_recon": np.concatenate([low_data["spike_recon"], high_data["spike_recon"]], axis=0),
        "all_shared_zero_recon": np.concatenate(
            [low_data["all_shared_zero_recon"], high_data["all_shared_zero_recon"]],
            axis=0,
        ),
        "ablated_recons": {
            dim: np.concatenate([low_data["ablated_recons"][dim], high_data["ablated_recons"][dim]], axis=0)
            for dim in low_data["ablated_recons"].keys()
        },
        "mass_labels": np.concatenate([low_data["mass_labels"], high_data["mass_labels"]], axis=0),
        "split_names": low_data["split_names"] + high_data["split_names"],
    }


def compute_spike_metrics(spike_true: np.ndarray, spike_pred: np.ndarray) -> dict:
    return {
        "spike_r2": r2_score_flat(spike_true, spike_pred),
        "spike_mse": mse_np(spike_true, spike_pred),
        "spike_r2_per_muscle": per_channel_r2(spike_true, spike_pred),
        "spike_mse_per_muscle": per_channel_mse(spike_true, spike_pred),
    }


def compute_importance_metrics(data, mask=None):
    total_items = data["mass_labels"].shape[0]
    if mask is None:
        mask = np.ones(total_items, dtype=bool)
    else:
        mask = np.asarray(mask).reshape(-1).astype(bool)

    if int(mask.sum()) == 0:
        return {
            "num_samples": 0,
            "baseline": None,
            "all_shared_zero": None,
            "per_dimension": [],
            "importance_ranking_by_delta_r2": [],
        }

    spike_true = data["spike_true"][mask]
    baseline = compute_spike_metrics(spike_true, data["spike_recon"][mask])
    all_shared_zero = compute_spike_metrics(spike_true, data["all_shared_zero_recon"][mask])
    all_shared_zero["delta_r2"] = all_shared_zero["spike_r2"] - baseline["spike_r2"]
    all_shared_zero["delta_mse"] = all_shared_zero["spike_mse"] - baseline["spike_mse"]
    all_shared_zero["delta_r2_per_muscle"] = [
        ablated - base
        for ablated, base in zip(all_shared_zero["spike_r2_per_muscle"], baseline["spike_r2_per_muscle"])
    ]
    all_shared_zero["delta_mse_per_muscle"] = [
        ablated - base
        for ablated, base in zip(all_shared_zero["spike_mse_per_muscle"], baseline["spike_mse_per_muscle"])
    ]

    per_dimension = []
    for dim_key in sorted(data["ablated_recons"].keys(), key=int):
        dim = int(dim_key)
        ablated_metrics = compute_spike_metrics(spike_true, data["ablated_recons"][dim_key][mask])
        ablated_metrics["dimension"] = dim
        ablated_metrics["delta_r2"] = ablated_metrics["spike_r2"] - baseline["spike_r2"]
        ablated_metrics["delta_mse"] = ablated_metrics["spike_mse"] - baseline["spike_mse"]
        ablated_metrics["delta_r2_per_muscle"] = [
            ablated - base
            for ablated, base in zip(ablated_metrics["spike_r2_per_muscle"], baseline["spike_r2_per_muscle"])
        ]
        ablated_metrics["delta_mse_per_muscle"] = [
            ablated - base
            for ablated, base in zip(ablated_metrics["spike_mse_per_muscle"], baseline["spike_mse_per_muscle"])
        ]
        per_dimension.append(ablated_metrics)

    ranking = [
        {"dimension": item["dimension"], "delta_r2": item["delta_r2"], "delta_mse": item["delta_mse"]}
        for item in sorted(per_dimension, key=lambda item: item["delta_r2"])
    ]

    return {
        "num_samples": int(mask.sum()),
        "baseline": baseline,
        "all_shared_zero": all_shared_zero,
        "per_dimension": per_dimension,
        "importance_ranking_by_delta_r2": ranking,
    }


def save_importance_summary_plot(output_dir: Path, metrics: dict, prefix: str):
    per_dimension = metrics["per_dimension"]
    if not per_dimension:
        return

    dims = [item["dimension"] for item in per_dimension]
    delta_r2 = [item["delta_r2"] for item in per_dimension]
    delta_mse = [item["delta_mse"] for item in per_dimension]
    baseline_r2 = metrics["baseline"]["spike_r2"]
    baseline_mse = metrics["baseline"]["spike_mse"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=200)
    axes[0].bar(dims, delta_r2, color="tab:blue")
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set_xlabel("Shared latent dimension")
    axes[0].set_ylabel("Ablated R2 - baseline R2")
    axes[0].set_title(f"{prefix} spike recon impact on R2\nbaseline={baseline_r2:.3f}")

    axes[1].bar(dims, delta_mse, color="tab:red")
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xlabel("Shared latent dimension")
    axes[1].set_ylabel("Ablated MSE - baseline MSE")
    axes[1].set_title(f"{prefix} spike recon impact on MSE\nbaseline={baseline_mse:.4f}")

    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix.lower()}_shared_spike_latent_importance.png")
    plt.close(fig)


def save_importance_heatmap(output_dir: Path, metrics: dict, prefix: str):
    per_dimension = metrics["per_dimension"]
    if not per_dimension:
        return

    delta_r2_matrix = np.asarray([item["delta_r2_per_muscle"] for item in per_dimension], dtype=np.float32)
    delta_mse_matrix = np.asarray([item["delta_mse_per_muscle"] for item in per_dimension], dtype=np.float32)
    dims = [item["dimension"] for item in per_dimension]
    muscle_labels = [f"M{i + 1}" for i in range(delta_r2_matrix.shape[1])]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), dpi=200)

    im0 = axes[0].imshow(delta_r2_matrix, aspect="auto", cmap="coolwarm")
    axes[0].set_title(f"{prefix} delta R2 by shared dimension and muscle")
    axes[0].set_xlabel("Muscle")
    axes[0].set_ylabel("Shared dim")
    axes[0].set_xticks(np.arange(len(muscle_labels)))
    axes[0].set_xticklabels(muscle_labels)
    axes[0].set_yticks(np.arange(len(dims)))
    axes[0].set_yticklabels(dims)
    fig.colorbar(im0, ax=axes[0], shrink=0.9)

    im1 = axes[1].imshow(delta_mse_matrix, aspect="auto", cmap="viridis")
    axes[1].set_title(f"{prefix} delta MSE by shared dimension and muscle")
    axes[1].set_xlabel("Muscle")
    axes[1].set_ylabel("Shared dim")
    axes[1].set_xticks(np.arange(len(muscle_labels)))
    axes[1].set_xticklabels(muscle_labels)
    axes[1].set_yticks(np.arange(len(dims)))
    axes[1].set_yticklabels(dims)
    fig.colorbar(im1, ax=axes[1], shrink=0.9)

    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix.lower()}_shared_spike_latent_importance_by_muscle.png")
    plt.close(fig)


def main():
    args = load_runtime_args()

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
    )

    covariate_data, _, ft_data, muscle_data, _, _, _, _ = next(iter(high_train_loader))
    args.covariate_dims = [covariate_data.shape[-2], covariate_data.shape[-1]]
    args.ft_dims = [ft_data.shape[-1]]
    args.spike_dims = [muscle_data.shape[-2], muscle_data.shape[-1]]
    args.moth_number = 10

    model, _ = model_trainer.get_model_and_optimizer(args)
    model = model_trainer.load_checkpoint(args.save_path, model, args.device)
    model.eval()

    low_data = gather_loader_outputs(model, low_test_loader, mass_label=0.0)
    high_data = gather_loader_outputs(model, high_test_loader, mass_label=1.0)
    combined = combine_outputs(low_data, high_data)

    mass_labels = combined["mass_labels"].reshape(-1)
    metrics = {
        "shared_latent_dim": int(model.shared_latent_dim),
        "spike_latent_dim": int(model.spike_latent_dim),
        "overall": compute_importance_metrics(combined),
        "by_mass": {
            "low": compute_importance_metrics(combined, mass_labels == 0.0),
            "high": compute_importance_metrics(combined, mass_labels == 1.0),
        },
    }

    output_dir = Path(args.save_path) / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "shared_spike_latent_importance.json", "w") as f:
        json.dump(to_serializable(metrics), f, indent=2)

    save_importance_summary_plot(output_dir, metrics["overall"], "overall")
    save_importance_summary_plot(output_dir, metrics["by_mass"]["low"], "low")
    save_importance_summary_plot(output_dir, metrics["by_mass"]["high"], "high")
    save_importance_heatmap(output_dir, metrics["overall"], "overall")
    save_importance_heatmap(output_dir, metrics["by_mass"]["low"], "low")
    save_importance_heatmap(output_dir, metrics["by_mass"]["high"], "high")

    print(json.dumps(to_serializable(metrics), indent=2))
    print(f"Saved shared spike latent importance analysis to {output_dir}")


if __name__ == "__main__":
    main()
