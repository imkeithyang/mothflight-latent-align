import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

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
    flower_true = []
    spike_true = []
    flower_latent = []
    spike_latent = []
    spike_latent_seq = []
    ft_true = []
    ft_pred = []
    moth_ids = []
    split_names = []
    mass_labels = []
    flower_prediction_target = getattr(model, "flower_prediction_target", "position")

    with torch.no_grad():
        for batch in loader:
            covariates = batch[0].to(model.device)
            ft_mean = batch[2].to(model.device)
            spikes = batch[3].to(model.device)
            moth_id = batch[6].to(model.device)
            mass = batch[7].to(model.device).float().view(covariates.shape[0], -1)

            outputs = model.forward(covariates, spikes, mass, moth_id)

            flower_true.append(select_flower_channels(covariates.detach().cpu(), flower_prediction_target))
            spike_true.append(spikes.detach().cpu())
            flower_latent.append(outputs["flower_latent"].detach().cpu())
            spike_latent.append(outputs["spike_latent"].detach().cpu())
            if "spike_latent_seq" in outputs:
                spike_latent_seq.append(outputs["spike_latent_seq"].detach().cpu())
            ft_true.append(ft_mean.detach().cpu())
            ft_pred.append(outputs["ft_pred"].detach().cpu())
            moth_ids.append(moth_id.detach().cpu().view(-1))
            split_names.extend(["high" if mass_label > 0.5 else "low"] * covariates.shape[0])
            mass_labels.append(np.full((covariates.shape[0], 1), mass_label, dtype=np.float32))

    num_items = len(split_names)
    return {
        "flower_true": torch.cat(flower_true, dim=0).numpy(),
        "spike_true": torch.cat(spike_true, dim=0).numpy(),
        "flower_latent": torch.cat(flower_latent, dim=0).numpy(),
        "spike_latent": torch.cat(spike_latent, dim=0).numpy(),
        "spike_latent_seq": torch.cat(spike_latent_seq, dim=0).numpy() if spike_latent_seq else None,
        "ft_true": torch.cat(ft_true, dim=0).numpy(),
        "ft_pred": torch.cat(ft_pred, dim=0).numpy(),
        "moth_ids": torch.cat(moth_ids, dim=0).numpy().astype(np.int64),
        "split_names": split_names,
        "mass_labels": np.concatenate(mass_labels, axis=0),
        "num_items": num_items,
    }


def combine_outputs(low_data, high_data):
    combined = {}
    for key in (
        "flower_true",
        "spike_true",
        "flower_latent",
        "spike_latent",
        "spike_latent_seq",
        "ft_true",
        "ft_pred",
        "moth_ids",
        "mass_labels",
    ):
        low_value = low_data.get(key, None)
        high_value = high_data.get(key, None)
        if low_value is None and high_value is None:
            combined[key] = None
        elif low_value is None:
            combined[key] = high_value
        elif high_value is None:
            combined[key] = low_value
        else:
            combined[key] = np.concatenate([low_value, high_value], axis=0)
    combined["split_names"] = low_data["split_names"] + high_data["split_names"]
    return combined


def pick_reference_indices(data, moth_id: int):
    references = []
    shared = data["spike_latent"][:, : data["shared_latent_dim"]]
    shared_seq = data.get("spike_latent_seq", None)
    for split_name in ("low", "high"):
        mask = (data["moth_ids"] == moth_id) & (np.asarray(data["split_names"]) == split_name)
        indices = np.where(mask)[0]
        if indices.size == 0:
            continue
        split_shared = shared[indices]
        center = np.median(split_shared, axis=0)
        ref_local_idx = int(np.argmin(np.linalg.norm(split_shared - center, axis=1)))
        ref_idx = int(indices[ref_local_idx])
        references.append(
            {
                "split": split_name,
                "index": ref_idx,
                "mass": float(data["mass_labels"][ref_idx, 0]),
                "moth_id": moth_id,
                "spike_latent": data["spike_latent"][ref_idx].astype(np.float32),
                "spike_latent_seq": None if shared_seq is None else shared_seq[ref_idx].astype(np.float32),
            }
        )
    return references


def build_sweep_bounds(
    data,
    moth_id: int,
    low_pct: float,
    high_pct: float,
    sweep_max: float = None,
    fixed_low: float = None,
    fixed_high: float = None,
):
    shared = data["spike_latent"][:, : data["shared_latent_dim"]]
    if fixed_low is not None and fixed_high is not None:
        low = np.full(shared.shape[1], float(fixed_low), dtype=np.float32)
        high = np.full(shared.shape[1], float(fixed_high), dtype=np.float32)
        return low, high
    mask = data["moth_ids"] == moth_id
    moth_shared = shared[mask]
    if moth_shared.size == 0:
        return None, None
    low = np.percentile(moth_shared, low_pct, axis=0)
    high = np.percentile(moth_shared, high_pct, axis=0)
    if sweep_max is not None:
        # Keep sweeps inside the observed latent distribution unless the user
        # explicitly caps an unusually large empirical upper bound.
        high = np.minimum(high, float(sweep_max))
    if np.allclose(low, high):
        scale = np.maximum(np.abs(low) * 0.1, 0.25)
        low = low - scale
        high = high + scale
    return low, high


def sweep_shared_latent(model, ref_record, low_vec, high_vec, steps):
    shared_dim = model.shared_latent_dim
    ref_spike_latent = torch.tensor(ref_record["spike_latent"], device=model.device, dtype=torch.float32).unsqueeze(0)
    ref_spike_latent_seq = None
    if ref_record.get("spike_latent_seq") is not None:
        ref_spike_latent_seq = torch.tensor(ref_record["spike_latent_seq"], device=model.device, dtype=torch.float32).unsqueeze(0)
    moth_id = torch.tensor([ref_record["moth_id"]], device=model.device, dtype=torch.long)
    mass = torch.tensor([[ref_record["mass"]]], device=model.device, dtype=torch.float32)
    cond = model._build_condition(mass, moth_id)
    sweep_outputs = []

    alphas = np.linspace(0.0, 1.0, steps)
    for alpha in alphas:
        shared_vec = low_vec * (1.0 - alpha) + high_vec * alpha
        shared_tensor = torch.tensor(shared_vec, device=model.device, dtype=torch.float32).unsqueeze(0)
        if getattr(model, "use_no_pooling", False) and ref_spike_latent_seq is not None:
            spike_latent_seq = ref_spike_latent_seq.clone()
            spike_latent_seq[:, :, :shared_dim] = shared_tensor.unsqueeze(1).expand(-1, spike_latent_seq.shape[1], -1)
            pooled_latent = spike_latent_seq.mean(dim=1)
            spike_recon = model.spike_decoder(spike_latent_seq).detach().cpu().numpy()[0]
        else:
            spike_latent = ref_spike_latent.clone()
            spike_latent[:, :shared_dim] = shared_tensor
            pooled_latent = spike_latent
            spike_recon = model.spike_decoder(spike_latent, cond).detach().cpu().numpy()[0]

        flower_mean, flower_recon = model._decode_flower(pooled_latent[:, :shared_dim])
        ft_pred = model._predict_ft(pooled_latent[:, : model.shared_latent_dim], mass, moth_id)

        sweep_outputs.append(
            {
                "alpha": float(alpha),
                "shared_vec": shared_vec.astype(np.float32),
                "spike_recon": spike_recon,
                "flower_recon_mean": flower_mean.detach().cpu().numpy()[0],
                "flower_recon": flower_recon.detach().cpu().numpy()[0],
                "ft_pred": ft_pred.detach().cpu().numpy()[0],
            }
        )

    return sweep_outputs


def render_sweep_frame(
    ref_records,
    sweep_outputs_by_ref,
    step_idx: int,
    shared_dim: int,
    flower_labels,
    spike_vmin: float,
    spike_vmax: float,
    flower_ylim,
    ft_ylim,
):
    n_rows = len(ref_records)
    fig, axes = plt.subplots(n_rows, 3, figsize=(14, 4 * n_rows), dpi=180)
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, ref_record in enumerate(ref_records):
        out = sweep_outputs_by_ref[row][step_idx]
        row_title = f"moth {ref_record['moth_id']} {ref_record['split']} | alpha={out['alpha']:.2f}"

        spike_ax = axes[row, 0]
        flower_ax = axes[row, 1]
        ft_ax = axes[row, 2]

        spike_ax.imshow(out["spike_recon"].T, aspect="auto", interpolation="none", vmin=spike_vmin, vmax=spike_vmax)
        spike_ax.set_title(f"{row_title}\nspike recon")
        spike_ax.set_xlabel("Timestep")
        spike_ax.set_ylabel("Muscle")

        x = np.arange(len(flower_labels))
        flower_vals = out["flower_recon_mean"]
        colors = [f"C{i}" for i in range(len(flower_labels))]
        flower_ax.bar(x, flower_vals, color=colors)
        flower_ax.set_xticks(x)
        flower_ax.set_xticklabels(flower_labels)
        flower_ax.set_ylabel("Value")
        flower_ax.set_ylim(*flower_ylim)
        flower_ax.set_title(f"{row_title}\nflower mean")

        x = np.arange(len(FT_LABELS))
        ft_vals = out["ft_pred"]
        ft_ax.bar(x, ft_vals, color=["tab:green", "tab:red", "tab:purple"])
        ft_ax.set_xticks(x)
        ft_ax.set_xticklabels(FT_LABELS)
        ft_ax.set_ylabel("Value")
        ft_ax.set_ylim(*ft_ylim)
        ft_ax.set_title(f"{row_title}\nFT prediction")

    fig.suptitle(
        f"Shared latent sweep step {step_idx + 1} / {len(sweep_outputs_by_ref[0])}",
        fontsize=15,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def save_moth_sweep_gif(
    output_dir: Path,
    moth_id: int,
    ref_records,
    sweep_outputs_by_ref,
    flower_labels,
):
    moth_dir = output_dir / "shared_latent_sweep_by_moth"
    moth_dir.mkdir(parents=True, exist_ok=True)

    all_spikes = []
    all_flower = []
    all_ft = []
    for outputs in sweep_outputs_by_ref:
        for out in outputs:
            all_spikes.append(out["spike_recon"])
            all_flower.append(out["flower_recon_mean"])
            all_ft.append(out["ft_pred"])

    spike_stack = np.concatenate([x.reshape(-1) for x in all_spikes])
    spike_vmin = float(np.percentile(spike_stack, 1.0))
    spike_vmax = float(np.percentile(spike_stack, 99.0))
    flower_stack = np.concatenate([x.reshape(-1) for x in all_flower])
    ft_stack = np.concatenate([x.reshape(-1) for x in all_ft])
    ft_lo = float(np.min(ft_stack))
    ft_hi = float(np.max(ft_stack))
    ft_pad = 0.1 * max(1.0, ft_hi - ft_lo)
    flower_lo = float(np.min(flower_stack))
    flower_hi = float(np.max(flower_stack))
    flower_span = flower_hi - flower_lo
    flower_pad = 0.15 * max(1.0, flower_span)
    if len(flower_labels) == 1 and flower_labels[0] == "Position":
        center = 0.5 * (flower_lo + flower_hi)
        half_range = max(1.0, 0.5 * flower_span + flower_pad)
        flower_ylim = (center - half_range, center + half_range)
    else:
        flower_ylim = (flower_lo - flower_pad, flower_hi + flower_pad)
    ft_ylim = (ft_lo - ft_pad, ft_hi + ft_pad)

    frames = []
    n_steps = len(sweep_outputs_by_ref[0])
    for step_idx in range(n_steps):
        frame = render_sweep_frame(
            ref_records=ref_records,
            sweep_outputs_by_ref=sweep_outputs_by_ref,
            step_idx=step_idx,
            shared_dim=len(sweep_outputs_by_ref[0][step_idx]["shared_vec"]),
            flower_labels=flower_labels,
            spike_vmin=spike_vmin,
            spike_vmax=spike_vmax,
            flower_ylim=flower_ylim,
            ft_ylim=ft_ylim,
        )
        frames.append(frame)

    gif_path = moth_dir / f"moth_{moth_id}_shared_latent_sweep.gif"
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=250,
        loop=0,
        optimize=False,
    )
    return gif_path


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

    low_data = gather_loader_outputs(model, low_test_loader, mass_label=0.0)
    high_data = gather_loader_outputs(model, high_test_loader, mass_label=1.0)
    combined = combine_outputs(low_data, high_data)
    combined["shared_latent_dim"] = int(model.shared_latent_dim)

    flower_recon_mode = "mean"
    flower_prediction_target = getattr(model, "flower_prediction_target", getattr(args, "flower_prediction_target", "position"))
    flower_labels = get_flower_labels(flower_prediction_target)
    output_dir = Path(args.save_path) / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep_steps = int(getattr(args, "latent_sweep_steps", 11))
    sweep_low = float(getattr(args, "latent_sweep_percentile_low", 5.0))
    sweep_high = float(getattr(args, "latent_sweep_percentile_high", 95.0))
    sweep_max = getattr(args, "latent_sweep_max", None)
    sweep_max = None if sweep_max is None else float(sweep_max)
    sweep_value_low = getattr(args, "latent_sweep_value_low", None)
    sweep_value_high = getattr(args, "latent_sweep_value_high", None)
    sweep_value_low = None if sweep_value_low is None else float(sweep_value_low)
    sweep_value_high = None if sweep_value_high is None else float(sweep_value_high)

    sweep_summary = {
        "shared_latent_dim": int(model.shared_latent_dim),
        "spike_latent_dim": int(model.spike_latent_dim),
        "steps": sweep_steps,
        "percentile_low": sweep_low,
        "percentile_high": sweep_high,
        "absolute_max": sweep_max,
        "fixed_value_low": sweep_value_low,
        "fixed_value_high": sweep_value_high,
        "flower_recon_mode": flower_recon_mode,
        "flower_prediction_target": flower_prediction_target,
        "flower_labels": flower_labels,
        "per_moth": {},
    }

    unique_moths = sorted(np.unique(combined["moth_ids"]).astype(int).tolist())
    for moth_id in unique_moths:
        ref_records = pick_reference_indices(combined, moth_id)
        if not ref_records:
            continue
        low_vec, high_vec = build_sweep_bounds(
            combined,
            moth_id,
            sweep_low,
            sweep_high,
            sweep_max,
            sweep_value_low,
            sweep_value_high,
        )
        if low_vec is None or high_vec is None:
            continue

        sweep_outputs_by_ref = []
        for ref_record in ref_records:
            sweep_outputs = sweep_shared_latent(
                model,
                ref_record,
                low_vec,
                high_vec,
                sweep_steps,
            )
            sweep_outputs_by_ref.append(sweep_outputs)

        gif_path = save_moth_sweep_gif(
            output_dir,
            moth_id,
            ref_records,
            sweep_outputs_by_ref,
            flower_labels,
        )
        sweep_summary["per_moth"][str(moth_id)] = {
            "references": ref_records,
            "shared_low": low_vec.astype(np.float32),
            "shared_high": high_vec.astype(np.float32),
            "gif_path": str(gif_path),
        }

    with open(output_dir / "shared_latent_sweep.json", "w") as f:
        json.dump(to_serializable(sweep_summary), f, indent=2)

    print(json.dumps(to_serializable(sweep_summary), indent=2))
    print(f"Saved shared latent sweep analysis to {output_dir}")


if __name__ == "__main__":
    main()
