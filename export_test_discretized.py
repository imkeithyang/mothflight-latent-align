import argparse
import json
from pathlib import Path

import numpy as np
import torch

import discretization
import MFT_dataset
import model_trainer_variational as model_trainer
import utils


DEFAULT_RUN_PATH = "/hpc/home/hy190/MURI/projects/mothtethered/mothflight_latent_align/run_500_repft_shared02_0"


def load_model(run_path: Path, device: str):
    args = utils.load_config(str(run_path))
    args.device = device
    model, _ = model_trainer.get_model_and_optimizer(args)
    model = model_trainer.load_checkpoint(str(run_path), model, device)
    model.to(device)
    model.eval()
    return args, model


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def flatten_batches(batch_values):
    arrays = [np.asarray(value) for value in batch_values]
    if len(arrays) == 0:
        return np.array([])
    return np.concatenate(arrays, axis=0)


def flatten_outputs(outputs):
    combined = {}
    for key, values in outputs.items():
        if key == "split_names":
            combined[key] = list(values)
        else:
            combined[key] = flatten_batches(values)
    return combined


def gather_test_outputs(model, loader, split_label: float):
    output = {
        "spike_input_discrete": [],
        "spike_input_model": [],
        "spike_recon": [],
        "spike_recon_discrete": [],
        "spike_counts_true": [],
        "spike_counts_pred": [],
        "ft_true": [],
        "ft_pred": [],
        "flower_true_mean": [],
        "flower_pred_mean": [],
        "moth_ids": [],
        "mass_values": [],
        "mass_labels": [],
        "split_names": [],
    }

    with torch.no_grad():
        for batch in loader:
            covariates = batch[0]
            ft_mean = batch[2]
            spikes_model = batch[3]
            spike_counts = batch[4]
            spike_input_discrete = batch[5]
            moth_ids = batch[6]
            mass = batch[7]

            outputs = model.forward(covariates, spikes_model, mass, moth_ids)

            spike_recon = to_numpy(outputs["spike_recon"])
            spike_counts_np = to_numpy(spike_counts).astype(int)
            spike_recon_discrete = discretization.peak_sampling(spike_recon, spike_counts_np)

            flower_true_mean = to_numpy(covariates.mean(dim=1)[..., :1])
            flower_pred_mean = to_numpy(outputs["flower_recon_mean"])

            output["spike_input_discrete"].append(to_numpy(spike_input_discrete))
            output["spike_input_model"].append(to_numpy(spikes_model))
            output["spike_recon"].append(spike_recon)
            output["spike_recon_discrete"].append(spike_recon_discrete)
            output["spike_counts_true"].append(spike_counts_np)
            output["spike_counts_pred"].append(to_numpy(outputs["spike_counts_pred"]))
            output["ft_true"].append(to_numpy(ft_mean))
            output["ft_pred"].append(to_numpy(outputs["ft_pred"]))
            output["flower_true_mean"].append(flower_true_mean)
            output["flower_pred_mean"].append(flower_pred_mean)
            output["moth_ids"].append(to_numpy(moth_ids).reshape(-1, 1))
            output["mass_values"].append(to_numpy(mass).reshape(-1, 1))
            output["mass_labels"].append(np.full((covariates.shape[0], 1), split_label, dtype=np.float32))
            output["split_names"].extend(["high" if split_label > 0.5 else "low"] * covariates.shape[0])

    return output


def concat_outputs(outputs):
    combined = {}
    for key, values in outputs.items():
        if key == "split_names":
            combined[key] = list(values)
        else:
            combined[key] = np.concatenate(values, axis=0)
    return combined


def main():
    parser = argparse.ArgumentParser(description="Export discrete test-set outputs for a trained model.")
    parser.add_argument("--run_path", type=str, default=DEFAULT_RUN_PATH, help="Path to the trained run directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device used for evaluation")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional output directory. Defaults to <run_path>/analysis/discretized_test_set",
    )
    args = parser.parse_args()

    run_path = Path(args.run_path)
    output_dir = Path(args.output_dir) if args.output_dir is not None else run_path / "analysis" / "discretized_test_set"
    utils.make_directory(str(output_dir))

    run_args, model = load_model(run_path, args.device)

    high_train_loader, high_test_loader, low_train_loader, low_test_loader = MFT_dataset.get_MFT(
        filepath_high=run_args.filepath_high,
        filepath_low=run_args.filepath_low,
        batch_size=run_args.batch_size,
        convolution=run_args.convolution,
        device=torch.device(args.device),
        split=run_args.split,
        normalize_ft=True,
        evaluate=True,
        perturb=False,
        data_seed=run_args.data_seed,
        ft_norm_mode=run_args.ft_norm_mode,
    )
    del high_train_loader, low_train_loader

    with torch.no_grad():
        low_outputs = gather_test_outputs(model, low_test_loader, split_label=0.0)
        high_outputs = gather_test_outputs(model, high_test_loader, split_label=1.0)

    low_outputs = flatten_outputs(low_outputs)
    high_outputs = flatten_outputs(high_outputs)

    combined = {}
    for key in low_outputs.keys():
        if key == "split_names":
            combined[key] = low_outputs[key] + high_outputs[key]
        else:
            combined[key] = np.concatenate([low_outputs[key], high_outputs[key]], axis=0)

    np.savez_compressed(
        output_dir / "test_discrete_predictions.npz",
        spike_input_discrete=combined["spike_input_discrete"],
        spike_input_model=combined["spike_input_model"],
        spike_recon=combined["spike_recon"],
        spike_recon_discrete=combined["spike_recon_discrete"],
        spike_counts_true=combined["spike_counts_true"],
        spike_counts_pred=combined["spike_counts_pred"],
        ft_true=combined["ft_true"],
        ft_pred=combined["ft_pred"],
        flower_true_mean=combined["flower_true_mean"],
        flower_pred_mean=combined["flower_pred_mean"],
        moth_ids=combined["moth_ids"],
        mass_labels=combined["mass_labels"],
        mass_values=combined["mass_values"],
        split_names=np.asarray(combined["split_names"], dtype="U5"),
    )

    metadata = {
        "run_path": str(run_path),
        "device": args.device,
        "source": "test split only",
        "recon_discretization": "discretization.peak_sampling(spike_recon, true_spike_counts)",
        "outputs": [
            "spike_input_discrete",
            "spike_input_model",
            "spike_recon",
            "spike_recon_discrete",
            "spike_counts_true",
            "spike_counts_pred",
            "ft_true",
            "ft_pred",
            "flower_true_mean",
            "flower_pred_mean",
            "moth_ids",
            "mass_labels",
            "mass_values",
            "split_names",
        ],
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved test discretization outputs to {output_dir}")


if __name__ == "__main__":
    main()
