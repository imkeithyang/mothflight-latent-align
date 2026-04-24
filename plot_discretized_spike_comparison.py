import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT = "/hpc/home/hy190/MURI/projects/mothtethered/mothflight_latent_align/run_500_repft_shared02_0/analysis/discretized_test_set/test_discrete_predictions.npz"
DEFAULT_OUTPUT_DIR = "/hpc/home/hy190/MURI/projects/mothtethered/mothflight_latent_align/run_500_repft_shared02_0/analysis/discretized_test_set"


def pick_examples(indices, max_examples=4):
    if len(indices) <= max_examples:
        return indices
    pick = [indices[0], indices[len(indices) // 3], indices[(2 * len(indices)) // 3], indices[-1]]
    return list(dict.fromkeys(pick))


def ensure_2d(spike_array):
    arr = np.asarray(spike_array)
    if arr.ndim == 3:
        arr = arr[0]
    return arr


def plot_per_moth(npz_path: Path, output_dir: Path):
    data = np.load(npz_path, allow_pickle=True)
    input_spikes = np.asarray(data["spike_input_discrete"])
    recon_spikes = np.asarray(data["spike_recon_discrete"])
    moth_ids = np.asarray(data["moth_ids"]).reshape(-1)
    mass_labels = np.asarray(data["mass_labels"]).reshape(-1)
    mass_values = np.asarray(data["mass_values"]).reshape(-1) if "mass_values" in data.files else mass_labels

    moth_out_dir = output_dir / "by_moth"
    moth_out_dir.mkdir(parents=True, exist_ok=True)

    for moth_id in sorted(np.unique(moth_ids)):
        moth_idx = np.where(moth_ids == moth_id)[0]
        if len(moth_idx) == 0:
            continue
        example_idx = pick_examples(list(moth_idx))
        fig, axes = plt.subplots(len(example_idx), 2, figsize=(10, 2.5 * len(example_idx)), dpi=200)
        if len(example_idx) == 1:
            axes = np.expand_dims(axes, axis=0)

        for row, idx in enumerate(example_idx):
            input_img = ensure_2d(input_spikes[idx])
            recon_img = ensure_2d(recon_spikes[idx])
            mass_label = float(mass_labels[idx])
            mass_value = float(mass_values[idx])

            axes[row, 0].imshow(input_img.T, aspect="auto", interpolation="none", cmap="Greys")
            axes[row, 0].set_title(f"Input discrete | moth {int(moth_id)} | mass {mass_value:.3f} | split {int(mass_label)}")
            axes[row, 0].set_ylabel(f"Example {idx}")
            axes[row, 0].set_xlabel("Time")

            axes[row, 1].imshow(recon_img.T, aspect="auto", interpolation="none", cmap="Greys")
            axes[row, 1].set_title(f"Recon discretized | moth {int(moth_id)}")
            axes[row, 1].set_xlabel("Time")

            for col in range(2):
                axes[row, col].set_yticks(range(10))
                axes[row, col].set_ylabel("Muscle")

        plt.tight_layout()
        out_path = moth_out_dir / f"moth_{int(moth_id)}_discretized_spike_input_vs_recon.png"
        plt.savefig(out_path)
        plt.close(fig)
        print(f"Saved comparison figure to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot discretized spike input vs recon side by side.")
    parser.add_argument("--npz_path", type=str, default=DEFAULT_INPUT, help="Path to the exported npz file")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory for the figure")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_per_moth(Path(args.npz_path), output_dir)


if __name__ == "__main__":
    main()
