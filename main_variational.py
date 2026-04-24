import subprocess
import sys
import os
import pickle
import gc

import model_trainer_variational as model_trainer
import MFT_dataset
import utils
import json
import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


args = utils.get_parser().parse_args()
if isinstance(args.device, str) and args.device.startswith("cuda") and not torch.cuda.is_available():
    args.device = "cpu"
args.save_path += str(args.seed)
args.save_path = os.path.abspath(args.save_path)
utils.make_directory(args.save_path)

high_train_loader, high_test_loader, low_train_loader, low_test_loader = MFT_dataset.get_MFT(filepath_high=args.filepath_high,
                                                                                             filepath_low=args.filepath_low,
                                                                                             batch_size=args.batch_size,
                                                                                             convolution=args.convolution,
                                                                                             device=args.device,
                                                                                             split=args.split,
                                                                                             data_seed=args.data_seed,
                                                                                             ft_norm_mode=args.ft_norm_mode)

covariate_data, _, FT_data, muscle_data, spikecount_data, _, _= next(iter(high_train_loader))
args.covariate_dims = [covariate_data.shape[-2], covariate_data.shape[-1]]
args.ft_dims = [FT_data.shape[-1]]
args.spike_dims = [muscle_data.shape[-2], muscle_data.shape[-1]]
args.moth_number = 10
print(args.covariate_dims,args.ft_dims,args.spike_dims)

with open(f'{args.save_path}/config.json', 'w') as f:
    print(f'config saved in {args.save_path}')
    json.dump(vars(args), f, indent=4)
    
utils.set_seeds(int(args.seed))


def _plot_loss_curves(log_dict, save_path, n_epochs, eval_every_n_epoch):
    train_epochs = np.arange(1, n_epochs + 1)
    eval_epochs = [epoch for epoch in range(1, n_epochs + 1) if epoch % eval_every_n_epoch == 0 or epoch == n_epochs]

    series = [
        ("Total", "list_epoch_loss", "eval_list_epoch_loss"),
        ("Align", "list_orth_epoch_loss", "eval_list_orth_epoch_loss"),
        ("FT", "list_ft_epoch_loss", "eval_list_ft_epoch_loss"),
        ("Recons", "list_recons_epoch_loss", "eval_list_recons_epoch_loss"),
        ("Spikes", "list_spikes_epoch_loss", "eval_list_spikes_epoch_loss"),
        ("Spike count", "list_spike_counts_epoch_loss", "eval_list_spike_counts_epoch_loss"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=False)
    axes = axes.flatten()
    for ax, (title, train_key, eval_key) in zip(axes, series):
        ax.plot(train_epochs, log_dict[train_key], label="train", linewidth=2)
        ax.plot(eval_epochs, log_dict[eval_key], label="val", linewidth=2, marker="o")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)

    fig.suptitle("Training / Validation Loss Curves", fontsize=16)
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, "training_validation_loss_curve.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def _run_auto_eval(args):
    eval_cmd = [
        sys.executable,
        "evaluate_latent_align.py",
        "--save_path",
        args.save_path,
        "--device",
        str(args.device),
        "--batch_size",
        str(args.batch_size),
        "--filepath_high",
        args.filepath_high,
        "--filepath_low",
        args.filepath_low,
        "--split",
        str(args.split),
        "--data_seed",
        str(args.data_seed),
        "--d_model",
        str(args.d_model),
        "--d_latent",
        str(args.d_latent),
        "--spike_latent_dim",
        str(getattr(args, "spike_latent_dim", 3)),
        "--d_latent_share",
        str(args.d_latent_share),
        "--d_latent_treat",
        str(args.d_latent_treat),
        "--dropout",
        str(args.dropout),
        "--n_heads",
        str(args.n_heads),
        "--d_ff",
        str(args.d_ff),
        "--e_layers",
        str(args.e_layers),
        "--optimizer",
        str(args.optimizer),
        "--lr",
        str(args.lr),
        "--convolution",
        str(args.convolution),
        "--ft_norm_mode",
        str(args.ft_norm_mode),
        "--ft_predictor_mode",
        str(getattr(args, "ft_predictor_mode", "per_moth_mass")),
        "--flower_recon_mode",
        str(getattr(args, "flower_recon_mode", "mean")),
        "--flower_decoder_latent_source",
        str(getattr(args, "flower_decoder_latent_source", "spike_shared")),
        "--model_architecture",
        str(getattr(args, "model_architecture", "variational")),
    ]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Running automatic evaluation: {' '.join(eval_cmd)}")
    subprocess.run(eval_cmd, check=True, cwd=script_dir)


def _run_shared_importance_eval(args):
    importance_cmd = [
        sys.executable,
        "evaluate_shared_latent_importance.py",
        "--save_path",
        args.save_path,
        "--device",
        "cpu",
        "--batch_size",
        str(args.batch_size),
        "--filepath_high",
        args.filepath_high,
        "--filepath_low",
        args.filepath_low,
        "--split",
        str(args.split),
        "--data_seed",
        str(args.data_seed),
        "--d_model",
        str(args.d_model),
        "--d_latent",
        str(args.d_latent),
        "--spike_latent_dim",
        str(getattr(args, "spike_latent_dim", 3)),
        "--d_latent_share",
        str(args.d_latent_share),
        "--d_latent_treat",
        str(args.d_latent_treat),
        "--dropout",
        str(args.dropout),
        "--n_heads",
        str(args.n_heads),
        "--d_ff",
        str(args.d_ff),
        "--e_layers",
        str(args.e_layers),
        "--optimizer",
        str(args.optimizer),
        "--lr",
        str(args.lr),
        "--convolution",
        str(args.convolution),
        "--ft_norm_mode",
        str(args.ft_norm_mode),
        "--ft_predictor_mode",
        str(getattr(args, "ft_predictor_mode", "per_moth_mass")),
        "--flower_recon_mode",
        str(getattr(args, "flower_recon_mode", "mean")),
        "--flower_decoder_latent_source",
        str(getattr(args, "flower_decoder_latent_source", "spike_shared")),
        "--model_architecture",
        str(getattr(args, "model_architecture", "variational")),
    ]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Running shared latent importance evaluation: {' '.join(importance_cmd)}")
    subprocess.run(importance_cmd, check=True, cwd=script_dir)


def _run_shared_latent_sweep_eval(args):
    sweep_cmd = [
        sys.executable,
        "evaluate_shared_latent_sweep.py",
        "--save_path",
        args.save_path,
        "--device",
        "cpu",
        "--batch_size",
        str(args.batch_size),
        "--filepath_high",
        args.filepath_high,
        "--filepath_low",
        args.filepath_low,
        "--split",
        str(args.split),
        "--data_seed",
        str(args.data_seed),
        "--d_model",
        str(args.d_model),
        "--d_latent",
        str(args.d_latent),
        "--spike_latent_dim",
        str(getattr(args, "spike_latent_dim", 3)),
        "--d_latent_share",
        str(args.d_latent_share),
        "--d_latent_treat",
        str(args.d_latent_treat),
        "--dropout",
        str(args.dropout),
        "--n_heads",
        str(args.n_heads),
        "--d_ff",
        str(args.d_ff),
        "--e_layers",
        str(args.e_layers),
        "--optimizer",
        str(args.optimizer),
        "--lr",
        str(args.lr),
        "--convolution",
        str(args.convolution),
        "--ft_norm_mode",
        str(args.ft_norm_mode),
        "--ft_predictor_mode",
        str(getattr(args, "ft_predictor_mode", "per_moth_mass")),
        "--flower_recon_mode",
        str(getattr(args, "flower_recon_mode", "mean")),
        "--flower_decoder_latent_source",
        str(getattr(args, "flower_decoder_latent_source", "spike_shared")),
        "--model_architecture",
        str(getattr(args, "model_architecture", "variational")),
        "--latent_sweep_steps",
        str(getattr(args, "latent_sweep_steps", 11)),
        "--latent_sweep_percentile_low",
        str(getattr(args, "latent_sweep_percentile_low", 5.0)),
        "--latent_sweep_percentile_high",
        str(getattr(args, "latent_sweep_percentile_high", 95.0)),
    ]
    latent_sweep_max = getattr(args, "latent_sweep_max", None)
    if latent_sweep_max is not None:
        sweep_cmd.extend(["--latent_sweep_max", str(latent_sweep_max)])
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Running shared latent sweep evaluation: {' '.join(sweep_cmd)}")
    subprocess.run(sweep_cmd, check=True, cwd=script_dir)

model, optimizer = model_trainer.get_model_and_optimizer(args)
log_dict = {}
log_dict, model = model_trainer.train(model, optimizer, 
                    args.n_epochs, 
                    high_train_loader, high_test_loader, low_train_loader, low_test_loader,
                    args.save_path, args.eval_every_n_epoch)

# model = model_trainer.load_checkpoint(args.save_path, model, args.device)
# model = model_trainer.train_cattn(model, optimizer,
#                     int(0.5 * args.n_epochs),
#                     high_train_loader, high_test_loader, low_train_loader, low_test_loader,

# final evaluation
best_model = model_trainer.load_checkpoint(args.save_path, model, args.device)
eval_loss, eval_orth_loss, eval_ft_loss, eval_recons_loss, eval_kld_loss, eval_spikes_loss, eval_spikes_count_loss = model_trainer.eval_step(best_model, high_test_loader, low_test_loader)

log_dict['final_eval_loss'] = eval_loss
log_dict['final_eval_orth_loss'] = eval_orth_loss
log_dict['final_eval_ft_loss'] = eval_ft_loss
log_dict['final_eval_recons_loss'] = eval_recons_loss
log_dict['final_eval_kld_loss'] = eval_kld_loss
log_dict['final_eval_spikes_loss'] = eval_spikes_loss
log_dict['final_eval_spikes_count_loss'] = eval_spikes_count_loss
# log_dict['attns'] = attns
# log_dict['attns_treat'] = attns_treat

_plot_loss_curves(log_dict, args.save_path, args.n_epochs, args.eval_every_n_epoch)

with open(args.save_path + '/loss_log.pkl', 'wb') as f:
    pickle.dump(log_dict, f)

del best_model
del model
gc.collect()
if torch.cuda.is_available() and str(args.device).startswith("cuda"):
    torch.cuda.empty_cache()

_run_auto_eval(args)
_run_shared_importance_eval(args)
_run_shared_latent_sweep_eval(args)
