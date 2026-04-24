import os
import torch
import numpy as np
import sys
import json
import argparse

def get_parser():
    """Get parser object."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    # global arg
    parser.add_argument("--seed",dest="seed",default=0,type=int,help="seed",required=False)
    parser.add_argument("--data_seed",dest="data_seed",default=0,type=int,help="seed",required=False)
    parser.add_argument("--device",dest="device",default="cuda:0",help="experiment specified device",required=False,)
    
    # dataset arg
    parser.add_argument("--filepath_high",dest="filepath_high",default='/hpc/group/tarokhlab/hy190/data/fatmoth_filtered/fatmoth_HIGH.pickle',help="dataset path high mass moth",required=False,)
    parser.add_argument("--filepath_low",dest="filepath_low",default='/hpc/group/tarokhlab/hy190/data/fatmoth_filtered/fatmoth_LOW.pickle',help="dataset path of low mass moth",required=False,)
    parser.add_argument("--batch_size",dest="batch_size",default=64, type=int, help="save path",required=False)
    parser.add_argument("--convolution",dest="convolution",default='gaussian', help="convolution on muscle",required=False)
    parser.add_argument("--split",dest="split",default=0.8, type=float, help="train-test split",required=False)
    parser.add_argument(
        "--ft_norm_mode",
        dest="ft_norm_mode",
        default="global",
        choices=["global", "per_moth", "none"],
        help="FT normalization mode",
        required=False,
    )
    parser.add_argument(
        "--ft_predictor_mode",
        dest="ft_predictor_mode",
        default="per_moth_mass",
        choices=["per_moth_mass"],
        help="FT predictor parameterization",
        required=False,
    )
    parser.add_argument(
        "--flower_recon_mode",
        dest="flower_recon_mode",
        default="mean",
        choices=["mean"],
        help="flower reconstruction target type",
        required=False,
    )
    parser.add_argument(
        "--flower_decoder_latent_source",
        dest="flower_decoder_latent_source",
        default="spike_shared",
        choices=["spike_shared"],
        help="latent source used by the flower decoder",
        required=False,
    )
    parser.add_argument(
        "--model_architecture",
        dest="model_architecture",
        default="variational",
        choices=["variational", "variational_no_pooling"],
        help="model architecture variant",
        required=False,
    )
    parser.add_argument(
        "--latent_sweep_steps",
        dest="latent_sweep_steps",
        default=11,
        type=int,
        help="number of interpolation steps for shared latent sweep analysis",
        required=False,
    )
    parser.add_argument(
        "--latent_sweep_percentile_low",
        dest="latent_sweep_percentile_low",
        default=1.0,
        type=float,
        help="lower percentile for shared latent sweep bounds",
        required=False,
    )
    parser.add_argument(
        "--latent_sweep_percentile_high",
        dest="latent_sweep_percentile_high",
        default=99.0,
        type=float,
        help="upper percentile for shared latent sweep bounds",
        required=False,
    )
    parser.add_argument(
        "--latent_sweep_max",
        dest="latent_sweep_max",
        default=None,
        type=float,
        help="optional upper cap for shared latent sweep bounds; by default use observed percentiles",
        required=False,
    )
    
    # model arg
    parser.add_argument("--d_model",dest="d_model",default=64,type=int,required=False)
    parser.add_argument("--d_latent",dest="d_latent",default=1,type=int,required=False)
    parser.add_argument(
        "--spike_latent_dim",
        dest="spike_latent_dim",
        default=3,
        type=int,
        help="latent dimension for the spike autoencoder",
        required=False,
    )
    parser.add_argument("--d_latent_share",dest="d_latent_share",default=2,type=int,required=False)
    parser.add_argument("--d_latent_treat",dest="d_latent_treat",default=3,type=int,required=False)
    parser.add_argument("--dropout",dest="dropout",default=0.1, type=float, help="dropout p",required=False)
    parser.add_argument("--n_heads",dest="n_heads",default=4,type=int,help="number of heads",required=False,)
    parser.add_argument("--d_ff",dest="d_ff",default=64, type=int, help="transformer mlp dim",required=False)
    parser.add_argument("--e_layers",dest="e_layers",default=2, type=int, help="encoder layers",required=False)
    
    # training arg
    parser.add_argument("--optimizer",dest="optimizer",default='Adam', required=False)
    parser.add_argument("--lr",dest="lr",default=0.001, type=float, required=False)
    parser.add_argument("--spike_decoder_lr",dest="spike_decoder_lr",default=0.001, type=float, required=False)
    parser.add_argument("--save_path",dest="save_path",default='./exp_all_variational', help="save path",required=False)
    parser.add_argument("--n_epochs",dest="n_epochs",default=200, type=int, help="training epoch",required=False)
    parser.add_argument("--eval_every_n_epoch",dest="eval_every_n_epoch",default=5, type=int, help="evaluate every n epoch",required=False)
    return parser

def make_directory(savepath):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    
def set_seeds(seed):   
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic=True
    
    
def load_config(save_path="./exp_all"):
    with open(f"{save_path}/config.json", "r") as f:
        config = json.load(f)
    args = argparse.Namespace(**config)
    args.save_path = save_path
    return args
