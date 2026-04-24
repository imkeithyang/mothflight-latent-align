import torch
import torch.optim as optim
import numpy as np
from tqdm.auto import tqdm

import models_variational as pooled_models
import models_variational_no_pooling as no_pooling_models


def _get_model_module(args):
    architecture = getattr(args, "model_architecture", "variational")
    if architecture == "variational":
        return pooled_models
    if architecture == "variational_no_pooling":
        return no_pooling_models
    raise ValueError(f"Unknown model_architecture: {architecture}")


def get_model_and_optimizer(args):
    models = _get_model_module(args)
    model = models.Transformer_TAR_net(covariate_dims=args.covariate_dims, 
          ft_dims=args.ft_dims, 
          spike_dims=args.spike_dims, 
          d_model=args.d_model, 
          d_latent=args.d_latent,
          spike_latent_dim=getattr(args, "spike_latent_dim", 3),
          d_latent_treat=args.d_latent_treat,
          d_latent_share=args.d_latent_share,
          num_moths=args.moth_number,
          dropout=args.dropout, 
          n_heads=args.n_heads, 
          d_ff=args.d_ff, 
          e_layers=args.e_layers, 
          device=args.device,
          ft_predictor_mode=getattr(args, "ft_predictor_mode", "per_moth_mass"),
          flower_recon_mode=getattr(args, "flower_recon_mode", "mean"),
          flower_decoder_latent_source=getattr(args, "flower_decoder_latent_source", "spike_shared"))

    architecture = getattr(args, "model_architecture", "variational")
    if architecture == "variational_no_pooling":
        spike_decoder_params = list(model.spike_decoder.parameters())
        spike_decoder_param_ids = {id(param) for param in spike_decoder_params}
        base_params = [param for param in model.parameters() if id(param) not in spike_decoder_param_ids]
        optimizer = getattr(optim, args.optimizer)(
            [
                {"params": base_params, "lr": args.lr},
                {"params": spike_decoder_params, "lr": getattr(args, "spike_decoder_lr", 0.001)},
            ],
            eps=1e-5,
        )
    else:
        optimizer = getattr(optim, args.optimizer)([{'params': model.parameters(), 'lr': args.lr}],eps=1e-5)
    return model, optimizer

def save_checkpoint(epoch, model, optimizer, loss, save_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        }

    torch.save(checkpoint, save_path + '/checkpoint.pth')
    
def load_checkpoint(save_path, model, device):
    checkpoint = torch.load(save_path + '/checkpoint.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def train(model, optimizer, n_epochs, 
          high_train_loader, high_test_loader, 
          low_train_loader, low_test_loader, 
          save_path, eval_every_n_epoch):
    
    pbar = tqdm(total=n_epochs)
    
    curr_min_eval_loss = float('inf')
    list_epoch_loss, list_orth_epoch_loss, list_ft_epoch_loss, list_recons_epoch_loss, list_kld_epoch_loss, list_spikes_epoch_loss, list_spike_counts_epoch_loss = [],[],[],[],[],[],[]
    eval_list_epoch_loss, eval_list_orth_epoch_loss, eval_list_ft_epoch_loss, eval_list_recons_epoch_loss, eval_list_kld_epoch_loss,eval_list_spikes_epoch_loss, eval_list_spike_counts_epoch_loss = [],[],[],[],[],[],[]
    
    for epoch in range(n_epochs):
        epoch_loss, orth_epoch_loss, ft_epoch_loss, recons_epoch_loss, kld_epoch_loss, spikes_epoch_loss, spike_counts_epoch_loss = train_step(model, optimizer, high_train_loader, low_train_loader)
        pbar.set_description("Epoch: {} | loss: {:.2f} | orth loss: {:.5f} | FT loss: {:.3f}| recons loss: {:.2f} | kld loss: {:.3f} | Spike loss: {:.3f} | Spike count loss: {:.3f}".format(
                epoch+1, 
                np.round(epoch_loss, 2),
                np.round(orth_epoch_loss, 2),
                np.round(ft_epoch_loss, 3),
                np.round(recons_epoch_loss, 3),
                np.round(kld_epoch_loss, 3),
                np.round(spikes_epoch_loss, 3),
                np.round(spike_counts_epoch_loss, 3),
            ))
        list_epoch_loss.append(epoch_loss)
        list_orth_epoch_loss.append(orth_epoch_loss)
        list_ft_epoch_loss.append(ft_epoch_loss)
        list_recons_epoch_loss.append(recons_epoch_loss)
        list_kld_epoch_loss.append(kld_epoch_loss)
        list_spikes_epoch_loss.append(spikes_epoch_loss)
        list_spike_counts_epoch_loss.append(spike_counts_epoch_loss)
        
        if epoch == n_epochs - 1 or (epoch + 1) % eval_every_n_epoch == 0:
            eval_epoch_loss, eval_orth_epoch_loss, eval_ft_epoch_loss, eval_recons_epoch_loss, eval_kld_epoch_loss, eval_spikes_epoch_loss, eval_spike_counts_epoch_loss = eval_step(model, high_test_loader, low_test_loader)
            
            eval_list_epoch_loss.append(eval_epoch_loss)
            eval_list_orth_epoch_loss.append(eval_orth_epoch_loss)
            eval_list_ft_epoch_loss.append(eval_ft_epoch_loss)
            eval_list_recons_epoch_loss.append(eval_recons_epoch_loss)
            eval_list_kld_epoch_loss.append(eval_kld_epoch_loss)
            eval_list_spikes_epoch_loss.append(eval_spikes_epoch_loss)
            eval_list_spike_counts_epoch_loss.append(eval_spike_counts_epoch_loss)

            print(f"Eval loss: {eval_epoch_loss:.2f} | Eval orth loss: {eval_orth_epoch_loss:.5f} | Eval FT loss: {eval_ft_epoch_loss:.3f} | Eval recons loss: {eval_recons_epoch_loss:.2f} | Eval kld loss: {eval_kld_epoch_loss:.3f} | Eval Spike loss: {eval_spikes_epoch_loss:.3f} | Eval Spike count loss: {eval_spike_counts_epoch_loss:.3f}")
            if eval_epoch_loss < curr_min_eval_loss:
                curr_min_eval_loss = eval_epoch_loss
                save_checkpoint(epoch, model, optimizer, eval_epoch_loss, save_path)
                
        pbar.update(1)
    pbar.close()
    
    log_dict = {"list_epoch_loss":list_epoch_loss, 
                   "list_orth_epoch_loss":list_orth_epoch_loss, 
                   "list_ft_epoch_loss":list_ft_epoch_loss, 
                   "list_recons_epoch_loss":list_recons_epoch_loss, 
                   "list_kld_epoch_loss":list_kld_epoch_loss, 
                   "list_spikes_epoch_loss":list_spikes_epoch_loss,
                   "list_spike_counts_epoch_loss":list_spike_counts_epoch_loss,
                   "eval_list_epoch_loss":eval_list_epoch_loss, 
                   "eval_list_orth_epoch_loss":eval_list_orth_epoch_loss, 
                   "eval_list_ft_epoch_loss":eval_list_ft_epoch_loss, 
                    "eval_list_recons_epoch_loss":eval_list_recons_epoch_loss, 
                   "eval_list_kld_epoch_loss":eval_list_kld_epoch_loss, 
                   "eval_list_spikes_epoch_loss":eval_list_spikes_epoch_loss,
                   "eval_list_spike_counts_epoch_loss":eval_list_spike_counts_epoch_loss,
                   }
    
    return log_dict, model
        
def train_step(model, optimizer, high_loader, low_loader):
    epoch_loss = 0
    orth_epoch_loss = 0
    ft_epoch_loss = 0
    recons_epoch_loss = 0
    kld_epoch_loss = 0
    spikes_epoch_loss = 0
    spike_counts_epoch_loss = 0
    cattn_epoch_loss = 0

    model.train()
    for batch_idx, (batch_high, batch_low) in enumerate(zip(high_loader, low_loader)):
        optimizer.zero_grad()
        orth_loss, ft_loss, recons_loss, kld_loss, spikes_loss, spike_counts_loss = model.train_step(batch_low, batch_high)
        loss_total = orth_loss + recons_loss + 0.01 * kld_loss + ft_loss + spikes_loss + 0.1 * spike_counts_loss 
        loss_total.backward()
        optimizer.step()
        
        epoch_loss += loss_total.item()
        orth_epoch_loss += orth_loss.item()
        ft_epoch_loss += ft_loss.item()
        recons_epoch_loss += recons_loss.item()
        kld_epoch_loss += kld_loss.item()
        spikes_epoch_loss += spikes_loss.item()
        spike_counts_epoch_loss += spike_counts_loss.item()
        # cattn_epoch_loss += cross_attn_loss.item()
        
    epoch_loss /= len(high_loader)
    orth_epoch_loss /= len(high_loader)
    ft_epoch_loss /= len(high_loader)
    recons_epoch_loss /= len(high_loader)
    kld_epoch_loss /= len(high_loader)
    spikes_epoch_loss /= len(high_loader)
    spike_counts_epoch_loss /= len(high_loader)
    # cattn_epoch_loss /= len(high_loader)
        
    return epoch_loss, orth_epoch_loss, ft_epoch_loss, recons_epoch_loss, kld_epoch_loss, spikes_epoch_loss, spike_counts_epoch_loss
        
    
def eval_step(model, high_loader, low_loader):   
    eval_epoch_loss = 0
    eval_orth_epoch_loss = 0
    eval_ft_epoch_loss = 0
    eval_recons_epoch_loss = 0
    eval_kld_epoch_loss = 0
    eval_spikes_epoch_loss = 0
    eval_spike_counts_epoch_loss = 0
    eval_cattn_epoch_loss = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, (batch_high, batch_low) in enumerate(zip(high_loader, low_loader)):
            orth_loss, ft_loss, recons_loss, kld_loss, spikes_loss, spike_counts_loss = model.train_step(batch_low, batch_high)
            loss_total = orth_loss +  recons_loss + 0.01 * kld_loss + ft_loss + spikes_loss + 0.1 * spike_counts_loss 
            
            eval_epoch_loss += loss_total.item()
            eval_orth_epoch_loss += orth_loss.item()
            eval_ft_epoch_loss += ft_loss.item()
            eval_recons_epoch_loss += recons_loss.item()
            eval_kld_epoch_loss += kld_loss.item()
            eval_spikes_epoch_loss += spikes_loss.item()
            eval_spike_counts_epoch_loss += spike_counts_loss.item()
            # eval_cattn_epoch_loss += cross_attn_loss.item()
            
        eval_epoch_loss /= len(high_loader)
        eval_orth_epoch_loss /=  len(high_loader)
        eval_ft_epoch_loss /= len(high_loader)
        eval_recons_epoch_loss /=  len(high_loader)
        eval_kld_epoch_loss /= len(high_loader)
        eval_spikes_epoch_loss /=  len(high_loader)
        eval_spike_counts_epoch_loss /= len(high_loader)
        # eval_cattn_epoch_loss /= len(high_loader)
            
    return eval_epoch_loss, eval_orth_epoch_loss, eval_ft_epoch_loss, eval_recons_epoch_loss, eval_kld_epoch_loss, eval_spikes_epoch_loss, eval_spike_counts_epoch_loss
            
