
import model_trainer_variational as model_trainer
import utils
import MFT_dataset
import matplotlib.pyplot as plt
import torch
import numpy as np

muscles_label = ["lax","lba","lsa","ldvm","ldlm","rdlm","rdvm","rsa","rba","rax"]
ft_label = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]

def reload_loader_model(args):
    high_train_loader, high_test_loader, low_train_loader, low_test_loader = MFT_dataset.get_MFT(filepath_high=args.filepath_high,
                                                                                                filepath_low=args.filepath_low,
                                                                                                batch_size=args.batch_size,
                                                                                                convolution=args.convolution,
                                                                                                device=args.device,
                                                                                                split=args.split,
                                                                                                evaluate=True,
                                                                                                data_seed=args.data_seed)

    covariate_data, _, FT_data, muscle_data, _, _, _, _ = next(iter(high_train_loader))
    args.covariate_dims = [covariate_data.shape[-2], covariate_data.shape[-1]]
    args.ft_dims = [FT_data.shape[-2], FT_data.shape[-1]]
    args.spike_dims = [muscle_data.shape[-2], muscle_data.shape[-1]]
    model, optimizer = model_trainer.get_model_and_optimizer(args)
    model.eval()
    model = model_trainer.load_checkpoint(args.save_path, model, args.device)
    
    return model, high_train_loader, high_test_loader, low_train_loader, low_test_loader

def load_perturb_spike(args):
        perturb_high_train_loader, perturb_high_test_loader, perturb_low_train_loader, perturb_low_test_loader = MFT_dataset.get_MFT(filepath_high=args.filepath_high,
                                                                                                filepath_low=args.filepath_low,
                                                                                                batch_size=args.batch_size,
                                                                                                convolution=args.convolution,
                                                                                                device=args.device,
                                                                                                split=args.split,
                                                                                                evaluate=True,
                                                                                                perturb=True,
                                                                                                data_seed=args.data_seed)
        return perturb_high_train_loader, perturb_high_test_loader, perturb_low_train_loader, perturb_low_test_loader


def fetch_inference_result(model, loader, perturb=None, by=1):
    model.eval()
    
    covariates = []
    test_data_ft, test_data_spike, test_data_spike_discretize, test_data_spike_count, test_moth_ids = [],[],[],[],[]
    pred_data_ft, pred_data_spike, pred_data_spike_count, decode_data_spike = [],[],[],[]
    counterfactual_pred_data_ft, counterfactual_pred_data_spike, counterfactual_pred_data_spike_count, counterfactual_decode_data_spike = [],[],[],[]
    spike_latents, spike_latents_fat = [], []
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(loader):
            covariates.append(batch_data[0].detach().cpu())
            test_data_ft.append(batch_data[2].detach().cpu())
            test_data_spike.append(batch_data[3].detach().cpu())
            test_data_spike_count.append(batch_data[4].detach().cpu())
            test_data_spike_discretize.append(batch_data[5].detach().cpu())
            test_moth_ids.append(batch_data[6].detach().cpu())
            batch_results = model.inference(batch_data[0], batch_data[7], batch_data[6], perturb, by)
            ft_pred, spike_pred, spike_counts_pred, spikes_decode, ft_pred_treat, spike_pred_treat, spike_counts_pred_treat, spikes_decode_treat = batch_results[0]
            spike_latent_wshare, spike_latent_fat_wshare = batch_results[1]
            
            pred_data_ft.append(ft_pred)
            pred_data_spike.append(spike_pred)
            pred_data_spike_count.append(spike_counts_pred)
            decode_data_spike.append(spikes_decode)
            counterfactual_pred_data_ft.append(ft_pred_treat)
            counterfactual_pred_data_spike.append(spike_pred_treat)
            counterfactual_pred_data_spike_count.append(spike_counts_pred_treat)
            counterfactual_decode_data_spike.append(spikes_decode_treat)
            spike_latents.append(spike_latent_wshare.detach().cpu())
            spike_latents_fat.append(spike_latent_fat_wshare.detach().cpu())
    
    covariates = torch.cat(covariates,dim=0).detach().cpu().numpy()
    test_moth_ids = torch.cat(test_moth_ids,dim=0).detach().cpu().numpy()
    test_data_ft = torch.cat(test_data_ft,dim=0).detach().cpu().numpy()
    test_data_spike = torch.cat(test_data_spike, dim=0).detach().cpu().numpy()
    test_data_spike_count = torch.cat(test_data_spike_count, dim=0).detach().cpu().numpy().astype(int)
    test_data_spike_discretize = torch.cat(test_data_spike_discretize, dim=0).detach().cpu().numpy().astype(int)
    pred_data_ft = torch.cat(pred_data_ft, dim=0).cpu().detach().numpy()
    pred_data_spike = torch.cat(pred_data_spike, dim=0).detach().cpu().numpy()
    pred_data_spike_count = np.round(torch.cat(pred_data_spike_count, dim=0).detach().cpu().numpy()).astype(int)
    decode_data_spike = torch.cat(decode_data_spike, dim=0).detach().cpu().numpy()
    
    # pred_data_cattn = torch.cat(pred_data_cattn, dim=0).detach().cpu().numpy()
    counterfactual_pred_data_ft = torch.cat(counterfactual_pred_data_ft, dim=0).detach().cpu().numpy()
    counterfactual_pred_data_spike = torch.cat(counterfactual_pred_data_spike, dim=0).detach().cpu().numpy()
    counterfactual_pred_data_spike_count = np.round(torch.cat(counterfactual_pred_data_spike_count, dim=0).detach().cpu().numpy()).astype(int)
    counterfactual_decode_data_spike = torch.cat(counterfactual_decode_data_spike, dim=0).detach().cpu().numpy()
    spike_latents = torch.cat(spike_latents, dim=0).detach().cpu().numpy()
    spike_latents_fat = torch.cat(spike_latents_fat, dim=0).detach().cpu().numpy()
    # counterfactual_pred_data_cattn = torch.cat(counterfactual_pred_data_cattn, dim=0).detach().cpu().numpy()
        
    return covariates, (
        test_data_ft, test_data_spike, test_data_spike_count, test_data_spike_discretize, test_moth_ids), (
        pred_data_ft, pred_data_spike, pred_data_spike_count, decode_data_spike), (
        counterfactual_pred_data_ft, counterfactual_pred_data_spike, counterfactual_pred_data_spike_count, counterfactual_decode_data_spike), (
        spike_latents, spike_latents_fat)


def fetch_inference_perturb_result(model, perturb_low_loader, perturb_high_loader):
    model.eval()
    
    low_covariates, high_covariates = [], []
    low_test_data_ft, low_test_data_spike, low_test_data_spike_discretize, low_test_data_spike_count, low_test_moth_ids = [],[],[],[],[]
    high_test_data_ft, high_test_data_spike, high_test_data_spike_discretize, high_test_data_spike_count, high_test_moth_ids = [],[],[],[],[]
    
    low_pred_data_ft, low_pred_data_spike, low_pred_data_spike_count, low_decode_data_spike = [],[],[],[]
    high_pred_data_ft, high_pred_data_spike, high_pred_data_spike_count, high_decode_data_spike = [],[],[],[]
    spike_latents, spike_latents_fat = [], []
    with torch.no_grad():
        for batch_idx, (batch_data_low, batch_data_high) in enumerate(zip(perturb_low_loader, perturb_high_loader)):
            low_covariates.append(batch_data_low[0].detach().cpu())
            low_test_data_ft.append(batch_data_low[2].detach().cpu())
            low_test_data_spike.append(batch_data_low[3].detach().cpu())
            low_test_data_spike_count.append(batch_data_low[4].detach().cpu())
            low_test_data_spike_discretize.append(batch_data_low[5].detach().cpu())
            low_test_moth_ids.append(batch_data_low[6].detach().cpu())
            
            high_covariates.append(batch_data_high[0].detach().cpu())
            high_test_data_ft.append(batch_data_high[2].detach().cpu())
            high_test_data_spike.append(batch_data_high[3].detach().cpu())
            high_test_data_spike_count.append(batch_data_high[4].detach().cpu())
            high_test_data_spike_discretize.append(batch_data_high[5].detach().cpu())
            high_test_moth_ids.append(batch_data_high[6].detach().cpu())
            
            batch_results = model.inference_perturb(batch_data_low[3], batch_data_high[3])
            ft_pred, spike_pred, spike_counts_pred, spikes_decode, ft_pred_treat, spike_pred_treat, spike_counts_pred_treat, spikes_decode_treat = batch_results[0]
            spike_latent_wshare, spike_latent_fat_wshare = batch_results[1]
            
            low_pred_data_ft.append(ft_pred)
            low_pred_data_spike.append(spike_pred)
            low_pred_data_spike_count.append(spike_counts_pred)
            low_decode_data_spike.append(spikes_decode)
            high_pred_data_ft.append(ft_pred_treat)
            high_pred_data_spike.append(spike_pred_treat)
            high_pred_data_spike_count.append(spike_counts_pred_treat)
            high_decode_data_spike.append(spikes_decode_treat)
            
            spike_latents.append(spike_latent_wshare.detach().cpu())
            spike_latents_fat.append(spike_latent_fat_wshare.detach().cpu())
    
    low_covariates = torch.cat(low_covariates,dim=0).detach().cpu().numpy()
    high_covariates = torch.cat(high_covariates,dim=0).detach().cpu().numpy()
    
    low_test_data_ft = torch.cat(low_test_data_ft,dim=0).detach().cpu().numpy()
    low_test_data_spike = torch.cat(low_test_data_spike, dim=0).detach().cpu().numpy()
    low_test_data_spike_count = torch.cat(low_test_data_spike_count, dim=0).detach().cpu().numpy().astype(int)
    low_test_data_spike_discretize = torch.cat(low_test_data_spike_discretize, dim=0).detach().cpu().numpy().astype(int)
    
    high_test_data_ft = torch.cat(high_test_data_ft,dim=0).detach().cpu().numpy()
    high_test_data_spike = torch.cat(high_test_data_spike, dim=0).detach().cpu().numpy()
    high_test_data_spike_count = torch.cat(high_test_data_spike_count, dim=0).detach().cpu().numpy().astype(int)
    high_test_data_spike_discretize = torch.cat(high_test_data_spike_discretize, dim=0).detach().cpu().numpy().astype(int)
    
    low_pred_data_ft = torch.cat(low_pred_data_ft, dim=0).cpu().detach().numpy()
    #low_pred_data_spike = torch.cat(low_pred_data_spike, dim=0).detach().cpu().numpy()
    #low_pred_data_spike_count = np.round(torch.cat(low_pred_data_spike_count, dim=0).detach().cpu().numpy()).astype(int)
    low_decode_data_spike = torch.cat(low_decode_data_spike, dim=0).detach().cpu().numpy()
    
    # pred_data_cattn = torch.cat(pred_data_cattn, dim=0).detach().cpu().numpy()
    high_pred_data_ft = torch.cat(high_pred_data_ft, dim=0).detach().cpu().numpy()
    #high_pred_data_spike = torch.cat(high_pred_data_spike, dim=0).detach().cpu().numpy()
    #high_pred_data_spike_count = np.round(torch.cat(high_pred_data_spike_count, dim=0).detach().cpu().numpy()).astype(int)
    high_decode_data_spike = torch.cat(high_decode_data_spike, dim=0).detach().cpu().numpy()
    
    spike_latents = torch.cat(spike_latents, dim=0).detach().cpu().numpy()
    spike_latents_fat = torch.cat(spike_latents_fat, dim=0).detach().cpu().numpy()
    # counterfactual_pred_data_cattn = torch.cat(counterfactual_pred_data_cattn, dim=0).detach().cpu().numpy()
        
    return (low_covariates, high_covariates), (
        low_test_data_ft, low_test_data_spike, low_test_data_spike_discretize, low_test_data_spike_count, low_test_moth_ids), (
        high_test_data_ft, high_test_data_spike, high_test_data_spike_discretize, high_test_data_spike_count, high_test_moth_ids), (
        low_pred_data_ft, low_pred_data_spike, low_pred_data_spike_count, low_decode_data_spike), (
        high_pred_data_ft, high_pred_data_spike, high_pred_data_spike_count, high_decode_data_spike), (
        spike_latents, spike_latents_fat)
        

def plot_by_index(plot_idx, covariates, 
                  test_data_ft, test_data_spike, test_data_spike_count, 
                  pred_data_ft, pred_data_spike, pred_data_spike_count, pred_data_cattn,
                  counterfactual_pred_data_ft, counterfactual_pred_data_spike, counterfactual_pred_data_spike_count, counterfactual_pred_data_cattn,
                  test_data_spike_discretize, pred_data_spike_discretize, counterfactual_pred_data_spike_discretize,
                  treatment="Low"):
        dpi = 300
        fig, ax = plt.subplots(figsize=(10,3), nrows=1, ncols=3, dpi = dpi)

        ax[0].plot(covariates[plot_idx, :, 0].flatten())
        ax[0].plot(covariates[plot_idx, :, 1].flatten())
        ax[0].legend(['Position', 'Velocity'])
        ax[0].set_title("Flower position and velocity")
        
        ax[1].scatter(list(range(6)), test_data_ft[plot_idx])
        ax[1].scatter(list(range(6)), pred_data_ft[plot_idx])
        ax[1].scatter(list(range(6)), counterfactual_pred_data_ft[plot_idx])
        ax[1].set_xticks(ticks = list(range(6)), labels=['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz'])
        ax[1].legend(['Empirical', 'Prerdicted', 'Counterfactual'])
        ax[1].set_title(f"Treatment: {treatment}, predicted mean FT")
        
        ax[2].scatter(list(range(6)), test_data_ft[plot_idx] - pred_data_ft[plot_idx])
        ax[2].set_xticks(ticks = list(range(6)), labels=['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz'])
        ax[2].set_title("Predicted mean FT Error")
        
        plt.tight_layout()
        plt.show()

        
        fig, ax = plt.subplots(figsize=(10,8), nrows=3, ncols=2, dpi = dpi)
        ax[0][0].imshow(test_data_spike[plot_idx].T, aspect=20, interpolation="none")
        ax[0][0].set_yticks(range(0,len(muscles_label)), muscles_label)
        ax[0][0].set_title(f"Treatment: {treatment}, spikes")
        ax[0][1].imshow(test_data_spike_discretize[plot_idx].T, aspect=20, interpolation="none")
        ax[0][1].set_yticks(range(0,len(muscles_label)), muscles_label)
        ax[0][1].set_title(f"Treatment: {treatment}, Discretized spikes")
        
        ax[1][0].imshow(pred_data_spike[plot_idx].T, aspect=20, interpolation="none")
        ax[1][0].set_yticks(range(0,len(muscles_label)), muscles_label)
        ax[1][0].set_title(f"Treatment: {treatment}, Predicted spikes")
        ax[1][1].imshow(pred_data_spike_discretize[plot_idx].T, aspect=20, interpolation="none")
        ax[1][1].set_yticks(range(0,len(muscles_label)), muscles_label)
        ax[1][1].set_title(f"Treatment: {treatment}, Discretized Predicted spikes")

        ax[2][0].imshow(counterfactual_pred_data_spike[plot_idx].T, aspect=20, interpolation="none")
        ax[2][0].set_yticks(range(0,len(muscles_label)), muscles_label)
        ax[2][0].set_title(f"Treatment: {treatment}, Predicted Counterfactual spikes")
        ax[2][1].imshow(counterfactual_pred_data_spike_discretize[plot_idx].T, aspect=20, interpolation="none")
        ax[2][1].set_yticks(range(0,len(muscles_label)), muscles_label)
        ax[2][1].set_title(f"Treatment: {treatment}, Discretized Predicted Counterfactual spikes")
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(10, 3), dpi=dpi)
        plt.bar(np.array(range(len(muscles_label)))-0.2, 
                test_data_spike_count[plot_idx].flatten(), 
                width=0.2, label='True', alpha=0.7)
        
        plt.bar(np.array(range(len(muscles_label))), 
                pred_data_spike_count[plot_idx].flatten(), 
                width=0.2, label='Predicted', alpha=0.7)
        
        plt.bar(np.array(range(len(muscles_label)))+0.2, 
                counterfactual_pred_data_spike_count[plot_idx].flatten(), 
                width=0.2, label='CF Predicted', alpha=0.7)

        plt.xticks(np.array(range(len(muscles_label))), muscles_label)
        plt.ylabel("Spike Count")
        plt.title(f"Treatment: {treatment}, Count Prediction")
        plt.legend()
        
        plt.tight_layout()
        plt.show()

        # fig, ax = plt.subplots(figsize=(11,2.5), nrows=1, ncols=3, dpi = dpi)
        # im1 = ax[0].imshow(pred_data_cattn[plot_idx].squeeze(0).T, vmin=0, vmax=0.5)
        # ax[0].set_xticks(list(range(0,len(muscles_label))), muscles_label)
        # ax[0].set_yticks(list(range(0,len(ft_label))), ft_label)
        # ax[0].set_xticklabels(muscles_label, rotation=90)
        
        # im2 = ax[1].imshow(counterfactual_pred_data_cattn[plot_idx].squeeze(0).T, vmin=0, vmax=0.5)
        # ax[1].set_xticks(list(range(0,len(muscles_label))), muscles_label)
        # ax[1].set_yticks(list(range(0,len(ft_label))), ft_label)
        # ax[1].set_xticklabels(muscles_label, rotation=90)
        
        # im3 = ax[2].imshow((pred_data_cattn[plot_idx] - counterfactual_pred_data_cattn[plot_idx]).squeeze(0).T, vmin=-0.1, vmax=0.1)
        # ax[2].set_xticks(list(range(0,len(muscles_label))), muscles_label)
        # ax[2].set_yticks(list(range(0,len(ft_label))), ft_label)
        # ax[2].set_xticklabels(muscles_label, rotation=90)

        # fig.colorbar(im1, ax=ax[0])
        # fig.colorbar(im2, ax=ax[1])
        # fig.colorbar(im3, ax=ax[2])
        
        # plt.show()