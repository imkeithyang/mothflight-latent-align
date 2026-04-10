import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes,
        output_size: int,
        dropout: float = 0.0,
        activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        if activation is None:
            activation = nn.ReLU()
        layers = []
        prev = input_size
        for hidden in hidden_sizes:
            layers.append(nn.Linear(prev, hidden))
            layers.append(activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hidden
        layers.append(nn.Linear(prev, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class AttentionPooling(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, d_model))
        self.linear = nn.Linear(d_model, d_model)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query.expand(x.size(0), -1).unsqueeze(1)
        k = self.tanh(self.linear(x))
        attn = torch.softmax(torch.bmm(q, k.transpose(1, 2)), dim=-1)
        return torch.bmm(attn, x).squeeze(1)


class SequenceEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        d_model: int = 64,
        nhead: int = 4,
        d_ff: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=seq_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pool = AttentionPooling(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.pool(x)


class SequenceDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        seq_len: int,
        output_dim: int,
        hidden_sizes,
        cond_dim: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.net = MLPModel(
            input_size=latent_dim + cond_dim,
            hidden_sizes=hidden_sizes,
            output_size=seq_len * output_dim,
            dropout=dropout,
        )

    def forward(self, latent: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        if cond is not None:
            latent = torch.cat([latent, cond], dim=-1)
        out = self.net(latent)
        return out.view(latent.shape[0], self.seq_len, self.output_dim)


class VectorDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_sizes,
        cond_dim: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = MLPModel(
            input_size=latent_dim + cond_dim,
            hidden_sizes=hidden_sizes,
            output_size=output_dim,
            dropout=dropout,
        )

    def forward(self, latent: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        if cond is not None:
            latent = torch.cat([latent, cond], dim=-1)
        return self.net(latent)


class Transformer_TAR_net(nn.Module):
    def __init__(
        self,
        covariate_dims=[400, 18],
        ft_dims=[6],
        spike_dims=[400, 10],
        d_model=64,
        d_latent=1,
        d_latent_share=3,
        d_latent_treat=2,
        num_moths=10,
        dropout=0.1,
        n_heads=4,
        d_ff=128,
        e_layers=2,
        device="cpu",
        l1_lambda=1e-3,
        l1_lambda_treat=1e-3,
        l1_lambda_share=1e-3,
        flower_latent_dim: Optional[int] = None,
        spike_latent_dim: Optional[int] = 3,
        moth_embed_dim: int = 1,
        align_weight: float = 1.0,
        flower_pos_weight: float = 1,
        ft_predictor_mode: str = "shared_shift",
        flower_recon_mode: str = "sequence",
        flower_decoder_latent_source: str = "shared",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.covariate_dims = covariate_dims
        self.ft_dims = ft_dims
        self.spike_dims = spike_dims
        self.seq_len = covariate_dims[0]
        self.flower_dim = covariate_dims[-1]
        self.spike_feature_dim = spike_dims[-1]
        self.ft_dim = ft_dims[-1]
        self.align_weight = align_weight
        self.flower_pos_weight = flower_pos_weight
        self.ft_predictor_mode = ft_predictor_mode
        self.flower_recon_mode = flower_recon_mode
        self.flower_decoder_latent_source = flower_decoder_latent_source
        if self.ft_predictor_mode not in {"shared_shift", "per_mass", "per_moth_mass"}:
            raise ValueError(f"Unknown ft_predictor_mode: {self.ft_predictor_mode}")
        if self.flower_recon_mode not in {"sequence", "mean"}:
            raise ValueError(f"Unknown flower_recon_mode: {self.flower_recon_mode}")
        if self.flower_decoder_latent_source not in {"shared", "full", "spike_shared"}:
            raise ValueError(f"Unknown flower_decoder_latent_source: {self.flower_decoder_latent_source}")

        self.flower_latent_dim = flower_latent_dim if flower_latent_dim is not None else d_latent
        self.spike_latent_dim = spike_latent_dim if spike_latent_dim is not None else 3
        self.mass_latent_dim = 1
        if self.spike_latent_dim < self.mass_latent_dim + 1:
            raise ValueError("spike_latent_dim must be at least 2 so one dim can store mass")
        self.shared_latent_dim = min(self.flower_latent_dim, self.spike_latent_dim - self.mass_latent_dim)
        if self.shared_latent_dim < 1:
            raise ValueError("shared_latent_dim must be at least 1")
        self.spike_private_dim = self.spike_latent_dim - self.shared_latent_dim - self.mass_latent_dim
        if self.spike_private_dim < 0:
            raise ValueError("spike_latent_dim must be at least shared_latent_dim + 1")
        self.flower_decoder_latent_dim = (
            self.flower_latent_dim if self.flower_decoder_latent_source == "full" else self.shared_latent_dim
        )

        self.num_moths = num_moths
        self.moth_embedding = nn.Embedding(num_moths, moth_embed_dim).to(self.device)
        self.cond_dim = moth_embed_dim

        self.flower_encoder = SequenceEncoder(
            input_dim=self.flower_dim,
            seq_len=self.seq_len,
            d_model=d_model,
            nhead=n_heads,
            d_ff=d_ff,
            num_layers=e_layers,
            dropout=dropout,
        ).to(self.device)
        self.flower_latent_head = MLPModel(
            input_size=d_model,
            hidden_sizes=[d_model],
            output_size=self.flower_latent_dim,
            dropout=dropout,
        ).to(self.device)
        self.spike_encoder = SequenceEncoder(
            input_dim=self.spike_feature_dim,
            seq_len=spike_dims[0],
            d_model=d_model,
            nhead=n_heads,
            d_ff=d_ff,
            num_layers=e_layers,
            dropout=dropout,
        ).to(self.device)
        self.spike_latent_head = MLPModel(
            input_size=d_model + self.cond_dim,
            hidden_sizes=[d_model, d_model],
            output_size=self.spike_latent_dim,
            dropout=dropout,
        ).to(self.device)

        decoder_hidden = [2 * d_model, 2 * d_model]
        self.flower_decoder = None
        self.flower_mean_decoder = None
        if self.flower_recon_mode == "sequence":
            self.flower_decoder = SequenceDecoder(
                latent_dim=self.flower_decoder_latent_dim,
                seq_len=self.seq_len,
                output_dim=self.flower_dim,
                hidden_sizes=decoder_hidden,
                cond_dim=0,
                dropout=dropout,
            ).to(self.device)
        else:
            self.flower_mean_decoder = VectorDecoder(
                latent_dim=self.flower_decoder_latent_dim,
                output_dim=self.flower_dim,
                hidden_sizes=decoder_hidden,
                cond_dim=0,
                dropout=dropout,
            ).to(self.device)
        self.spike_decoder = SequenceDecoder(
            latent_dim=self.spike_latent_dim,
            seq_len=spike_dims[0],
            output_dim=self.spike_feature_dim,
            hidden_sizes=decoder_hidden,
            cond_dim=self.cond_dim,
            dropout=dropout,
        ).to(self.device)
        self.flower_to_spike_bridge = None
        if self.spike_private_dim > 0:
            self.flower_to_spike_bridge = MLPModel(
                input_size=self.shared_latent_dim,
                hidden_sizes=[d_model, d_model],
                output_size=self.spike_private_dim,
                dropout=dropout,
            ).to(self.device)

        self.ft_predictor = nn.Linear(self.shared_latent_dim + self.mass_latent_dim + moth_embed_dim, self.ft_dim).to(self.device)
        self.ft_predictor_low = None
        self.ft_predictor_high = None
        self.ft_predictor_low_by_moth = None
        self.ft_predictor_high_by_moth = None
        if self.ft_predictor_mode == "per_mass":
            ft_input_dim = self.shared_latent_dim + moth_embed_dim
            self.ft_predictor_low = nn.Linear(ft_input_dim, self.ft_dim).to(self.device)
            self.ft_predictor_high = nn.Linear(ft_input_dim, self.ft_dim).to(self.device)
        elif self.ft_predictor_mode == "per_moth_mass":
            ft_input_dim = self.shared_latent_dim
            self.ft_predictor_low_by_moth = nn.ModuleList(
                [nn.Linear(ft_input_dim, self.ft_dim) for _ in range(self.num_moths)]
            ).to(self.device)
            self.ft_predictor_high_by_moth = nn.ModuleList(
                [nn.Linear(ft_input_dim, self.ft_dim) for _ in range(self.num_moths)]
            ).to(self.device)
        self.spike_count_predictor = nn.Linear(self.spike_latent_dim, self.spike_feature_dim).to(self.device)
        self.poisson_loss = nn.PoissonNLLLoss(log_input=True, full=True, reduction="mean")

    def _build_condition(self, mass: torch.Tensor, moth_ids: Optional[torch.Tensor]) -> torch.Tensor:
        batch_size = mass.shape[0]
        if moth_ids is None:
            return torch.zeros(batch_size, self.moth_embedding.embedding_dim, device=self.device)
        return self.moth_embedding(moth_ids.long().view(-1))

    def _unpack_batch(self, batch):
        if len(batch) == 7:
            flower, ft_full, ft_mean, spikes, spike_counts, moth_ids, mass = batch
        elif len(batch) == 8:
            flower, ft_full, ft_mean, spikes, spike_counts, _, moth_ids, mass = batch
        else:
            raise ValueError(f"Unexpected batch length {len(batch)}")
        return flower, ft_mean, spikes, spike_counts, moth_ids, mass

    def _encode_flower(self, flower: torch.Tensor) -> torch.Tensor:
        return self.flower_latent_head(self.flower_encoder(flower))

    def _encode_spike(self, spike: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.spike_latent_head(torch.cat([self.spike_encoder(spike), cond], dim=-1))

    def _init_spike_latent(self, mass: torch.Tensor) -> torch.Tensor:
        mass = mass.float().view(mass.shape[0], -1)
        spike_latent = torch.zeros(
            mass.shape[0],
            self.spike_latent_dim,
            device=mass.device,
            dtype=mass.dtype,
        )
        spike_latent[:, self.shared_latent_dim : self.shared_latent_dim + self.mass_latent_dim] = mass
        return spike_latent

    def _flower_shared_latent(self, flower_latent: torch.Tensor) -> torch.Tensor:
        return flower_latent[:, : self.shared_latent_dim]

    def _flower_decoder_latent(self, flower_latent: torch.Tensor, spike_latent: torch.Tensor) -> torch.Tensor:
        if self.flower_decoder_latent_source == "spike_shared":
            return spike_latent[:, : self.shared_latent_dim]
        if self.flower_decoder_latent_source == "full":
            return flower_latent
        return self._flower_shared_latent(flower_latent)

    def _cross_flower_decoder_latent(self, spike_latent: torch.Tensor) -> torch.Tensor:
        spike_shared = spike_latent[:, : self.shared_latent_dim]
        if self.flower_decoder_latent_source != "full":
            return spike_shared

        flower_latent = torch.zeros(
            spike_latent.shape[0],
            self.flower_latent_dim,
            device=spike_latent.device,
            dtype=spike_latent.dtype,
        )
        flower_latent[:, : self.shared_latent_dim] = spike_shared
        return flower_latent

    def _replace_shared_latent(self, spike_latent: torch.Tensor, flower_latent: torch.Tensor) -> torch.Tensor:
        cross_spike_latent = spike_latent.clone()
        cross_spike_latent[:, : self.shared_latent_dim] = self._flower_shared_latent(flower_latent)
        return cross_spike_latent

    def _predict_ft_shared(self, flower_latent: torch.Tensor, mass: torch.Tensor, moth_ids: Optional[torch.Tensor]) -> torch.Tensor:
        mass = mass.float().view(mass.shape[0], -1)
        if moth_ids is None:
            moth_emb = torch.zeros(flower_latent.shape[0], self.moth_embedding.embedding_dim, device=flower_latent.device)
        else:
            moth_emb = self.moth_embedding(moth_ids.long().view(-1))
        return self.ft_predictor(torch.cat([flower_latent, mass, moth_emb], dim=-1))

    def _predict_ft_per_mass(self, flower_latent: torch.Tensor, mass: torch.Tensor, moth_ids: Optional[torch.Tensor]) -> torch.Tensor:
        mass = mass.float().view(mass.shape[0], -1)
        if moth_ids is None:
            moth_emb = torch.zeros(flower_latent.shape[0], self.moth_embedding.embedding_dim, device=flower_latent.device)
        else:
            moth_emb = self.moth_embedding(moth_ids.long().view(-1))
        ft_input = torch.cat([flower_latent, moth_emb], dim=-1)
        ft_pred = torch.zeros(flower_latent.shape[0], self.ft_dim, device=flower_latent.device)
        low_mask = mass.view(-1) <= 0.5
        high_mask = ~low_mask
        if low_mask.any():
            ft_pred[low_mask] = self.ft_predictor_low(ft_input[low_mask])
        if high_mask.any():
            ft_pred[high_mask] = self.ft_predictor_high(ft_input[high_mask])
        return ft_pred

    def _predict_ft_per_moth_mass(self, flower_latent: torch.Tensor, mass: torch.Tensor, moth_ids: Optional[torch.Tensor]) -> torch.Tensor:
        mass = mass.float().view(mass.shape[0], -1)
        ft_pred = torch.zeros(flower_latent.shape[0], self.ft_dim, device=flower_latent.device, dtype=flower_latent.dtype)
        low_mask = mass.view(-1) <= 0.5
        high_mask = ~low_mask

        if moth_ids is None:
            if low_mask.any():
                low_pred = torch.stack(
                    [head(flower_latent[low_mask]) for head in self.ft_predictor_low_by_moth],
                    dim=0,
                ).mean(dim=0)
                ft_pred[low_mask] = low_pred
            if high_mask.any():
                high_pred = torch.stack(
                    [head(flower_latent[high_mask]) for head in self.ft_predictor_high_by_moth],
                    dim=0,
                ).mean(dim=0)
                ft_pred[high_mask] = high_pred
            return ft_pred

        moth_ids = moth_ids.long().view(-1)
        for moth_idx in range(self.num_moths):
            moth_mask = moth_ids == moth_idx
            if not moth_mask.any():
                continue
            low_moth_mask = moth_mask & low_mask
            high_moth_mask = moth_mask & high_mask
            if low_moth_mask.any():
                ft_pred[low_moth_mask] = self.ft_predictor_low_by_moth[moth_idx](flower_latent[low_moth_mask])
            if high_moth_mask.any():
                ft_pred[high_moth_mask] = self.ft_predictor_high_by_moth[moth_idx](flower_latent[high_moth_mask])
        return ft_pred

    def _predict_ft(self, flower_latent: torch.Tensor, mass: torch.Tensor, moth_ids: Optional[torch.Tensor]) -> torch.Tensor:
        if self.ft_predictor_mode == "per_mass":
            return self._predict_ft_per_mass(flower_latent, mass, moth_ids)
        if self.ft_predictor_mode == "per_moth_mass":
            return self._predict_ft_per_moth_mass(flower_latent, mass, moth_ids)
        return self._predict_ft_shared(flower_latent, mass, moth_ids)


    def _flower_recon_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pos_loss = F.mse_loss(prediction[..., 0], target[..., 0], reduction="mean")
        if target.shape[-1] == 1:
            return self.flower_pos_weight * pos_loss
        vel_loss = F.mse_loss(prediction[..., 1:], target[..., 1:], reduction="mean")
        return self.flower_pos_weight * pos_loss + vel_loss

    def _flower_target(self, flower: torch.Tensor) -> torch.Tensor:
        if self.flower_recon_mode == "mean":
            return flower.mean(dim=1)
        return flower

    def _decode_flower(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.flower_recon_mode == "mean":
            flower_recon_mean = self.flower_mean_decoder(latent)
            flower_recon = flower_recon_mean.unsqueeze(1).expand(-1, self.seq_len, -1)
            return flower_recon_mean, flower_recon
        flower_recon = self.flower_decoder(latent)
        return flower_recon.mean(dim=1), flower_recon

    def _decode_pair(self, flower_latent: torch.Tensor, spike_latent: torch.Tensor, cond: torch.Tensor):
        cross_spike_latent = self._replace_shared_latent(spike_latent, flower_latent)
        flower_recon_mean, flower_recon = self._decode_flower(
            self._flower_decoder_latent(flower_latent, spike_latent)
        )
        spike_recon = self.spike_decoder(spike_latent, cond)
        cross_spike_recon = self.spike_decoder(cross_spike_latent, cond)
        cross_flower_recon_mean, cross_flower_recon = self._decode_flower(
            self._cross_flower_decoder_latent(spike_latent)
        )
        return (
            flower_latent,
            spike_latent,
            cross_spike_latent,
            flower_recon_mean,
            flower_recon,
            spike_recon,
            cross_spike_recon,
            cross_flower_recon_mean,
            cross_flower_recon,
        )

    def _batch_step(self, batch, mass_target: torch.Tensor):
        flower, ft_target, spikes, spike_counts, moth_ids, mass = self._unpack_batch(batch)
        flower = flower.to(self.device)
        ft_target = ft_target.to(self.device)
        spikes = spikes.to(self.device)
        spike_counts = spike_counts.to(self.device)
        moth_ids = moth_ids.to(self.device)
        mass = mass.to(self.device)
        mass_target = mass_target.to(self.device).float().view(mass.shape[0], -1)

        cond = self._build_condition(mass, moth_ids)
        spike_latent = self._encode_spike(spikes, cond)
        if self.flower_decoder_latent_source == "spike_shared":
            flower_latent = spike_latent[:, : self.shared_latent_dim]
        else:
            flower_latent = self._encode_flower(flower)
        mass_input = mass.float().view(mass.shape[0], -1)
        (
            flower_latent,
            spike_latent,
            cross_spike_latent,
            flower_recon_mean,
            flower_recon,
            spike_recon,
            cross_spike_recon,
            cross_flower_recon_mean,
            cross_flower_recon,
        ) = self._decode_pair(
            flower_latent, spike_latent, cond
        )
        ft_latent = self._flower_shared_latent(flower_latent)
        ft_pred = self._predict_ft(ft_latent, mass_input, moth_ids)
        spike_counts_pred = self.spike_count_predictor(spike_latent)
        spike_counts_pred_from_flower = self.spike_count_predictor(cross_spike_latent)
        flower_target = self._flower_target(flower)
        if self.flower_recon_mode == "mean":
            flower_recon_target = flower_recon_mean
            cross_flower_recon_target = cross_flower_recon_mean
            flower_recon_scale = 1.0
        else:
            flower_recon_target = flower_recon
            cross_flower_recon_target = cross_flower_recon
            flower_recon_scale = float(flower.shape[1])

        if self.flower_decoder_latent_source == "spike_shared":
            align_loss = torch.tensor(0.0, device=self.device)
        else:
            align_loss = self.align_weight * F.mse_loss(
                spike_latent[:, : self.shared_latent_dim],
                self._flower_shared_latent(flower_latent),
                reduction="mean",
            )
        flower_recon_loss = self._flower_recon_loss(flower_recon_target, flower_target)
        spike_recon_loss = F.mse_loss(spike_recon, spikes, reduction="mean")
        cross_flower_loss = self._flower_recon_loss(cross_flower_recon_target, flower_target)
        cross_spike_loss = F.mse_loss(cross_spike_recon, spikes, reduction="mean")
        ft_loss = F.mse_loss(ft_pred, ft_target, reduction="mean")
        count_loss = 0.5 * (
            self.poisson_loss(spike_counts_pred, spike_counts) + self.poisson_loss(spike_counts_pred_from_flower, spike_counts)
        )
        mass_slice = spike_latent[:, self.shared_latent_dim : self.shared_latent_dim + 1]
        mass_loss = F.binary_cross_entropy_with_logits(mass_slice, mass_target)

        losses = {
            "align": align_loss + mass_loss,
            "ft": ft_loss,
            "recons": flower_recon_loss / flower_recon_scale,# + cross_flower_loss / flower_recon_scale,
            "kld": torch.tensor(0.0, device=self.device),
            "spikes": spike_recon_loss, #+ cross_spike_loss,
            "spike_counts": count_loss,
        }
        outputs = {
            "flower_latent": flower_latent,
            "spike_latent": spike_latent,
            "flower_recon_mean": flower_recon_mean,
            "flower_recon": flower_recon,
            "spike_recon": spike_recon,
            "cross_spike_recon": cross_spike_recon,
            "cross_flower_recon_mean": cross_flower_recon_mean,
            "cross_flower_recon": cross_flower_recon,
            "ft_pred": ft_pred,
            "spike_counts_pred": spike_counts_pred,
        }
        return losses, outputs

    def train_step(self, batch, batch_treatment):
        low_label = torch.zeros(batch[0].shape[0], 1, device=self.device)
        high_label = torch.ones(batch_treatment[0].shape[0], 1, device=self.device)
        losses_a, _ = self._batch_step(batch, low_label)
        losses_b, _ = self._batch_step(batch_treatment, high_label)

        return (
            0.5 * (losses_a["align"] + losses_b["align"]),
            0.5 * (losses_a["ft"] + losses_b["ft"]),
            0.5 * (losses_a["recons"] + losses_b["recons"]),
            0.5 * (losses_a["kld"] + losses_b["kld"]),
            0.5 * (losses_a["spikes"] + losses_b["spikes"]),
            0.5 * (losses_a["spike_counts"] + losses_b["spike_counts"]),
        )

    def forward(self, covariates, covariates_treat=None, mass=None, moth_ids=None):
        if mass is None:
            mass = torch.zeros(covariates.shape[0], 1, device=covariates.device)
        mass = mass.to(self.device).float().view(mass.shape[0], -1)
        moth_ids = moth_ids.to(self.device) if moth_ids is not None else None
        cond = self._build_condition(mass, moth_ids)
        if covariates_treat is not None:
            spike_latent = self._encode_spike(covariates_treat.to(self.device), cond)
            if self.flower_decoder_latent_source == "spike_shared":
                flower_latent = spike_latent[:, : self.shared_latent_dim]
            else:
                flower_latent = self._encode_flower(covariates.to(self.device))
        else:
            flower_latent = self._encode_flower(covariates.to(self.device))
            spike_latent = self._replace_shared_latent(self._init_spike_latent(mass), flower_latent)
        (
            flower_latent,
            spike_latent,
            cross_spike_latent,
            flower_recon_mean,
            flower_recon,
            spike_recon,
            cross_spike_recon,
            cross_flower_recon_mean,
            cross_flower_recon,
        ) = self._decode_pair(
            flower_latent, spike_latent, cond
        )
        ft_pred = self._predict_ft(self._flower_shared_latent(flower_latent), mass, moth_ids)
        spike_counts_pred = self.spike_count_predictor(spike_latent)
        return {
            "flower_recon": flower_recon,
            "flower_recon_mean": flower_recon_mean,
            "spike_recon": spike_recon,
            "cross_spike_recon": cross_spike_recon,
            "cross_flower_recon_mean": cross_flower_recon_mean,
            "cross_flower_recon": cross_flower_recon,
            "ft_pred": ft_pred,
            "spike_counts_pred": spike_counts_pred,
            "flower_latent": flower_latent,
            "spike_latent": spike_latent,
        }

    def inference(self, covariates, mass, moth_ids=None, perturb=None, by=1):
        covariates = covariates.to(self.device)
        mass = mass.to(self.device).float().view(mass.shape[0], -1)
        moth_ids = moth_ids.to(self.device) if moth_ids is not None else None
        cond = self._build_condition(mass, moth_ids)
        flower_latent = self._encode_flower(covariates)
        spike_latent = self._replace_shared_latent(self._init_spike_latent(mass), flower_latent)
        if perturb is not None:
            spike_latent = spike_latent.clone()
            spike_latent[:, perturb] = spike_latent[:, perturb] + by
        (
            flower_latent,
            spike_latent,
            cross_spike_latent,
            flower_recon_mean,
            flower_recon,
            spike_recon,
            cross_spike_recon,
            cross_flower_recon_mean,
            cross_flower_recon,
        ) = self._decode_pair(
            flower_latent, spike_latent, cond
        )
        ft_pred = self._predict_ft(self._flower_shared_latent(flower_latent), mass, moth_ids)
        ft_pred_treat = self._predict_ft(spike_latent[:, : self.shared_latent_dim], mass, moth_ids)
        spike_counts_pred = torch.exp(self.spike_count_predictor(spike_latent))
        spike_counts_pred_treat = torch.exp(self.spike_count_predictor(cross_spike_latent))
        return (
            ft_pred,
            spike_recon,
            spike_counts_pred,
            cross_spike_recon,
            ft_pred_treat,
            spike_recon,
            spike_counts_pred_treat,
            cross_spike_recon,
        ), (
            flower_latent,
            spike_latent,
        )

    def inference_perturb(self, spike, spike_treat):
        spike = spike.to(self.device)
        spike_treat = spike_treat.to(self.device)
        mass = torch.zeros(spike.shape[0], 1, device=self.device)
        cond = torch.zeros(spike.shape[0], self.cond_dim, device=self.device)
        spike_latent = self._encode_spike(spike, cond)
        spike_latent_treat = self._encode_spike(spike_treat, cond)
        flower_latent = spike_latent[:, : self.shared_latent_dim]
        flower_latent_treat = spike_latent_treat[:, : self.shared_latent_dim]
        (
            flower_latent,
            spike_latent,
            cross_spike_latent,
            flower_recon_mean,
            flower_recon,
            spike_recon,
            cross_spike_recon,
            cross_flower_recon_mean,
            cross_flower_recon,
        ) = self._decode_pair(
            flower_latent, spike_latent, cond
        )
        (
            flower_latent_treat,
            spike_latent_treat,
            cross_spike_latent_treat,
            flower_recon_mean_treat,
            flower_recon_treat,
            spike_recon_treat,
            cross_spike_recon_treat,
            cross_flower_recon_mean_treat,
            cross_flower_recon_treat,
        ) = self._decode_pair(
            flower_latent_treat, spike_latent_treat, cond
        )
        ft_pred = self._predict_ft(self._flower_shared_latent(flower_latent), mass, None)
        ft_pred_treat = self._predict_ft(self._flower_shared_latent(flower_latent_treat), mass, None)
        return (
            ft_pred,
            spike_recon,
            torch.exp(self.spike_count_predictor(spike_latent)),
            cross_spike_recon,
            ft_pred_treat,
            spike_recon_treat,
            torch.exp(self.spike_count_predictor(spike_latent_treat)),
            cross_spike_recon_treat,
        ), (
            flower_latent,
            flower_latent_treat,
        )
