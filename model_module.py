"""
model_module.py

A small variational autoencoder for log-normalized HVG expression.

This is a teaching baseline, not scVI. It uses an MSE decoder on already
log-normalized values and does not model UMI counts or library size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def get_device() -> torch.device:
    """Prefer CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class VaeConfig:
    input_dim: int
    latent_dim: int = 20
    hidden_dim: int = 256
    dropout: float = 0.1


class VAE(nn.Module):
    def __init__(self, cfg: VaeConfig):
        super().__init__()
        self.cfg = cfg

        self.encoder = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.fc_logvar = nn.Linear(cfg.hidden_dim, cfg.latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.input_dim),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(
    x: torch.Tensor,
    recon: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    beta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon_loss = torch.mean(torch.sum((x - recon) ** 2, dim=1))
    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    total = recon_loss + float(beta) * kl
    return total, recon_loss, kl


def make_dataloader(
    X: np.ndarray,
    *,
    batch_size: int = 256,
    shuffle: bool = True,
) -> DataLoader:
    if X.ndim != 2:
        raise ValueError("X must be 2D (cells x genes).")
    tensor = torch.from_numpy(np.asarray(X, dtype=np.float32, order="C"))
    return DataLoader(TensorDataset(tensor), batch_size=int(batch_size), shuffle=shuffle, drop_last=False)


ProgressCallback = Callable[[int, int, float, float, float], None]


def train_vae(
    X: np.ndarray,
    *,
    latent_dim: int = 20,
    hidden_dim: int = 256,
    dropout: float = 0.1,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    beta: float = 1.0,
    seed: int = 7,
    max_grad_norm: float = 1.0,
    device: Optional[torch.device] = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> Tuple[VAE, List[float], List[float], List[float]]:
    """
    Train the VAE on a dense log-normalized HVG matrix.

    Returns
    -------
    model, total_losses, recon_losses, kl_losses
    """
    if device is None:
        device = get_device()

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))

    if X.size == 0:
        raise ValueError("Empty matrix.")
    if not np.isfinite(X).all():
        raise ValueError("Matrix contains NaN/Inf; check preprocessing.")

    dl = make_dataloader(X, batch_size=batch_size, shuffle=True)
    cfg = VaeConfig(
        input_dim=X.shape[1],
        latent_dim=int(latent_dim),
        hidden_dim=int(hidden_dim),
        dropout=float(dropout),
    )
    model = VAE(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))

    losses: List[float] = []
    recon_losses: List[float] = []
    kl_losses: List[float] = []

    model.train()
    n_epochs = int(epochs)
    for epoch in range(n_epochs):
        epoch_total = []
        epoch_recon = []
        epoch_kl = []
        for (xb,) in dl:
            xb = xb.to(device)
            opt.zero_grad(set_to_none=True)
            recon, mu, logvar = model(xb)
            loss, recon_loss, kl = vae_loss(xb, recon, mu, logvar, beta=float(beta))
            loss.backward()
            if max_grad_norm is not None and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(max_grad_norm))
            opt.step()
            epoch_total.append(loss.detach().float().cpu().item())
            epoch_recon.append(recon_loss.detach().float().cpu().item())
            epoch_kl.append(kl.detach().float().cpu().item())

        total_mean = float(np.mean(epoch_total)) if epoch_total else float("nan")
        recon_mean = float(np.mean(epoch_recon)) if epoch_recon else float("nan")
        kl_mean = float(np.mean(epoch_kl)) if epoch_kl else float("nan")
        losses.append(total_mean)
        recon_losses.append(recon_mean)
        kl_losses.append(kl_mean)
        if progress_cb is not None:
            progress_cb(epoch + 1, n_epochs, total_mean, recon_mean, kl_mean)

    return model, losses, recon_losses, kl_losses


@torch.no_grad()
def encode_latent(
    model: VAE,
    X: np.ndarray,
    *,
    batch_size: int = 512,
    device: Optional[torch.device] = None,
    use_mean: bool = True,
) -> np.ndarray:
    """Encode cells to latent space. Default uses the posterior mean (deterministic)."""
    if device is None:
        device = get_device()

    model.eval()
    dl = make_dataloader(X, batch_size=batch_size, shuffle=False)
    zs = []
    for (xb,) in dl:
        xb = xb.to(device)
        mu, logvar = model.encode(xb)
        z = mu if use_mean else model.reparameterize(mu, logvar)
        zs.append(z.detach().cpu().numpy())
    return np.vstack(zs).astype(np.float32, copy=False)
