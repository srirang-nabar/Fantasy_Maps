"""Convolutional β-VAE (plan.md Stage 2).

Same architecture across the β-sweep — β only reweights the KL term in the loss,
never the network — so any disentanglement difference is attributable to β, not
capacity. Encoder downsamples to 4x4 with stride-2 convs, a linear head produces
(mu, logvar); the decoder mirrors it with a sigmoid output in [0,1]. Sized
dynamically for img_size in {64, 128}.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

_LOG2PI = math.log(2 * math.pi)


class ConvVAE(nn.Module):
    def __init__(self, img_size: int = 64, latent_dim: int = 16, ch: int = 32):
        super().__init__()
        if img_size < 8 or (img_size & (img_size - 1)) != 0:
            raise ValueError(f"img_size must be a power of two >= 8, got {img_size}")
        self.img_size = img_size
        self.latent_dim = latent_dim
        n_down = int(math.log2(img_size)) - 2  # downsample to 4x4
        chans = [3] + [min(ch * (2 ** i), 256) for i in range(n_down)]

        enc = []
        for i in range(n_down):
            enc += [nn.Conv2d(chans[i], chans[i + 1], 4, stride=2, padding=1),
                    nn.BatchNorm2d(chans[i + 1]), nn.ReLU(inplace=True)]
        self.encoder = nn.Sequential(*enc)
        self.feat_ch = chans[-1]
        self.flat = self.feat_ch * 4 * 4
        self.fc_mu = nn.Linear(self.flat, latent_dim)
        self.fc_logvar = nn.Linear(self.flat, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.flat)

        dec = []
        rev = chans[::-1]  # [feat_ch, ..., 3]
        for i in range(n_down):
            out_c = rev[i + 1]
            last = i == n_down - 1
            dec += [nn.ConvTranspose2d(rev[i], out_c, 4, stride=2, padding=1)]
            if not last:
                dec += [nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)]
        dec += [nn.Sigmoid()]
        self.decoder = nn.Sequential(*dec)

    def encode(self, x):
        h = self.encoder(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        h = self.fc_dec(z).view(-1, self.feat_ch, 4, 4)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z


def vae_loss(recon, x, mu, logvar, beta: float):
    """Returns (loss, recon_term, kl_term). loss == recon_term + beta*kl_term exactly.

    recon_term is per-image summed pixel MSE (Gaussian decoder, unit variance),
    averaged over the batch; kl_term is the analytic KL to N(0, I), same reduction.
    """
    recon_term = F.mse_loss(recon, x, reduction="none").flatten(1).sum(1).mean()
    kl_term = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(1).mean()
    loss = recon_term + beta * kl_term
    return loss, recon_term, kl_term


class PatchDiscriminator(nn.Module):
    """Spectral-norm PatchGAN (pix2pix-style): classifies overlapping patches real/fake.

    Used by the VAE-GAN (train_gan.py) to replace pixel-MSE with an adversarial signal,
    which is what actually produces sharp edges instead of blur. Spectral norm keeps
    adversarial training stable without extra gradient penalties.
    """

    def __init__(self, ch: int = 64, n_layers: int = 3):
        super().__init__()
        SN = nn.utils.spectral_norm
        layers = [SN(nn.Conv2d(3, ch, 4, 2, 1)), nn.LeakyReLU(0.2, inplace=True)]
        c = ch
        for _ in range(1, n_layers):
            oc = min(c * 2, 512)
            layers += [SN(nn.Conv2d(c, oc, 4, 2, 1)), nn.InstanceNorm2d(oc, affine=True),
                       nn.LeakyReLU(0.2, inplace=True)]
            c = oc
        oc = min(c * 2, 512)
        layers += [SN(nn.Conv2d(c, oc, 4, 1, 1)), nn.InstanceNorm2d(oc, affine=True),
                   nn.LeakyReLU(0.2, inplace=True), SN(nn.Conv2d(oc, 1, 4, 1, 1))]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # (B, 1, h, w) patch logits


def d_hinge_loss(real_logits, recon_logits, samp_logits):
    """Discriminator hinge loss: push real >+1, fakes (recon + prior samples) <-1."""
    return (F.relu(1.0 - real_logits).mean()
            + 0.5 * (F.relu(1.0 + recon_logits).mean() + F.relu(1.0 + samp_logits).mean()))


def g_hinge_loss(recon_logits, samp_logits):
    """Generator adversarial loss: maximize the discriminator's score on both fakes."""
    return -recon_logits.mean() - samp_logits.mean()


def _log_normal(z, mu, logvar):
    """Elementwise log N(z; mu, exp(logvar))."""
    return -0.5 * (_LOG2PI + logvar + (z - mu) ** 2 * torch.exp(-logvar))


def tc_vae_loss(recon, x, mu, logvar, z, beta: float, dataset_size: int,
                alpha: float = 1.0, gamma: float = 1.0):
    """β-TCVAE loss (Chen et al. 2018): decompose the KL and weight total correlation by β.

    KL(q(z|x)||p(z)) splits into index-code MI + total correlation (TC) + dimension-wise KL;
    β-TCVAE penalizes TC specifically (α=γ=1 by default). q(z) and its marginals are estimated
    with minibatch-weighted sampling over the batch, corrected to `dataset_size`. Returns
    (loss, recon_term, {mi, tc, dwkl}); by construction mi+tc+dwkl == the MC KL exactly.
    """
    recon_term = F.mse_loss(recon, x, reduction="none").flatten(1).sum(1).mean()
    B = z.size(0)

    logqz_condx = _log_normal(z, mu, logvar).sum(1)                       # (B,)
    logpz = _log_normal(z, torch.zeros_like(z), torch.zeros_like(z)).sum(1)

    # log q(z_i | x_j) elementwise for every pair -> (B, B, D)
    mat = _log_normal(z.unsqueeze(1), mu.unsqueeze(0), logvar.unsqueeze(0))
    logMN = math.log(B * dataset_size)
    logqz = torch.logsumexp(mat.sum(2), dim=1) - logMN                   # (B,)
    logqz_prodmarginals = (torch.logsumexp(mat, dim=1) - logMN).sum(1)   # (B,)

    mi = (logqz_condx - logqz).mean()
    tc = (logqz - logqz_prodmarginals).mean()
    dwkl = (logqz_prodmarginals - logpz).mean()
    loss = recon_term + alpha * mi + beta * tc + gamma * dwkl
    return loss, recon_term, {"mi": mi, "tc": tc, "dwkl": dwkl}
