"""Stage 2 model/training-path sanity tests (plan.md Stage 2). CPU, no GPU."""
import pytest
import torch

from fantasy_maps import models

pytestmark = pytest.mark.gate_stage2


@pytest.mark.parametrize("img_size", [64, 128])
def test_forward_shapes(img_size):
    m = models.ConvVAE(img_size, latent_dim=16, ch=16)
    x = torch.rand(2, 3, img_size, img_size)
    recon, mu, logvar, z = m(x)
    assert recon.shape == x.shape
    assert mu.shape == logvar.shape == z.shape == (2, 16)
    assert torch.all((recon >= 0) & (recon <= 1)), "sigmoid decoder output must be in [0,1]"


def test_elbo_decomposition_sums_exactly():
    m = models.ConvVAE(64, 16, 16)
    x = torch.rand(4, 3, 64, 64)
    recon, mu, logvar, _z = m(x)
    for beta in (1.0, 4.0):
        loss, recon_t, kl_t = models.vae_loss(recon, x, mu, logvar, beta)
        assert torch.isclose(loss, recon_t + beta * kl_t, atol=1e-6), "loss must equal recon + beta*KL"


def test_tcvae_decomposition_sums_to_mc_kl():
    """β-TCVAE's mi+tc+dwkl must equal the Monte-Carlo KL of the sampled z exactly,
    and with α=β=γ=1 the total loss must equal recon + that KL."""
    torch.manual_seed(0)
    m = models.ConvVAE(64, 16, 16)
    x = torch.rand(32, 3, 64, 64)
    recon, mu, logvar, z = m(x)
    loss, recon_t, parts = models.tc_vae_loss(recon, x, mu, logvar, z, beta=1.0, dataset_size=10000)
    mc_kl = (models._log_normal(z, mu, logvar).sum(1)
             - models._log_normal(z, torch.zeros_like(z), torch.zeros_like(z)).sum(1)).mean()
    assert torch.isclose(parts["mi"] + parts["tc"] + parts["dwkl"], mc_kl, atol=1e-4)
    assert torch.isclose(loss, recon_t + parts["mi"] + parts["tc"] + parts["dwkl"], atol=1e-4)
    assert parts["tc"].item() > -1.0  # sanity: TC is a finite, well-scaled quantity


def test_reparameterize_deterministic_in_eval():
    m = models.ConvVAE(64, 8, 16).eval()
    mu = torch.randn(3, 8)
    logvar = torch.randn(3, 8)
    assert torch.equal(m.reparameterize(mu, logvar), mu), "eval mode must use the mean, no sampling"


def _structured_batch(n=4, size=64):
    """Smooth low-frequency images — representative of maps and cleanly overfittable
    (unlike incompressible random noise)."""
    lin = torch.linspace(0, 1, size)
    gy, gx = torch.meshgrid(lin, lin, indexing="ij")
    imgs = []
    for k in range(n):
        r = 0.5 + 0.5 * torch.sin((k + 1) * 3.14159 * gx)
        g = 0.5 + 0.5 * torch.cos((k + 1) * 3.14159 * gy)
        b = (gx + gy) / 2
        imgs.append(torch.stack([r, g, b]))
    return torch.stack(imgs)


def test_overfit_tiny_batch():
    torch.manual_seed(0)
    m = models.ConvVAE(64, 16, 16)
    x = _structured_batch()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    m.train()
    first = None
    for step in range(300):
        recon, mu, logvar, _z = m(x)
        loss, recon_t, _kl = models.vae_loss(recon, x, mu, logvar, beta=0.0)
        opt.zero_grad(); loss.backward(); opt.step()
        if step == 0:
            first = recon_t.item()
    assert recon_t.item() < 0.5 * first, f"recon should drop overfitting 4 images ({first:.1f} -> {recon_t.item():.1f})"


def test_checkpoint_roundtrip_reproduces_output(tmp_path):
    torch.manual_seed(1)
    m = models.ConvVAE(64, 16, 16).eval()
    x = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        out1, mu1, _lv, _z = m(x)
    torch.save(m.state_dict(), tmp_path / "m.pt")

    m2 = models.ConvVAE(64, 16, 16)
    m2.load_state_dict(torch.load(tmp_path / "m.pt"))
    m2.eval()
    with torch.no_grad():
        out2, mu2, _lv2, _z2 = m2(x)
    assert torch.allclose(out1, out2, atol=1e-6)
    assert torch.allclose(mu1, mu2, atol=1e-6)
