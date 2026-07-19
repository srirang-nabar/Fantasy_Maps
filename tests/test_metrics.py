"""Calibration certificate for the disentanglement metrics (plan.md Stage 3, hard gate).

Synthetic codes with known answers: a permuted copy of the factors is perfectly
disentangled (MIG/DCI near 1); a random rotation is fully entangled (near 0);
codes independent of the factors are uninformative (near 0). If these fail, no
MIG/DCI on a real checkpoint means anything — so they gate the whole stage.
"""
import numpy as np
import pytest

from fantasy_maps import metrics

pytestmark = pytest.mark.gate_stage3

N, K = 5000, 5


def _independent_factors(seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, size=(N, K))


def _random_rotation(dim, seed=1):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    return q


def test_mig_perfect_is_high():
    f = _independent_factors()
    codes = f[:, ::-1].copy()  # a permutation of the factors — perfectly disentangled
    assert metrics.mig(codes, f)["mig"] > 0.7


def test_mig_survives_extra_noise_latents():
    f = _independent_factors()
    rng = np.random.default_rng(2)
    codes = np.concatenate([f, rng.uniform(0, 1, size=(N, 6))], axis=1)  # 5 real + 6 noise latents
    assert metrics.mig(codes, f)["mig"] > 0.6


def test_mig_entangled_is_low():
    f = _independent_factors()
    codes = f @ _random_rotation(K)  # mixes every factor into every latent
    assert metrics.mig(codes, f)["mig"] < 0.25


def test_mig_random_codes_near_zero():
    f = _independent_factors()
    codes = np.random.default_rng(3).uniform(0, 1, size=(N, K))  # unrelated to factors
    assert metrics.mig(codes, f)["mig"] < 0.1


def test_mig_orders_perfect_above_entangled():
    f = _independent_factors()
    perfect = metrics.mig(f.copy(), f)["mig"]
    entangled = metrics.mig(f @ _random_rotation(K), f)["mig"]
    assert perfect > 3 * entangled


def test_dci_perfect_vs_entangled():
    f = _independent_factors()
    perfect = metrics.dci(f.copy(), f)
    entangled = metrics.dci(f @ _random_rotation(K), f)
    assert perfect["disentanglement"] > 0.7
    assert entangled["disentanglement"] < 0.35
    # a rotation is invertible, so both stay predictable (informativeness high) —
    # DCI's disentanglement axis is what separates them, not informativeness.
    assert perfect["informativeness"] > 0.85
    assert entangled["informativeness"] > 0.7


def test_dci_random_codes_uninformative():
    f = _independent_factors()
    codes = np.random.default_rng(4).uniform(0, 1, size=(N, K))
    assert metrics.dci(codes, f)["informativeness"] < 0.2
