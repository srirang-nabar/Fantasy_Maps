"""Stage 1 hard-gate tests against the *generated dataset* (plan.md Stage 1).

Browser-free and fast: they read the finished data/raw/ and adjudicate the three
pre-registered dataset checks (HYPOTHESES.md) — completeness, factor coverage,
and the pairwise-correlation ceiling. Run the moment generation completes:

    uv run pytest -m gate_stage1_data -q

The whole module skips if nothing has been generated yet, so it is safe to run
early. Once you claim generation is done, the completeness test is what proves it.
"""
import pytest

from fantasy_maps import audit

pytestmark = pytest.mark.gate_stage1_data


@pytest.fixture(scope="module")
def data():
    a = audit.audit_dataset()
    if a.n_records == 0:
        pytest.skip("no dataset generated yet (data/raw is empty)")
    return a


def test_completeness(data):
    """One record per expected seed, one correctly-sized PNG per record, no dupes."""
    assert not data.duplicate_seeds, (
        f"{len(data.duplicate_seeds)} seed(s) recorded more than once, "
        f"e.g. {data.duplicate_seeds[:10]}"
    )
    assert not data.unexpected_seeds, (
        f"{len(data.unexpected_seeds)} seed(s) outside "
        f"[{data.config['seed_start']}, {data.config['seed_end']}], "
        f"e.g. {data.unexpected_seeds[:10]}"
    )
    assert not data.missing_seeds, (
        f"{len(data.missing_seeds)} expected seed(s) not generated, "
        f"e.g. {data.missing_seeds[:10]}"
    )
    assert not data.missing_images, (
        f"{len(data.missing_images)} record(s) have no PNG, "
        f"e.g. {data.missing_images[:10]}"
    )
    assert not data.orphan_images, (
        f"{len(data.orphan_images)} PNG(s) have no record, "
        f"e.g. {data.orphan_images[:10]}"
    )
    assert not data.wrong_size_images, (
        f"{len(data.wrong_size_images)} PNG(s) are not "
        f"{data.config['out_size']}x{data.config['out_size']}, "
        f"e.g. {data.wrong_size_images[:10]}"
    )


def test_factor_coverage(data):
    """Every ground-truth factor must span a real range (no degenerate columns)."""
    bad = {name: c for name, c in data.coverage.items() if not c["ok"]}
    assert not bad, f"degenerate factor(s): {bad}"


def test_factor_correlation_ceiling(data):
    """Pre-registered ceiling: pairwise |rho| between factors <= max_abs_correlation.

    Exceeding it means the factor set is entangled and must be redesigned BEFORE
    training — record the redesign as an amendment in HYPOTHESES.md.
    """
    a, b, rho = data.worst_pair
    ceiling = data.config["gates"]["max_abs_correlation"]
    assert data.max_abs_offdiag <= ceiling, (
        f"|rho|={data.max_abs_offdiag:.3f} between '{a}' and '{b}' "
        f"(rho={rho:+.3f}) exceeds ceiling {ceiling}; redesign the factor set "
        f"before training and amend HYPOTHESES.md"
    )
