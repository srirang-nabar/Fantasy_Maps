"""Gate tests for the data engine (plan.md Stage 1).

Slow tests (real browser) are marked gate_stage1; run with:
    uv run pytest -m gate_stage1 -q
"""
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fantasy_maps import generate

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FACTORS = {
    "land_fraction", "mountain_fraction_of_land", "coast_fraction_of_land",
    "mean_land_height", "river_count", "total_river_length", "lake_count", "n_cells",
}


@pytest.fixture(scope="module")
def server():
    proc = subprocess.Popen(
        ["python3", "-m", "http.server", "8917", "-d", str(generate.SNAPSHOT)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1)
    yield 8917
    proc.terminate()


def _gen(tmp_path, monkeypatch, server, seeds, shard=0):
    monkeypatch.setattr(generate, "RAW", tmp_path)
    generate.generate_shard(seeds, shard=shard, port=server)
    jsonl = tmp_path / f"factors_shard{shard}.jsonl"
    recs = [json.loads(l) for l in jsonl.read_text().splitlines()]
    return recs, tmp_path


@pytest.mark.gate_stage1
def test_record_schema_and_ranges(tmp_path, monkeypatch, server):
    recs, out = _gen(tmp_path, monkeypatch, server, [777])
    assert len(recs) == 1
    r = recs[0]
    assert REQUIRED_FACTORS <= set(r["factors"])
    f = r["factors"]
    assert 0.0 < f["land_fraction"] < 1.0
    assert 0.0 <= f["mountain_fraction_of_land"] <= 1.0
    assert 0.0 < f["coast_fraction_of_land"] <= 1.0
    assert f["n_cells"] > 1000
    assert r["template"], "template knob must be recorded"
    assert r["snapshot"] == generate.snapshot_hash()
    img = Image.open(out / "maps" / "777.png")
    assert img.size == (generate.OUT_SIZE, generate.OUT_SIZE)


@pytest.mark.gate_stage1
def test_determinism_same_seed(tmp_path, monkeypatch, server):
    recs1, out1 = _gen(tmp_path / "a", monkeypatch, server, [4242])
    recs2, out2 = _gen(tmp_path / "b", monkeypatch, server, [4242])
    assert recs1[0]["factors"] == recs2[0]["factors"], "factors must be exactly deterministic"
    assert recs1[0]["template"] == recs2[0]["template"]
    a = np.asarray(Image.open(out1 / "maps" / "4242.png"), dtype=np.int16)
    b = np.asarray(Image.open(out2 / "maps" / "4242.png"), dtype=np.int16)
    # antialiasing jitter tolerance measured in feasibility: tiny; assert well under it
    frac_diff = (np.abs(a - b).sum(axis=2) > 0).mean()
    assert frac_diff < 0.01, f"pixel jitter {frac_diff:.4%} exceeds tolerance"
    assert np.abs(a - b).max() <= 16


@pytest.mark.gate_stage1
def test_resume_skips_done(tmp_path, monkeypatch, server):
    _gen(tmp_path, monkeypatch, server, [901, 902])
    jsonl = tmp_path / "factors_shard0.jsonl"
    n_before = len(jsonl.read_text().splitlines())
    generate.generate_shard([901, 902, 903], shard=0, port=server)  # only 903 is new
    lines = jsonl.read_text().splitlines()
    assert len(lines) == n_before + 1
    seeds = [json.loads(l)["seed"] for l in lines]
    assert sorted(seeds) == [901, 902, 903]
    assert len(set(seeds)) == 3, "no duplicate records on resume"


def test_done_seeds_tolerates_torn_line(tmp_path):
    j = tmp_path / "factors_shard0.jsonl"
    j.write_text('{"seed": 1, "x": 1}\n{"seed": 2, "x"')  # torn tail
    assert generate.done_seeds(j) == {1}
