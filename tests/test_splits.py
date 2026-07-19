"""Fast, data-free tests for the train/val/eval splits (fantasy_maps.dataset)."""
from fantasy_maps import audit, dataset


def test_splits_partition_all_seeds():
    cfg = audit.load_config()
    s = dataset.make_splits(cfg)
    all_seeds = set(range(cfg["seed_start"], cfg["seed_end"] + 1))
    train, val, evl = set(s["train"]), set(s["val"]), set(s["eval"])
    assert train | val | evl == all_seeds, "splits must cover every seed"
    assert train.isdisjoint(val) and train.isdisjoint(evl) and val.isdisjoint(evl), "splits must be disjoint"


def test_eval_is_the_reserved_tail():
    cfg = audit.load_config()
    s = dataset.make_splits(cfg)
    assert set(s["eval"]) == set(range(cfg["eval_seed_start"], cfg["seed_end"] + 1))
    assert len(s["eval"]) == 3000


def test_val_fraction_roughly_ten_percent():
    s = dataset.make_splits()
    pool = len(s["train"]) + len(s["val"])
    assert 0.08 < len(s["val"]) / pool < 0.12


def test_fingerprint_is_stable():
    a = dataset.split_fingerprint(dataset.make_splits())
    b = dataset.split_fingerprint(dataset.make_splits())
    assert a == b and len(a) == 16
