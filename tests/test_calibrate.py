"""calibrate.py：温度表 / softmax / 缺字母 floor / 截断份额（数值都是实测搬来的）。"""
import math

from llm_odds.calibrate import (ALL_MISSING_FLOOR, CALIBRATION_TEMPERATURE,
                                MISSING_MARGIN, default_temperature,
                                distribute, fill_truncated,
                                resolve_temperature_key)
from llm_odds.errors import BadRequest
from llm_odds.protocol import LETTERS


def test_temperature_table_golden():
    assert CALIBRATION_TEMPERATURE == {"2b": 1.42, "4b": 1.367,
                                       "4b-v0.2": 1.532, "9b": 1.089}
    assert MISSING_MARGIN == 2.0
    assert ALL_MISSING_FLOOR == -10.0


def test_resolve_temperature_key():
    assert resolve_temperature_key("4b-v0.2") == "4b-v0.2"
    # 表里没有 2b-v0.2 → 退到主键 2b
    assert resolve_temperature_key("2B-v0.2") == "2b"
    assert resolve_temperature_key("") == ""
    assert default_temperature("4b-v0.2") == 1.532
    assert default_temperature("9b") == 1.089
    assert default_temperature("unknown-size") == 1.42


def test_distribute_softmax():
    probs, missing, trunc = distribute(["x", "y"], {"A": 0.0, "B": -1.0}, 2.0)
    assert (missing, trunc) == (0, 0)
    e = math.exp(-0.5)  # softmax([0, -1] / 2) = [1, e^-0.5] 归一
    assert abs(probs["x"] - 1 / (1 + e)) < 1e-12
    assert abs(probs["y"] - e / (1 + e)) < 1e-12
    assert abs(sum(probs.values()) - 1.0) < 1e-12


def test_distribute_missing_floor():
    # C 缺 → floor = min(seen) − 2.0 = −5；lps = [−1, −3, −5]
    probs, missing, trunc = distribute(
        ["0", "1", "2"], {"A": -1.0, "B": -3.0}, 1.0)
    assert (missing, trunc) == (1, 0)
    w = [1.0, math.exp(-2), math.exp(-4)]
    total = sum(w)
    for key, wi in zip(("0", "1", "2"), w):
        assert abs(probs[key] - wi / total) < 1e-12


def test_distribute_all_missing_is_uniform():
    probs, missing, _ = distribute(["x", "y"], {}, 2.0)
    assert missing == 2
    assert abs(probs["x"] - 0.5) < 1e-12
    assert abs(probs["y"] - 0.5) < 1e-12


def test_distribute_temperature_sharpens():
    hot, _, _ = distribute(["x", "y"], {"A": 0.0, "B": -1.0}, 4.0)
    cold, _, _ = distribute(["x", "y"], {"A": 0.0, "B": -1.0}, 1.0)
    assert cold["x"] > hot["x"] > 0.5


def test_distribute_rejects_bad_temperature():
    for t in (0, -1.0, None):
        try:
            distribute(["x"], {"A": 0.0}, t)
        except BadRequest:
            continue
        raise AssertionError("temperature=%r 应当报 BadRequest" % (t,))


def test_distribute_rejects_no_ids():
    try:
        distribute([], {"A": 0.0}, 1.0)
    except BadRequest:
        return
    raise AssertionError("空选项应当报 BadRequest")


def test_truncation_share():
    ids = [str(i) for i in range(17)]
    seen = {LETTERS[0]: 0.0}  # 只有 A 现身，其余走 floor
    probs, missing, trunc = distribute(ids, seen, 1.0, max_kept=16)
    assert trunc == 1
    assert missing == 15
    assert abs(sum(probs.values()) - 1.0) < 1e-12
    # 被丢的第 17 个：share = 1/(17+1) = 1/18
    assert abs(probs["16"] - 1.0 / 18) < 1e-12
    # 保留项都乘了 keep = 1 − 1/18；A 在保留项里权重最大
    assert probs["0"] > probs["15"]


def test_fill_truncated_direct():
    probs, trunc = fill_truncated({"a": 0.6, "b": 0.4}, ["a", "b", "c"], 2)
    assert trunc == 1
    assert abs(probs["c"] - 0.25) < 1e-12          # N=3 → share = 1/4
    assert abs(probs["a"] - 0.6 * 0.75) < 1e-12    # keep = 3/4
    assert abs(sum(probs.values()) - 1.0) < 1e-12
    probs2, t2 = fill_truncated({"a": 1.0}, ["a"], 1)
    assert t2 == 0 and probs2 == {"a": 1.0}
