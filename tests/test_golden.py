"""examples/ 契约 golden：mock / mock_probs 全确定性，逐字段比对。

这些 JSON 是**跨项目边界**的样例（宿主照着解析），改动输出形状必须先改
expected 文件 —— 防手滑改字段。
"""
import json
from pathlib import Path

from llm_odds import api

EX = Path(__file__).resolve().parent.parent / "examples"
REG = {}


def _load(name):
    return json.loads((EX / name).read_text(encoding="utf-8"))


def _plain(obj):
    return json.loads(json.dumps(obj, ensure_ascii=False))


def test_golden_choice():
    got = _plain(api.ask_item(_load("ask.choice.request.json"), registry=REG))
    assert got == _load("ask.choice.expected.json")


def test_golden_noul():
    got = _plain(api.ask_item(_load("ask.noul.request.json"), registry=REG))
    assert got == _load("ask.noul.expected.json")


def test_golden_score():
    got = _plain(api.ask_item(_load("ask.score.request.json"), registry=REG))
    assert got == _load("ask.score.expected.json")


def test_golden_ask_many():
    req = _load("ask-many.request.json")
    got = _plain(api.ask_many(req.get("items"), params=req.get("params"),
                              registry=REG))
    assert got == _load("ask-many.expected.json")


def test_golden_files_present():
    for name in ("ask.choice", "ask.noul", "ask.score"):
        assert (EX / (name + ".request.json")).exists()
        assert (EX / (name + ".expected.json")).exists()
    assert (EX / "ask-many.request.json").exists()
    assert (EX / "ask-many.expected.json").exists()
