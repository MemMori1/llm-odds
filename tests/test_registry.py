"""registry.py：注册表读取 / 三层合并 / 温度解析（全部离线）。"""
import json
import os
import tempfile
from pathlib import Path

from llm_odds import registry as rg
from llm_odds.errors import BadRequest, UnknownProfile


def _write(d: str, name: str, text: str) -> Path:
    p = Path(d) / name
    p.write_text(text, encoding="utf-8")
    return p


def test_load_registry_missing_is_empty():
    assert rg.load_registry("/nonexistent/definitely/nope.json") == {}


def test_load_registry_good():
    with tempfile.TemporaryDirectory() as d:
        p = _write(d, "models.json", json.dumps(
            {"default": "m", "models": {"m": {"backend": "mock"}}}))
        reg = rg.load_registry(str(p))
        assert reg["default"] == "m"


def test_load_registry_bad_json():
    with tempfile.TemporaryDirectory() as d:
        p = _write(d, "bad.json", "{oops")
        try:
            rg.load_registry(str(p))
        except BadRequest:
            return
        raise AssertionError("坏 JSON 应当报 BadRequest")


def test_load_registry_wrong_shapes():
    for text in ("[]", '{"models": 3}', '{"models": []}'):
        with tempfile.TemporaryDirectory() as d:
            p = _write(d, "models.json", text)
            try:
                rg.load_registry(str(p))
            except BadRequest:
                continue
            raise AssertionError("%s 应当报 BadRequest" % text)


def test_registry_path_env_and_explicit():
    old = os.environ.get(rg.ENV_VAR)
    os.environ[rg.ENV_VAR] = "/tmp/llm-odds-test-env.json"
    try:
        assert str(rg.registry_path()) == "/tmp/llm-odds-test-env.json"
        assert str(rg.registry_path("/tmp/explicit.json")) == "/tmp/explicit.json"
    finally:
        if old is None:
            os.environ.pop(rg.ENV_VAR, None)
        else:
            os.environ[rg.ENV_VAR] = old


def test_get_profile_builtin_is_copy():
    prof = rg.get_profile({}, None)
    prof["top_k"] = 999
    assert rg.BUILTIN["top_k"] == 40
    assert rg.get_profile({}, None)["top_k"] == 40


def test_get_profile_merge_skips_none():
    reg = {"models": {"m": {"backend": "mock", "top_k": 20,
                            "timeout_s": None}}}
    prof = rg.get_profile(reg, "m")
    assert prof["backend"] == "mock"
    assert prof["top_k"] == 20
    assert prof["timeout_s"] == 60.0  # None 不覆盖内建


def test_get_profile_unknown():
    try:
        rg.get_profile({"models": {}}, "nope")
    except UnknownProfile:
        return
    raise AssertionError("未知档位应当报 UnknownProfile")


def test_get_profile_entry_not_object():
    try:
        rg.get_profile({"models": {"m": 3}}, "m")
    except BadRequest:
        return
    raise AssertionError("条目不是对象应当报 BadRequest")


def test_resolve_temperature_paths():
    assert rg.resolve_temperature({"temperature": 2.5}) == 2.5
    assert rg.resolve_temperature({"size": "9b"}) == 1.089
    assert rg.resolve_temperature(
        {"gguf": "/x/jevk5-4b-v0.2-Q4_K_M.gguf"}) == 1.532
    assert rg.resolve_temperature({}) == 1.42


def test_default_name():
    assert rg.default_name({"default": "m"}) == "m"
    assert rg.default_name({}) is None
    assert rg.default_name({"default": ""}) is None
