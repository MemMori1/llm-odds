"""api：契约层 —— 三层覆盖 / mock_probs / 错误码 / 缓存 / 批量 / 模板捕获。"""
import json

from llm_odds import api
from llm_odds.backends.base import LetterResult
from llm_odds.errors import BadRequest
from llm_odds.registry import BUILTIN
from llm_odds.session import Session

REG = {}  # 显式空注册表：测试绝不读用户真实的 models.json
REG_M1 = {"default": "m1",
          "models": {"m1": {"backend": "mock", "top_k": 10,
                            "temperature": 2.0, "size": "4b"},
                     "m2": {"backend": "mock", "gguf":
                            "/tmp/jevk5-4b-v0.2-Q4_K_M.gguf"}}}


def _ask(item, registry=REG, session=None):
    return api.ask_item(item, registry=registry, session=session)


def _choice(**kw):
    item = {"v": 1, "task": "choice", "evidence": "ev",
            "options": ["alpha", "beta"]}
    item.update(kw)
    return item


class _Cap:
    """假后端：捕获 prompt，回固定字母 logprob。"""

    def __init__(self, logprobs=None, raise_exc=None):
        self.logprobs = logprobs if logprobs is not None else {"A": 0.0, "B": -1.0}
        self.raise_exc = raise_exc
        self.prompt = None

    def letter_logprobs(self, prompt, params):
        self.prompt = prompt
        if self.raise_exc:
            raise self.raise_exc
        return LetterResult(logprobs=dict(self.logprobs), tokens=7,
                            seconds=0.05, attempts=1)


def _with_cap(cap, fn):
    orig = api.make_backend
    api.make_backend = lambda p: cap
    try:
        return fn()
    finally:
        api.make_backend = orig


# -- 基本形状 --------------------------------------------------------------

def test_mock_choice_shape():
    r = _ask(_choice(params={"backend": "mock"}))
    assert r["ok"] and r["v"] == 1
    assert abs(sum(r["probs"].values()) - 1) < 1e-9
    assert r["choice"] in ("0", "1")
    assert r["letter"] in ("A", "B")
    assert r["confidence"] == r["probs"][r["choice"]]
    assert r["model"]["backend"] == "mock"
    assert r["metrics"]["attempts"] == 1
    assert isinstance(r["warnings"], list)
    assert r["error"] is None


def test_mock_deterministic():
    a = _ask(_choice(params={"backend": "mock"}))
    b = _ask(_choice(params={"backend": "mock"}))
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_mock_probs_shortcut_exact():
    r = _ask(_choice(options=[{"id": "keep", "text": "x"},
                              {"id": "other", "text": "y"}],
                     params={"backend": "mock",
                             "mock_probs": {"keep": 0.75, "other": 0.25}}))
    assert r["ok"]
    assert r["probs"] == {"keep": 0.75, "other": 0.25}
    assert r["choice"] == "keep" and r["letter"] == "A"
    assert r["metrics"]["attempts"] == 0  # 快捷路没碰后端


def test_mock_probs_missing_id():
    r = _ask(_choice(params={"backend": "mock", "mock_probs": {"0": 1.0}}))
    assert not r["ok"] and r["error"]["code"] == "bad_request"


def test_mock_probs_all_zero():
    r = _ask(_choice(params={"backend": "mock",
                             "mock_probs": {"0": 0, "1": 0}}))
    assert not r["ok"] and r["error"]["code"] == "bad_request"


def test_noul_default_options():
    r = _ask({"v": 1, "task": "noul", "evidence": "e",
              "params": {"backend": "mock",
                         "mock_probs": {"true": 0.75, "false": 0.25}}})
    assert r["ok"]
    assert list(r["probs"]) == ["true", "false"]
    assert r["noul"] == 0.75 and r["choice"] is None and r["letter"] is None


def test_score_expectation():
    r = _ask({"v": 1, "task": "score", "evidence": "e",
              "options": ["l0", "l1", "l2"],
              "params": {"backend": "mock",
                         "mock_probs": {"0": 0.5, "1": 0.25, "2": 0.25}}})
    assert r["ok"]
    assert abs(r["score"] - 0.75) < 1e-12


def test_score_needs_numeric_ids():
    r = _ask({"v": 1, "task": "score", "evidence": "e",
              "options": [{"id": "low", "text": "l"}],
              "params": {"backend": "mock"}})
    assert not r["ok"] and r["error"]["code"] == "bad_request"


def test_choice_letter_mapping():
    r = _ask(_choice(options=[{"id": "p1", "text": "a"},
                              {"id": "p2", "text": "b"},
                              {"id": "p3", "text": "c"}],
                     params={"backend": "mock",
                             "mock_probs": {"p1": 0.1, "p2": 0.7, "p3": 0.2}}))
    assert r["choice"] == "p2" and r["letter"] == "B"


def test_low_signal_warning():
    r = _ask(_choice(params={"backend": "mock",
                             "mock_probs": {"0": 0.51, "1": 0.49}}))
    assert "low_signal" in r["warnings"]


# -- 错误码 ----------------------------------------------------------------

def test_choice_requires_options():
    r = _ask({"v": 1, "task": "choice", "evidence": "e"})
    assert r["error"]["code"] == "no_options"


def test_bad_task():
    r = _ask({"v": 1, "task": "vibes", "evidence": "e"})
    assert not r["ok"] and r["error"]["code"] == "bad_request"
    assert "task" not in r  # 无效 task 不回显


def test_bad_v():
    r = _ask({"v": 2, "task": "choice", "options": ["a"]})
    assert r["error"]["code"] == "bad_request"


def test_bad_params_type():
    r = _ask(_choice(params=["nope"]))
    assert r["error"]["code"] == "bad_request"


def test_bad_options_type():
    r = _ask({"v": 1, "task": "choice", "options": "abcd"})
    assert r["error"]["code"] in ("no_options", "bad_request")


def test_duplicate_option_ids():
    r = _ask(_choice(options=[{"id": "a", "text": "x"},
                              {"id": "a", "text": "y"}]))
    assert r["error"]["code"] == "bad_request"


def test_bad_temperature():
    r = _ask(_choice(params={"backend": "mock", "temperature": 0}))
    assert r["error"]["code"] == "bad_request"


def test_bad_backend():
    r = _ask(_choice(params={"backend": "gpt4-cloud"}))
    assert r["error"]["code"] == "bad_request"


def test_unknown_profile():
    r = _ask(_choice(params={"profile": "nope"}), registry=REG_M1)
    assert r["error"]["code"] == "unknown_profile"


def test_internal_error_structured():
    cap = _Cap(raise_exc=RuntimeError("炸了"))
    r = _with_cap(cap, lambda: _ask(_choice()))
    assert not r["ok"] and r["error"]["code"] == "internal"
    assert "炸了" in r["error"]["message"]


def test_error_shape_minimal():
    r = _ask({"v": 1, "task": "choice", "evidence": "e"})
    assert set(r) <= {"v", "id", "ok", "task", "error"}
    assert r["error"]["code"] and r["error"]["message"]


# -- 三层覆盖与回显 --------------------------------------------------------

def test_profile_default_and_echo():
    r = _ask({"v": 1, "task": "noul", "evidence": "e",
              "params": {"mock_probs": {"true": 1.0, "false": 0.0}}},
             registry=REG_M1)
    assert r["ok"]
    m = r["model"]
    assert m["profile"] == "m1" and m["backend"] == "mock"
    assert m["top_k"] == 10 and m["temperature"] == 2.0 and m["size"] == "4b"


def test_profile_gguf_derives_size_and_temperature():
    r = _ask({"v": 1, "task": "noul", "evidence": "e",
              "params": {"profile": "m2"}},
             registry=REG_M1)
    assert r["ok"]
    m = r["model"]
    assert m["size"] == "4b-v0.2"
    assert m["temperature"] == 1.532
    assert m["name"] == "jevk5-4b-v0.2-Q4_K_M"


def test_request_overrides_profile():
    r = _ask({"v": 1, "task": "noul", "evidence": "e",
              "params": {"temperature": 1.0, "top_k": 5,
                         "mock_probs": {"true": 1.0, "false": 0.0}}},
             registry=REG_M1)
    assert r["model"]["temperature"] == 1.0
    assert r["model"]["top_k"] == 5


def test_builtin_profile_echo():
    r = _ask(_choice(params={"backend": "mock"}))
    assert r["model"] == {"profile": "", "backend": "mock", "name": "mock",
                          "size": "", "temperature": BUILTIN["temperature"]
                          if BUILTIN["temperature"] else 1.42, "top_k": 40}


def test_llama_server_derives_size_from_found_gguf():
    # 没给档位线索 → 学 hsbg，从"会被加载的那个 GGUF"推 size/温度
    orig = api.find_gguf
    api.find_gguf = lambda: "/x/jevk5-4b-v0.2-Q4_K_M.gguf"
    try:
        r = _with_cap(_Cap(), lambda: _ask(_choice()))
    finally:
        api.find_gguf = orig
    assert r["ok"] is True
    m = r["model"]
    assert m["size"] == "4b-v0.2"
    assert m["name"] == "jevk5-4b-v0.2-Q4_K_M"
    assert m["temperature"] == 1.532


def test_mock_backend_does_not_scan_gguf():
    # mock 后端不该去找 GGUF（保持机器无关的确定性）
    calls = []
    orig = api.find_gguf
    api.find_gguf = lambda: calls.append(1) or "/x/jevk5-4b-Q4_K_M.gguf"
    try:
        r = _ask(_choice(params={"backend": "mock"}))
    finally:
        api.find_gguf = orig
    assert r["ok"] is True and calls == []


# -- prompt 渲染（option_prefix / 模板） -----------------------------------

def test_option_prefix_visible_in_prompt():
    cap = _Cap()
    _with_cap(cap, lambda: _ask(_choice()))
    text = cap.prompt.text
    assert "0: alpha" in text and "1: beta" in text
    assert "<|im_start|>assistant\n<think>\n\n</think>\n\n" in text
    assert "Apply the supplied criterion" in text


def test_option_prefix_false():
    cap = _Cap()
    _with_cap(cap, lambda: _ask(_choice(params={"option_prefix": False})))
    assert "0: alpha" not in cap.prompt.text
    assert "alpha" in cap.prompt.text


def test_include_raw():
    cap = _Cap(logprobs={"A": -0.5, "B": -2.0})
    r = _with_cap(cap, lambda: _ask(_choice(params={"include_raw": True})))
    assert r["raw"] == {"logprobs": {"A": -0.5, "B": -2.0}}
    r2 = _with_cap(_Cap(), lambda: _ask(_choice()))
    assert "raw" not in r2


def test_missing_letters_warning():
    cap = _Cap(logprobs={"A": 0.0})  # 两个选项只见到 A
    r = _with_cap(cap, lambda: _ask(_choice()))
    assert r["metrics"]["missing_letters"] == 1
    assert "missing_letters" in r["warnings"]


def test_truncation_warning_and_sum():
    opts = ["o%d" % i for i in range(17)]
    cap = _Cap()
    r = _with_cap(cap, lambda: _ask(_choice(options=opts)))
    assert r["metrics"]["truncated"] == 1
    assert "truncated_options" in r["warnings"]
    assert abs(sum(r["probs"].values()) - 1) < 1e-9
    assert "16: " not in cap.prompt.text  # 第 17 个没进 prompt


def test_elapsed_from_backend_seconds():
    cap = _Cap()  # seconds=0.05
    r = _with_cap(cap, lambda: _ask(_choice()))
    assert r["metrics"]["elapsed_ms"] == 50.0
    assert r["metrics"]["tokens_evaluated"] == 7


# -- 缓存 ------------------------------------------------------------------

def test_cache_miss_then_hit():
    s = Session(default_ttl=5)
    item = _choice(params={"backend": "mock"})
    r1 = _ask(item, session=s)
    assert r1["metrics"]["cache"] == "miss"
    r2 = _ask(item, session=s)
    assert r2["metrics"]["cache"] == "hit"
    assert r2["probs"] == r1["probs"]


def test_cache_off_by_default():
    s = Session()
    item = _choice(params={"backend": "mock"})
    assert _ask(item, session=s)["metrics"]["cache"] == "off"
    assert _ask(item, session=s)["metrics"]["cache"] == "off"


def test_cache_ttl_param_enables():
    s = Session()
    item = _choice(params={"backend": "mock", "cache_ttl_s": 5})
    assert _ask(item, session=s)["metrics"]["cache"] == "miss"
    assert _ask(item, session=s)["metrics"]["cache"] == "hit"


# -- 批量 ------------------------------------------------------------------

def test_ask_many_merges_global_params():
    resp = api.ask_many([
        {"v": 1, "task": "choice", "id": "a", "options": ["x", "y"]},
        {"v": 1, "task": "choice", "id": "b", "options": ["x", "y"],
         "params": {"mock_probs": {"0": 0.9, "1": 0.1}}},
    ], params={"backend": "mock"}, registry=REG)
    r0, r1 = resp["results"]
    assert r0["ok"] and r1["ok"]
    assert r0["id"] == "a" and r1["id"] == "b"
    assert r1["probs"]["0"] == 0.9


def test_ask_many_empty_raises():
    try:
        api.ask_many([], registry=REG)
    except BadRequest:
        return
    raise AssertionError("空 items 应当抛 BadRequest")


def test_ask_many_item_error_isolated():
    resp = api.ask_many([
        {"v": 1, "task": "choice", "options": ["x", "y"],
         "params": {"backend": "mock"}},
        {"v": 1, "task": "nope"},
    ], registry=REG)
    assert resp["results"][0]["ok"] is True
    assert resp["results"][1]["ok"] is False
    assert resp["results"][1]["error"]["code"] == "bad_request"


# -- 库形态与 models_info ---------------------------------------------------

def test_ask_library_signature():
    r = api.ask(evidence="e", options=["a", "b"],
                params={"backend": "mock"}, registry=REG)
    assert r.ok is True and r["ok"] is True   # 属性与键都能取
    assert r.choice in ("0", "1")


def test_models_info_check_true_mock_only():
    info = api.models_info(registry={"default": "m1",
                                     "models": {"m1": {"backend": "mock"}}},
                           check=True)
    assert info["default"] == "m1"
    assert info["models"]["m1"]["available"] is True
    assert info["default_available"] is True
    assert "available" not in info["builtin"]  # 有默认档位就不白探内建


def test_models_info_check_false_no_probe():
    info = api.models_info(registry={"models": {"m1": {"backend": "mock"}}},
                           check=False)
    assert info["default"] == ""
    assert "available" not in info["models"]["m1"]
    assert info["default_available"] is False


def test_available_mock_true():
    assert api._available({"backend": "mock"}) is True
