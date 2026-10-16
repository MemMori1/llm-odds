"""backends：假 urlopen 桩喂"服务端原始响应"（离线，不碰网络）。"""
import io
import json
import socket
import urllib.error
import urllib.parse
import urllib.request

from llm_odds import backends as be
from llm_odds.backends.base import LetterResult
from llm_odds.backends.llama_server import LlamaServerBackend
from llm_odds.backends.mock import MockBackend
from llm_odds.backends.openai_compat import OpenAICompatBackend
from llm_odds.errors import BadRequest, LogprobsUnavailable, ModelUnavailable, OddsTimeout
from llm_odds.protocol import (LETTERS, SYSTEM_MESSAGE, build_user_json,
                               render_prompt)
from llm_odds.types import Params, Prompt

_ORIG_URLOPEN = urllib.request.urlopen


class _FakeResp:
    def __init__(self, payload, status=200):
        self._data = json.dumps(payload).encode()
        self.status = status

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Stub:
    """按路径配桩：dict → 200 响应；list → 依次取；字符串 → 网络错。"""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def __call__(self, req, timeout=None):
        url = getattr(req, "full_url", str(req))
        path = urllib.parse.urlsplit(url).path
        data = getattr(req, "data", None)
        self.calls.append({"path": path, "url": url,
                           "body": json.loads(data.decode()) if data else None})
        item = self.routes.get(path)
        if item is None:
            raise AssertionError("没配桩的路径：%s" % path)
        if isinstance(item, list):
            act = item.pop(0)
        else:
            act = item
        if callable(act):
            return act(req)
        if isinstance(act, dict):
            return _FakeResp(act)
        if act == "urlerror":
            raise urllib.error.URLError("stub 连不上")
        if act == "timeout":
            raise socket.timeout("stub 超时")
        if isinstance(act, tuple) and act[0] == "http":
            raise urllib.error.HTTPError(url, act[1], "stub", {},
                                         io.BytesIO(act[2].encode()))
        raise AssertionError("桩写法不对：%r" % (act,))


def _prompt():
    system = SYSTEM_MESSAGE
    user = build_user_json("evidence", "criterion", ["a: x", "b: y"])
    return Prompt(system=system, user=user,
                  text=render_prompt(system, user))


def _with_stub(routes, fn):
    stub = _Stub(routes)
    urllib.request.urlopen = stub
    try:
        return fn(), stub
    finally:
        urllib.request.urlopen = _ORIG_URLOPEN


def _llama_params(**kw):
    base = {"backend": "llama-server", "url": "http://127.0.0.1:8090",
            "top_k": 16, "timeout_s": 1.0, "retries": 0}
    base.update(kw)
    return Params(**base)


_OK_TOKENIZE = {"tokens": [1, 2, 3]}
_OK_COMPLETION = {
    "completion_probabilities": [{
        "top_logprobs": [{"token": "A", "logprob": -0.1},
                         {"token": "B", "logprob": -2.0},
                         {"token": "E", "logprob": -9.0}],
    }],
    "tokens_evaluated": 123,
}


def test_llama_read_flow_payloads():
    def go():
        p = _llama_params()
        b = LlamaServerBackend(p)
        return b.letter_logprobs(_prompt(), p)

    res, stub = _with_stub({"/tokenize": _OK_TOKENIZE,
                            "/completion": _OK_COMPLETION}, go)
    assert res.logprobs == {"A": -0.1, "B": -2.0, "E": -9.0}
    assert res.tokens == 123
    assert res.attempts == 1
    assert [c["path"] for c in stub.calls] == ["/tokenize", "/completion"]
    tok = stub.calls[0]["body"]
    assert tok["add_special"] is False and tok["parse_special"] is True
    comp = stub.calls[1]["body"]
    assert comp["prompt"] == [1, 2, 3]          # 必须先 tokenize 再喂 id
    assert comp["n_predict"] == 1
    assert comp["n_probs"] == 16
    assert comp["temperature"] == 0
    assert comp["cache_prompt"] is False


def test_llama_missing_completion_probabilities():
    def go():
        p = _llama_params()
        return LlamaServerBackend(p).letter_logprobs(_prompt(), p)

    try:
        _with_stub({"/tokenize": _OK_TOKENIZE,
                    "/completion": {"content": "A"}}, go)
    except LogprobsUnavailable as e:
        assert "completion_probabilities" in str(e)
        return
    raise AssertionError("缺 completion_probabilities 应当报 logprobs_unavailable")


def test_llama_tokenize_empty():
    def go():
        p = _llama_params()
        return LlamaServerBackend(p).letter_logprobs(_prompt(), p)

    try:
        _with_stub({"/tokenize": {"tokens": []}}, go)
    except ModelUnavailable as e:
        assert "tokenize" in str(e)
        return
    raise AssertionError("空 tokenize 应当报 model_unavailable")


def test_llama_health_404_is_false_not_raise():
    def go():
        p = _llama_params()
        return LlamaServerBackend(p).health()

    ok, stub = _with_stub(
        {"/health": ("http", 404, "not found")}, go)
    assert ok is False
    assert stub.calls[0]["body"] is None  # health 必须是 GET


def test_llama_health_ok():
    def go():
        p = _llama_params()
        return LlamaServerBackend(p).health()

    ok, _ = _with_stub({"/health": {"status": "ok"}}, go)
    assert ok is True


def test_llama_http_500_reports_code():
    def go():
        p = _llama_params()
        return LlamaServerBackend(p).letter_logprobs(_prompt(), p)

    try:
        _with_stub({"/tokenize": _OK_TOKENIZE,
                    "/completion": ("http", 500, '{"error":"boom"}')}, go)
    except ModelUnavailable as e:
        assert "HTTP 500" in str(e) and "boom" in str(e)
        return
    raise AssertionError("HTTP 500 应当报 model_unavailable")


def test_llama_retry_then_success():
    def go():
        p = _llama_params(retries=1)
        return LlamaServerBackend(p).letter_logprobs(_prompt(), p)

    res, stub = _with_stub({"/tokenize": _OK_TOKENIZE,
                            "/completion": ["urlerror", _OK_COMPLETION]}, go)
    assert res.attempts == 2
    assert [c["path"] for c in stub.calls].count("/completion") == 2


def test_llama_net_error_exhausts_retries():
    def go():
        p = _llama_params(retries=1)
        return LlamaServerBackend(p).letter_logprobs(_prompt(), p)

    try:
        _with_stub({"/tokenize": _OK_TOKENIZE,
                    "/completion": "urlerror"}, go)
    except ModelUnavailable as e:
        assert "连不上" in str(e)
        return
    raise AssertionError("连不上应当报 model_unavailable")


def test_llama_timeout_is_odds_timeout():
    def go():
        p = _llama_params()
        return LlamaServerBackend(p).letter_logprobs(_prompt(), p)

    try:
        _with_stub({"/tokenize": _OK_TOKENIZE,
                    "/completion": "timeout"}, go)
    except OddsTimeout:
        return
    raise AssertionError("超时应当报 timeout")


def test_mock_backend_deterministic():
    p = Params(backend="mock")
    b = MockBackend(p)
    r1 = b.letter_logprobs(_prompt(), p)
    r2 = b.letter_logprobs(_prompt(), p)
    assert r1.logprobs == r2.logprobs
    assert set(r1.logprobs) == set(LETTERS)
    assert all(-2.0 <= v <= 0.0 for v in r1.logprobs.values())
    other = Prompt(system="s", user="u", text="另一段 prompt")
    assert b.letter_logprobs(other, p).logprobs != r1.logprobs
    assert b.health() is True


def test_openai_read_flow():
    routes = {"/v1/chat/completions": {
        "choices": [{"logprobs": {"content": [{"top_logprobs": [
            {"token": "A", "logprob": -0.5},
            {"token": " B", "logprob": -1.0},   # 前导空格 → 归一成 B
            {"token": "z", "logprob": -0.2},    # 非字母 → 丢弃
        ]}]}}],
        "usage": {"prompt_tokens": 42},
    }}

    def go():
        p = Params(backend="openai", url="http://127.0.0.1:1234/v1",
                   model="qwen3-8b", top_k=20, timeout_s=1.0)
        return OpenAICompatBackend(p).letter_logprobs(_prompt(), p), p

    (res, p), stub = _with_stub(routes, go)
    assert res.logprobs == {"A": -0.5, "B": -1.0}
    assert res.tokens == 42
    body = stub.calls[0]["body"]
    assert body["model"] == "qwen3-8b"
    assert body["logprobs"] is True and body["top_logprobs"] == 20
    assert body["messages"][0]["role"] == "system"


def test_openai_no_logprobs():
    routes = {"/v1/chat/completions": {"choices": [{"logprobs": None}]}}

    def go():
        p = Params(backend="openai", url="http://127.0.0.1:1234/v1",
                   timeout_s=1.0)
        return OpenAICompatBackend(p).letter_logprobs(_prompt(), p)

    try:
        _with_stub(routes, go)
    except LogprobsUnavailable:
        return
    raise AssertionError("logprobs 为 null 应当报 logprobs_unavailable")


def test_openai_400_logprob_rejected():
    routes = {"/v1/chat/completions":
              ("http", 400, '{"error":"logprobs is not supported"}')}

    def go():
        p = Params(backend="openai", url="http://127.0.0.1:1234/v1",
                   timeout_s=1.0)
        return OpenAICompatBackend(p).letter_logprobs(_prompt(), p)

    try:
        _with_stub(routes, go)
    except LogprobsUnavailable:
        return
    raise AssertionError("400 + logprob 字样应当报 logprobs_unavailable")


def test_make_backend_registry():
    assert set(be.BACKENDS) == {"llama-server", "openai", "mock"}
    assert isinstance(be.make_backend(Params(backend="mock")), MockBackend)
    try:
        be.make_backend(Params(backend="nope"))
    except BadRequest:
        return
    raise AssertionError("未知后端应当报 BadRequest")
