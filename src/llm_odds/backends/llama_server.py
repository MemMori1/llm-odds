"""llama.cpp `llama-server` 后端：JevK5 的真 logprob 读法（主通道）。

协议要点（与 hsbg-helper 实测逐条一致，别改）：

- ⚠️ 必须先把 prompt 用 `/tokenize` 转成 token id 再喂 `/completion`：
  直接把字符串当 prompt 传，`<|im_start|>` 会被拆成 6 个普通字符，模板结构全废
  （官方注释原话）。
- `/completion`：`n_predict=1`、`n_probs=top_k`、`temperature=0`（读概率用 0，
  校准温度在 core 应用）、`cache_prompt=false`（默认；作为参数可开）。
- 读 `completion_probabilities[0].top_logprobs`；拿不到 → `logprobs_unavailable`
  （LM Studio 自带 HTTP 服务就是这一类 —— 它不支持 `n_probs`）。
- ⚠️ `/health` **只认 GET**（POST 404），且 `HTTPError` 必须先于 `URLError` catch
  （顺序写反会把 404 报成"连不上"）。
"""
import json
import socket
import time
import urllib.error
import urllib.request
from typing import Dict, Optional

from ..errors import LogprobsUnavailable, ModelUnavailable, OddsTimeout
from ..types import Params, Prompt
from .base import Backend, LetterResult


class _NetError(Exception):
    """连接层的可重试错误（连不上 / 超时）。"""

    def __init__(self, msg: str, timed_out: bool = False) -> None:
        super().__init__(msg)
        self.timed_out = timed_out


def _is_timeout(exc: object) -> bool:
    return isinstance(exc, (socket.timeout, TimeoutError))


class LlamaServerBackend(Backend):
    name = "llama-server"

    def __init__(self, params: Params) -> None:
        super().__init__(params)
        self.url = (params.url or "http://127.0.0.1:8090").rstrip("/")

    # -- HTTP ----------------------------------------------------------

    def _request(self, path: str, payload: Optional[dict] = None,
                 timeout: Optional[float] = None) -> dict:
        """GET（payload=None）或 POST（payload=dict）。

        ⚠️ `HTTPError` 必须在 `URLError` **之前** catch —— 前者是后者的子类，
        顺序写反会把 404 报成"连不上"。
        """
        url = self.url + path
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(
                    req, timeout=timeout or self.p.timeout_s) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", "replace")[:200]
            except Exception:
                pass
            raise ModelUnavailable("%s → HTTP %s %s" % (url, e.code, body)) from e
        except (socket.timeout, TimeoutError) as e:
            raise _NetError("请求超时：%s" % url, timed_out=True) from e
        except urllib.error.URLError as e:
            # 连接超时会被包在 URLError.reason 里 —— 也归"超时"
            raise _NetError("连不上 %s：%s" % (url, e),
                            timed_out=_is_timeout(getattr(e, "reason", None))) from e
        except (ValueError, OSError) as e:
            raise ModelUnavailable("响应无法解析：%s" % e) from e

    def _post(self, path: str, payload: dict) -> dict:
        return self._request(path, payload)

    def health(self) -> bool:
        """`/health` 只认 GET（POST 会 404）。"""
        try:
            d = self._request("/health", timeout=min(self.p.timeout_s, 3.0))
        except (ModelUnavailable, _NetError):
            return False
        return d.get("status") == "ok"

    # -- 读字母 --------------------------------------------------------

    def letter_logprobs(self, prompt: Prompt, params: Params) -> LetterResult:
        attempts = 0
        while True:
            attempts += 1
            try:
                return self._read_once(prompt, attempts)
            except _NetError as e:
                if attempts >= params.retries + 1:
                    if e.timed_out:
                        raise OddsTimeout(str(e)) from e
                    raise ModelUnavailable(str(e)) from e

    def _read_once(self, prompt: Prompt, attempts: int) -> LetterResult:
        t0 = time.time()
        # ⚠️ 必须先 tokenize：字符串直传会把 <|im_start|> 拆成普通字符
        toks = self._post("/tokenize", {
            "content": prompt.text, "add_special": False, "parse_special": True,
        }).get("tokens")
        if not toks:
            raise ModelUnavailable("tokenize 没返回 token")
        out = self._post("/completion", {
            "prompt": toks, "n_predict": 1, "n_probs": self.p.top_k,
            "temperature": 0, "cache_prompt": bool(self.p.cache_prompt),
        })
        try:
            top = out["completion_probabilities"][0]["top_logprobs"]
        except (KeyError, IndexError, TypeError) as e:
            raise LogprobsUnavailable(
                "服务没返回 completion_probabilities（该服务可能不支持 n_probs，"
                "LM Studio 自带 HTTP 服务就不支持，需要 llama-server）：%s" % e) from e
        seen = {}  # type: Dict[str, float]
        for entry in top or []:
            tok = str(entry.get("token", ""))
            try:
                lp = float(entry["logprob"])
            except (KeyError, TypeError, ValueError):
                continue
            if tok:
                seen[tok] = lp
        return LetterResult(logprobs=seen,
                            tokens=int(out.get("tokens_evaluated", 0) or 0),
                            seconds=time.time() - t0, attempts=attempts)
