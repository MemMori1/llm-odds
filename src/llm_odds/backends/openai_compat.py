"""OpenAI 兼容后端（`POST {url}/chat/completions` + `logprobs`）。

用于将来换 MLX / vLLM / 其它走 OpenAI 协议且**支持 logprobs** 的服务。
⚠️ 各家支持不一：支持则读答案位置 token 的 `top_logprobs`；不支持 →
`logprobs_unavailable`（**不静默降级成生成式** —— 那会把置信度全丢掉）。
"""
import json
import socket
import time
import urllib.error
import urllib.request
from typing import Dict

from ..errors import LogprobsUnavailable, ModelUnavailable, OddsTimeout
from ..protocol import LETTERS
from ..types import Params, Prompt
from .base import Backend, LetterResult


class OpenAICompatBackend(Backend):
    name = "openai"

    def __init__(self, params: Params) -> None:
        super().__init__(params)
        self.url = (params.url or "http://127.0.0.1:1234/v1").rstrip("/")

    def health(self) -> bool:
        try:
            with urllib.request.urlopen(
                    self.url + "/models",
                    timeout=min(self.p.timeout_s, 3.0)) as r:
                return int(getattr(r, "status", 200) or 200) < 400
        except Exception:
            return False

    def letter_logprobs(self, prompt: Prompt, params: Params) -> LetterResult:
        t0 = time.time()
        payload = {
            "model": params.model or "",
            "messages": [{"role": "system", "content": prompt.system},
                         {"role": "user", "content": prompt.user}],
            "max_tokens": 1, "temperature": 0,
            "logprobs": True, "top_logprobs": params.top_k,
        }
        out = self._post("/chat/completions", payload)
        try:
            entries = out["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        except (KeyError, IndexError, TypeError) as e:
            raise LogprobsUnavailable(
                "服务没返回 logprobs（该 OpenAI 兼容服务不支持 top_logprobs，"
                "或 logprobs 字段为 null）：%s" % e) from e
        seen = {}  # type: Dict[str, float]
        for entry in entries or []:
            if not isinstance(entry, dict):
                continue
            tok = str(entry.get("token", "")).strip()
            try:
                lp = float(entry["logprob"])
            except (KeyError, TypeError, ValueError):
                continue
            # chat 服务的 token 有的带前导空格；只把"字母本身"收进分布
            if tok in LETTERS:
                seen[tok] = lp
        usage = out.get("usage") or {}
        tokens = int(usage.get("prompt_tokens")
                     or usage.get("total_tokens") or 0)
        return LetterResult(logprobs=seen, tokens=tokens,
                            seconds=time.time() - t0, attempts=1)

    def _post(self, path: str, payload: dict) -> dict:
        url = self.url + path
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.p.timeout_s) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", "replace")[:200]
            except Exception:
                pass
            if e.code == 400 and "logprob" in body.lower():
                raise LogprobsUnavailable(
                    "服务拒绝了 logprobs 参数：HTTP 400 %s" % body) from e
            raise ModelUnavailable("%s → HTTP %s %s" % (url, e.code, body)) from e
        except (socket.timeout, TimeoutError) as e:
            raise OddsTimeout("请求超时：%s" % url) from e
        except urllib.error.URLError as e:
            if isinstance(getattr(e, "reason", None),
                          (socket.timeout, TimeoutError)):
                raise OddsTimeout("请求超时：%s" % url) from e
            raise ModelUnavailable("连不上 %s：%s" % (url, e)) from e
        except (ValueError, OSError) as e:
            raise ModelUnavailable("响应无法解析：%s" % e) from e
