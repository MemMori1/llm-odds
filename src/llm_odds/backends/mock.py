"""确定性假后端：测试 / selftest / 演示。

不给 `mock_probs` 时，按 prompt 文本的 SHA-256 给每个字母生成 logprob
（取值 [−2, 0]，同一输入永远同一输出）。给了 `mock_probs` 时是 api 层的
快捷路（`api._mock_shortcut`），根本不进这里。
"""
import hashlib

from ..protocol import LETTERS
from ..types import Params, Prompt
from .base import Backend, LetterResult


class MockBackend(Backend):
    name = "mock"

    def health(self) -> bool:
        return True

    def letter_logprobs(self, prompt: Prompt, params: Params) -> LetterResult:
        digest = hashlib.sha256(prompt.text.encode("utf-8")).digest()
        lps = {}
        for i, letter in enumerate(LETTERS):
            lps[letter] = -float(digest[i]) / 255.0 * 2.0
        return LetterResult(logprobs=lps, tokens=0, seconds=0.0, attempts=1)
