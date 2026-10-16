"""后端接口：把 prompt 变成"答案位置各字母的 logprob"。

模板渲染、概率化、校准全部在 core（protocol.py / calibrate.py）；
后端**只管一次前向怎么发、怎么读**。
"""
from dataclasses import dataclass, field
from typing import Dict

from ..types import Params, Prompt


@dataclass
class LetterResult:
    """一次读字母的结果。

    `logprobs` 的键是服务端给的**原始 token 字符串**（llama-server 上就是
    `"A"`~`"P"`）；core 只从中取 `LETTERS` 里的字母（与 hsbg 读法一致）。
    """
    logprobs: Dict[str, float] = field(default_factory=dict)
    tokens: int = 0
    seconds: float = 0.0
    attempts: int = 1


class Backend:
    """所有后端的公共接口。"""

    name = "base"

    def __init__(self, params: Params) -> None:
        self.p = params

    def health(self) -> bool:
        return False

    def letter_logprobs(self, prompt: Prompt, params: Params) -> LetterResult:
        raise NotImplementedError
