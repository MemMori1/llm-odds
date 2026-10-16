"""核心数据类型：提问、回答、生效参数、渲染好的 prompt。

与 hsbg-helper 的 `provider.py` 同源（从那里抽取）。唯一差异：
`Question.options()` 返回**不带前缀**的裸文本，`"id: "` 前缀由
`protocol.option_lines()` 按 `option_prefix` 参数统一添加。
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .protocol import LETTERS


@dataclass
class Question:
    """一个类型化的提问（qtype: choice / noul / score）。"""
    qtype: str
    instructions: str
    criteria: Sequence[str] = ()
    key: str = ""

    def options(self) -> List[Tuple[str, str]]:
        """(选项 id, 裸文本)。前缀由 `protocol.option_lines()` 统一加。"""
        if self.qtype == "noul":
            return [("true", "The proposition is true."),
                    ("false", "The proposition is false.")]
        if self.qtype == "choice":
            crit = self.criteria
            if isinstance(crit, dict):
                return [(k, v or k) for k, v in crit.items()]
            return [(str(c), str(c)) for c in crit]
        return [(str(i), lvl) for i, lvl in enumerate(self.criteria)]


@dataclass
class Answer:
    """一个类型化回答。"""
    key: str
    qtype: str
    confidence: float
    probabilities: Dict[str, float] = field(default_factory=dict)
    choice: Optional[str] = None
    noul: Optional[float] = None
    score: Optional[float] = None
    input_tokens: int = 0
    seconds: float = 0.0
    #: choice 题选中项的字母（A/B/C…）
    letter: str = ""
    #: 选项超过 max_options 被截断时的数量
    truncated: int = 0

    def describe(self) -> str:
        if self.qtype == "noul":
            return "yes" if (self.noul or 0) >= 0.5 else "no"
        if self.qtype == "choice":
            return self.choice or "?"
        if self.score is not None:
            return "%.2f" % self.score
        return "?"


def answer_from_probs(question: Question, probs: Dict[str, float],
                      tokens: int = 0, seconds: float = 0.0) -> Answer:
    """官方 `prompt.answer` 的回答形态。"""
    if not probs:
        raise ValueError("空的概率分布")
    a = Answer(key=question.key or question.instructions,
               qtype=question.qtype,
               confidence=max(probs.values()),
               probabilities=dict(probs), input_tokens=tokens, seconds=seconds)
    if question.qtype == "noul":
        a.noul = probs.get("true", 0.0)
    elif question.qtype == "choice":
        a.choice = max(probs, key=probs.get)
        opts = [k for k, _ in question.options()]
        if a.choice in opts:
            a.letter = LETTERS[opts.index(a.choice)]
    else:
        a.score = sum(int(k) * v for k, v in probs.items())
    return a


@dataclass
class Params:
    """一次提问的生效参数（请求 > profile > 内置默认，三层合并后的结果）。"""
    backend: str = "llama-server"
    url: str = ""
    model: str = ""
    profile: str = ""
    temperature: Optional[float] = None
    top_k: int = 40
    timeout_s: float = 60.0
    retries: int = 1
    max_options: int = len(LETTERS)
    option_prefix: bool = True
    include_raw: bool = False
    template: str = "qwen3.5-think-off"
    system: Optional[str] = None
    letter_floor_margin: float = 2.0
    cache_ttl_s: float = 0.0
    cache_prompt: bool = False
    mock_probs: Optional[Dict[str, float]] = None


@dataclass
class Prompt:
    """渲染好的 prompt：text 给 completion 类后端，system/user 给 chat 类后端。"""
    system: str
    user: str
    text: str
