"""llm-odds —— 本地模型的「校准概率顾问」。

给一段上下文（evidence）和一列选项，拿回**校准过的概率分布**。
模型可换（模型注册表 + 后端接口）、参数可控（请求 > profile > 内置默认
三层覆盖、响应回显）、接入形态三种（库 / CLI / MCP），零第三方依赖。

契约（v1）见 `schemas/`；设计见 `docs/DESIGN.md`。
"""
__version__ = "0.1.0"

from .errors import (BadRequest, LogprobsUnavailable, ModelUnavailable,
                     NoOptions, OddsError, OddsTimeout, UnknownProfile)
from .types import Answer, Params, Prompt, Question, answer_from_probs
from .api import ask, ask_many, models_info

__all__ = [
    "ask", "ask_many", "models_info",
    "Question", "Answer", "answer_from_probs", "Params", "Prompt",
    "OddsError", "BadRequest", "NoOptions", "UnknownProfile",
    "ModelUnavailable", "LogprobsUnavailable", "OddsTimeout",
    "__version__",
]
