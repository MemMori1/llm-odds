"""System One 协议：system 消息、chat 模板、选项渲染。

规则逐字来自官方 `jevk5/prompt.py`（Apache-2.0），与 hsbg-helper
的 `provider.py` **逐字节一致** —— 不要在这里"改良"文案：
JevK5 的校准温度就是按这套 prompt 标出来的。

- system：固定一句话，要求"只回选项的大写字母，不要解释"。
- user：一个 JSON：{"evidence", "criterion", "options": [{"letter", "description"}]}
- 模板：Qwen3.5 chat template 关掉 thinking 后**预填空 think 块**：
  `<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n` ← 答案位置的字母正是下一个 token。
"""
import json
from typing import List, Optional, Sequence, Tuple

from .errors import BadRequest

#: 官方 prompt.py 的字母表（16 个 = 单次前向的最大选项数）
LETTERS = "ABCDEFGHIJKLMNOP"

SYSTEM_MESSAGE = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)

CHAT_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)

#: 内置模板名 → 模板串
TEMPLATES = {"qwen3.5-think-off": CHAT_TEMPLATE}


def resolve_template(template: Optional[str]) -> str:
    """把 `params.template` 收敛成模板串：内置名或带 {system}/{user} 的模板。"""
    if not template:
        return CHAT_TEMPLATE
    if template in TEMPLATES:
        return TEMPLATES[template]
    if "{system}" in template and "{user}" in template:
        return template
    raise BadRequest("未知模板：%r（给内置名（%s）或含 {system}/{user} 的模板串）"
                     % (template, "、".join(sorted(TEMPLATES))))


def option_lines(options: Sequence[Tuple[str, str]], prefix: bool = True) -> List[str]:
    """选项文本。`prefix=True` 时补 `"id: "` 前缀（对齐官方 JevBench 映射形状）。"""
    if prefix:
        return ["%s: %s" % (oid, text) for oid, text in options]
    return [text for _, text in options]


def build_payload(evidence: str, criterion: str,
                  lines: Sequence[str]) -> dict:
    """官方 `prompt.messages` 的 user 消息载荷。"""
    return {
        "evidence": evidence,
        "criterion": criterion,
        "options": [{"letter": LETTERS[i], "description": d}
                    for i, d in enumerate(lines)],
    }


def build_user_json(evidence: str, criterion: str,
                    lines: Sequence[str]) -> str:
    return json.dumps(build_payload(evidence, criterion, lines),
                      ensure_ascii=False)


def render_prompt(system: str, user: str, template: Optional[str] = None) -> str:
    """完整 prompt 字符串（含模板），供 `/tokenize` 用。"""
    return resolve_template(template).format(system=system, user=user)
