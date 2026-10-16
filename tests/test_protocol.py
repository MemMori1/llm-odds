"""protocol.py：官方形状逐字节对齐（字母表 / system / 模板 / 选项渲染）。"""
import json

from llm_odds import protocol as pr
from llm_odds.errors import BadRequest


def test_letters():
    assert pr.LETTERS == "ABCDEFGHIJKLMNOP"
    assert len(pr.LETTERS) == 16


def test_system_message_golden():
    # 逐字节抄自官方 jevk5/prompt.py —— 改动会让校准温度失准
    assert pr.SYSTEM_MESSAGE == (
        "Apply the supplied criterion to the supplied evidence. "
        "Choose exactly one listed option. "
        "Respond with only its uppercase letter, with no explanation or reasoning.")


def test_template_golden():
    assert pr.CHAT_TEMPLATE == (
        "<|im_start|>system\n{system}<|im_end|>\n"
        "<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_option_lines_prefix():
    pairs = [("slot1", "Murloc 3/3"), ("slot2", "Pirate 2/2")]
    assert pr.option_lines(pairs, prefix=True) == [
        "slot1: Murloc 3/3", "slot2: Pirate 2/2"]
    assert pr.option_lines(pairs, prefix=False) == [
        "Murloc 3/3", "Pirate 2/2"]


def test_build_payload_shape():
    payload = pr.build_payload("ev", "crit", ["a", "b"])
    assert payload == {
        "evidence": "ev",
        "criterion": "crit",
        "options": [{"letter": "A", "description": "a"},
                    {"letter": "B", "description": "b"}],
    }
    assert json.loads(pr.build_user_json("ev", "crit", ["a", "b"])) == payload


def test_render_prompt_roundtrip():
    text = pr.render_prompt("S", "U")
    assert text.startswith("<|im_start|>system\nS<|im_end|>")
    assert "<|im_start|>user\nU<|im_end|>" in text
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_render_prompt_custom_template():
    assert pr.render_prompt("s", "u", "SYS[{system}] USER[{user}]") == \
        "SYS[s] USER[u]"


def test_resolve_template_builtin_and_none():
    assert pr.resolve_template(None) == pr.CHAT_TEMPLATE
    assert pr.resolve_template("") == pr.CHAT_TEMPLATE
    assert pr.resolve_template("qwen3.5-think-off") == pr.CHAT_TEMPLATE


def test_resolve_template_bad():
    try:
        pr.resolve_template("no-such-template")
    except BadRequest:
        pass
    else:
        raise AssertionError("未知模板应当报 BadRequest")
