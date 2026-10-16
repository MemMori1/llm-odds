"""概率化与校准：logprob → 归一化概率分布。

数值全部从 hsbg-helper 实测搬来（2026-10），不要重学：

    p_i = softmax(logp_i / T)         T = 校准温度（profile/params，默认查表）
    logp_i 缺字母 → floor = min(seen) − letter_floor_margin（默认 2.0；全缺 → −10.0）
    选项数 > max_options → 截断：给被丢选项留 1/(N+1) 份额（N=原选项数），其余按比例扣

温度错了概率会整体偏平或偏尖 —— 换模型先过 DESIGN.md 附录 B 的验收清单。
"""
import math
from typing import Dict, Optional, Sequence, Tuple

from .errors import BadRequest
from .protocol import LETTERS

#: 各档位的校准温度（来自官方 gguf.py 文档串与 JevK5-GGUF 卡）
CALIBRATION_TEMPERATURE = {
    "2b": 1.42,
    "4b": 1.367,          # v0.3
    "4b-v0.2": 1.532,     # v0.2 默认
    "9b": 1.089,
}

#: 字母不在 top-k 里时的兜底扣分（官方 MISSING_MARGIN）
MISSING_MARGIN = 2.0

#: 一个字母都没看见时所有选项的兜底 logprob（等价于均匀分布）
ALL_MISSING_FLOOR = -10.0


def resolve_temperature_key(size: str) -> str:
    """把模型档位字符串收敛到校准温度表里真实存在的键。

    `server.size_key()` 会给出 `"4b-v0.2"` 这种带版本号的完整键，
    但表里不是每个版本都单列（2B v0.2 的值就是 `"2b"` 那条）。先试完整键，
    再退到不带版本号的主键，最后交给调用方兜底。
    """
    if not size:
        return ""
    key = size.strip().lower()
    if key in CALIBRATION_TEMPERATURE:
        return key
    head = key.split("-", 1)[0]
    if head in CALIBRATION_TEMPERATURE:
        return head
    return key


def default_temperature(size: str) -> float:
    """按档位查校准温度；查不到退回 2B 的值（1.42）。"""
    return CALIBRATION_TEMPERATURE.get(resolve_temperature_key(size), 1.42)


def distribute(all_ids: Sequence[str], seen: Dict[str, float],
               temperature: float, floor_margin: float = MISSING_MARGIN,
               max_kept: Optional[int] = None
               ) -> Tuple[Dict[str, float], int, int]:
    """把 {字母: logprob} 变成 {选项 id: p}（∑p = 1）。

    返回 (probs, missing, truncated)：
    - `missing`：前 `max_kept` 个选项里有多少个字母不在 top-k 里（用了 floor）
    - `truncated`：有多少个选项没进 prompt（拿到保守份额，不是 0）
    """
    if temperature is None or temperature <= 0:
        raise BadRequest("temperature 必须为正数")
    if not all_ids:
        raise BadRequest("没有选项可分配概率")
    n_kept = len(all_ids) if max_kept is None else max(1, min(int(max_kept), len(all_ids)))
    floor = (min(seen.values()) - floor_margin) if seen else ALL_MISSING_FLOOR
    lps = []
    missing = 0
    for i in range(n_kept):
        letter = LETTERS[i]
        if letter in seen:
            lps.append(float(seen[letter]))
        else:
            lps.append(floor)
            missing += 1
    top = max(lps)
    weights = [math.exp((z - top) / temperature) for z in lps]
    total = sum(weights)
    probs = {all_ids[i]: weights[i] / total for i in range(n_kept)}
    probs, truncated = fill_truncated(probs, all_ids, n_kept)
    return probs, missing, truncated


def fill_truncated(probs: Dict[str, float], all_ids: Sequence[str],
                   n_kept: int) -> Tuple[Dict[str, float], int]:
    """截断料理：给被丢的选项留 1/(N+1) 份额，保留项按比例扣（∑p 恒为 1）。

    ⚠️ 被截断的选项不能给 0 概率（会读成"绝不可能"）。官方用 knockout 分趟，
    这里做保守处理 —— 被丢弃的选项各得 share，保留项统一乘 keep，
    这样分布覆盖全部选项且和仍为 1。
    返回 (probs, truncated)。
    """
    n_total = len(all_ids)
    truncated = n_total - n_kept
    if truncated <= 0:
        return probs, 0
    share = 1.0 / (n_total + 1)
    keep = max(0.0, 1.0 - truncated * share)
    out = {k: v * keep for k, v in probs.items()}
    for i in range(n_kept, n_total):
        out[all_ids[i]] = share
    return out, truncated
