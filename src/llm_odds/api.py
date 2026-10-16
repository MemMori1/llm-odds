"""api.ask / ask_many / models_info —— 契约 v1 的唯一实现。

三条硬规则（对应 DESIGN.md §5 与 §7）：

1. **响应永远是 JSON 形状**：失败也是 `{"ok": false, "error": {...}}`，
   不向调用方抛异常（除了 ask_many 的顶层结构错误）。
2. **三层参数覆盖**：BUILTIN ← profile（注册表）← `item["params"]`；
   `model` 块回显最终生效值（可复现、可审计）。
3. **prompt 面只出不进**：这一层只把渲染好的 prompt 交给后端读概率，
   不做任何其它 IO（进程内缓存见 session.py）。
"""
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .backends import BACKENDS, make_backend
from .calibrate import MISSING_MARGIN, distribute, fill_truncated
from .errors import BadRequest, NoOptions, OddsError
from .protocol import LETTERS, SYSTEM_MESSAGE, build_user_json, option_lines, render_prompt
from .registry import (BUILTIN, default_name, get_profile, load_registry,
                       registry_path, resolve_temperature)
from .server import DEFAULT_PORT, find_gguf, size_key
from .session import Session, cache_key
from .types import Params, Prompt

TASKS = ("choice", "noul", "score")

#: criterion 缺省（英文 —— 模型面文本必须英文，ADR-4）
DEFAULT_CRITERION = {
    "choice": "Which option best fits the evidence?",
    "noul": "Is the proposition true?",
    "score": "Which level best matches the evidence?",
}

#: noul 题固定的两个选项（与官方 prompt.py 一致）
NOUL_OPTIONS = [("true", "The proposition is true."),
                ("false", "The proposition is false.")]


class Resp(dict):
    """既是 JSON（可直接 json.dumps）又能点属性取（`resp.choice`）。"""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


def _reg(registry: Optional[dict]) -> dict:
    return registry if registry is not None else load_registry()


def _norm_options(task: str, options) -> List[Tuple[str, str]]:
    if task == "noul":
        return list(NOUL_OPTIONS)
    if options is None:
        raise NoOptions("task=%s 必须给 options" % task)
    if not isinstance(options, list) or not options:
        raise NoOptions("options 必须是非空数组")
    pairs = []  # type: List[Tuple[str, str]]
    for i, o in enumerate(options):
        if isinstance(o, str):
            if not o:
                raise BadRequest("选项文本不能为空（第 %d 个）" % (i + 1))
            pairs.append((str(i), o))
        elif isinstance(o, dict):
            oid, text = o.get("id"), o.get("text")
            if not isinstance(oid, str) or not oid:
                raise BadRequest("选项 id 必须是非空字符串（第 %d 个）" % (i + 1))
            if not isinstance(text, str):
                raise BadRequest("选项 text 必须是字符串（第 %d 个）" % (i + 1))
            pairs.append((oid, text))
        else:
            raise BadRequest("选项只能是字符串或 {id, text}（第 %d 个）" % (i + 1))
    ids = [k for k, _ in pairs]
    if len(set(ids)) != len(ids):
        dup = sorted({k for k in ids if ids.count(k) > 1})
        raise BadRequest("选项 id 重复：%s" % "、".join(dup))
    return pairs


def _effective(registry: dict, item: dict) -> Tuple[Params, dict]:
    """三层覆盖 → (Params, model 回显块)。"""
    req = item.get("params")
    if req is None:
        req = {}
    if not isinstance(req, dict):
        raise BadRequest("params 必须是对象")
    profile_name = str(req.get("profile") or default_name(registry) or "")
    prof = get_profile(registry, profile_name or None)
    merged = dict(prof)
    for k, v in req.items():
        if k != "profile" and v is not None:
            merged[k] = v

    def pick(key, kind, default):
        v = merged.get(key)
        if v is None:
            return default
        if kind is bool:
            if isinstance(v, bool):
                return v
        elif kind is int:
            if isinstance(v, bool):
                pass
            elif isinstance(v, int):
                return v
            elif isinstance(v, float) and v.is_integer():
                return int(v)
        elif kind is float:
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
        elif kind is str:
            if isinstance(v, str):
                return v
        raise BadRequest("参数 %s 类型不对：%r" % (key, v))

    backend = pick("backend", str, "llama-server")
    if backend not in BACKENDS:
        raise BadRequest("未知后端：%r（可选 %s）"
                         % (backend, "、".join(sorted(BACKENDS))))
    top_k = pick("top_k", int, 40)
    if top_k < 2:
        raise BadRequest("top_k 至少 2")
    timeout_s = pick("timeout_s", float, 60.0)
    if timeout_s <= 0:
        raise BadRequest("timeout_s 必须为正数")
    retries = pick("retries", int, 1)
    if retries < 0:
        raise BadRequest("retries 不能为负")
    max_options = pick("max_options", int, len(LETTERS))
    if not 2 <= max_options <= len(LETTERS):
        raise BadRequest("max_options 必须在 2..%d" % len(LETTERS))
    margin = pick("letter_floor_margin", float, MISSING_MARGIN)
    if margin < 0:
        raise BadRequest("letter_floor_margin 不能为负")
    cache_ttl = pick("cache_ttl_s", float, 0.0)
    if cache_ttl < 0:
        raise BadRequest("cache_ttl_s 不能为负")
    url = pick("url", str, "")
    if not url and backend == "llama-server":
        try:
            port = int(merged.get("port") or DEFAULT_PORT)
        except (TypeError, ValueError):
            port = DEFAULT_PORT
        url = "http://127.0.0.1:%d" % port

    gguf = str(merged.get("gguf") or "")
    if not gguf and not merged.get("size") and backend == "llama-server":
        # 学 hsbg：没给档位线索时，从"实际会被 llama-server 加载的那个 GGUF"推 ——
        # 否则 4B 会套 2B 的校准温度（1.42 vs 1.532），概率整体偏平。
        try:
            found = find_gguf()
        except Exception:
            found = None
        if found:
            gguf = found
            merged = dict(merged)
            merged["gguf"] = found
    size = str(merged.get("size") or "") or (size_key(gguf) if gguf else "")

    try:
        temperature = resolve_temperature(merged)
    except (TypeError, ValueError):
        raise BadRequest("temperature 类型不对：%r" % merged.get("temperature"))
    if temperature <= 0:
        raise BadRequest("temperature 必须为正数")

    mock_probs = merged.get("mock_probs")
    if mock_probs is not None:
        if not isinstance(mock_probs, dict):
            raise BadRequest("mock_probs 必须是对象")
        clean = {}  # type: Dict[str, float]
        for k, v in mock_probs.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
                raise BadRequest("mock_probs 必须是非负数：%r" % (k,))
            clean[str(k)] = float(v)
        mock_probs = clean

    p = Params(
        backend=backend, url=url,
        model=pick("model", str, ""), profile=profile_name,
        temperature=temperature, top_k=top_k, timeout_s=timeout_s,
        retries=retries, max_options=max_options,
        option_prefix=pick("option_prefix", bool, True),
        include_raw=pick("include_raw", bool, False),
        template=pick("template", str, "qwen3.5-think-off"),
        system=pick("system", str, None),
        letter_floor_margin=margin, cache_ttl_s=cache_ttl,
        cache_prompt=pick("cache_prompt", bool, False),
        mock_probs=mock_probs,
    )
    gguf = str(merged.get("gguf") or "")
    name = str(merged.get("name") or "") or (
        Path(gguf).stem if gguf else (p.model or p.backend))
    echo = {"profile": p.profile, "backend": p.backend, "name": name,
            "size": size, "temperature": p.temperature, "top_k": p.top_k}
    return p, echo


def _compute(base: dict, task: str, evidence: str, criterion: str,
             pairs: Sequence[Tuple[str, str]], p: Params, echo: dict) -> dict:
    ids = [k for k, _ in pairs]
    seen = None  # type: Optional[Dict[str, float]]
    if p.mock_probs is not None:
        # 快捷路：直给分布，绕过模型与温度（但仍走截断料理，∑p = 1）
        kept = ids[:p.max_options]
        for oid in kept:
            if oid not in p.mock_probs:
                raise BadRequest("mock_probs 未覆盖选项 %r" % oid)
        weights = [p.mock_probs[oid] for oid in kept]
        total = sum(weights)
        if total <= 0:
            raise BadRequest("mock_probs 权重全为 0")
        probs = {oid: w / total for oid, w in zip(kept, weights)}
        probs, truncated = fill_truncated(probs, ids, len(kept))
        missing, tokens, seconds, attempts = 0, 0, 0.0, 0
    else:
        lines = option_lines(pairs[:p.max_options], prefix=p.option_prefix)
        system = p.system or SYSTEM_MESSAGE
        user = build_user_json(evidence, criterion, lines)
        prompt = Prompt(system=system, user=user,
                        text=render_prompt(system, user, p.template))
        backend = make_backend(p)
        res = backend.letter_logprobs(prompt, p)
        seen = res.logprobs
        probs, missing, truncated = distribute(
            ids, seen, p.temperature, p.letter_floor_margin, p.max_options)
        tokens, seconds, attempts = res.tokens, res.seconds, res.attempts

    ranked = [{"id": k, "p": v}
              for k, v in sorted(probs.items(), key=lambda kv: -kv[1])]
    warnings = []
    if missing:
        warnings.append("missing_letters")
    if truncated:
        warnings.append("truncated_options")
    if len(ranked) >= 2 and ranked[0]["p"] - ranked[1]["p"] < 0.05:
        warnings.append("low_signal")

    out = dict(base)
    out.update({
        "ok": True, "error": None,
        "probs": probs, "ranked": ranked,
        "choice": None, "letter": None, "confidence": max(probs.values()),
        "noul": None, "score": None,
        "warnings": warnings,
        "model": echo,
        "metrics": {"elapsed_ms": round(seconds * 1000, 1),
                    "tokens_evaluated": int(tokens),
                    "missing_letters": int(missing),
                    "truncated": int(truncated),
                    "cache": "off",
                    "attempts": int(attempts)},
    })
    if task == "noul":
        out["noul"] = probs.get("true", 0.0)
    elif task == "choice":
        choice = max(probs, key=probs.get)
        out["choice"] = choice
        if choice in ids and ids.index(choice) < len(LETTERS):
            out["letter"] = LETTERS[ids.index(choice)]
    else:
        out["score"] = sum(int(k) * v for k, v in probs.items())
    if p.include_raw:
        out["raw"] = {"logprobs": dict(seen) if seen else {}}
    return out


def ask_item(item, registry: Optional[dict] = None,
             session: Optional[Session] = None) -> Resp:
    """契约形状的单题入口（CLI / MCP / 库都走它）。"""
    if not isinstance(item, dict):
        return Resp({"v": 1, "id": "", "ok": False,
                     "error": {"code": "bad_request",
                               "message": "请求必须是 JSON 对象"}})
    rid = str(item.get("id") or "")
    base = {"v": 1, "id": rid, "ok": False, "error": None}
    task = item.get("task")
    try:
        if item.get("v") not in (None, 1):
            raise BadRequest("不支持的契约版本：%r" % item.get("v"))
        if task not in TASKS:
            raise BadRequest("task 必须是 choice / noul / score，得到 %r" % task)
        base["task"] = task
        evidence = item.get("evidence") or ""
        if not isinstance(evidence, str):
            raise BadRequest("evidence 必须是字符串")
        criterion = item.get("criterion") or DEFAULT_CRITERION[task]
        if not isinstance(criterion, str):
            raise BadRequest("criterion 必须是字符串")
        pairs = _norm_options(task, item.get("options"))
        if task == "score":
            for k, _ in pairs:
                try:
                    int(k)
                except (TypeError, ValueError):
                    raise BadRequest("score 题的选项 id 必须是整数序号，得到 %r" % k)
        p, echo = _effective(_reg(registry), item)
    except OddsError as e:
        out = dict(base)
        out["error"] = {"code": e.code, "message": str(e)}
        return Resp(out)

    key = None
    ttl = 0.0
    if session is not None:
        ttl = p.cache_ttl_s if p.cache_ttl_s > 0 else session.default_ttl
        if ttl > 0:
            key = cache_key(item)
            hit = session.get(key)
            if hit is not None:
                hit["metrics"]["cache"] = "hit"
                return Resp(hit)
    try:
        out = _compute(base, task, evidence, criterion, pairs, p, echo)
    except OddsError as e:
        out = dict(base)
        out["error"] = {"code": e.code, "message": str(e)}
        return Resp(out)
    except Exception as e:  # 契约要求 internal 也是结构化 JSON
        out = dict(base)
        out["error"] = {"code": "internal",
                        "message": "%s: %s" % (type(e).__name__, e)}
        return Resp(out)
    out["metrics"]["cache"] = "miss" if key is not None else "off"
    if key is not None:
        session.put(key, out, ttl)
    return Resp(out)


def ask(evidence: str = "", criterion: Optional[str] = None,
        options=None, task: str = "choice", qid: str = "",
        params: Optional[dict] = None, registry: Optional[dict] = None,
        session: Optional[Session] = None) -> Resp:
    """库形态的便捷入口（DESIGN.md §9.1）：

        resp = ask(evidence="…", criterion="…", options=["a", "b"])
        resp.ok, resp.choice, resp.probs
    """
    item = {"v": 1, "task": task, "evidence": evidence}  # type: dict
    if qid:
        item["id"] = qid
    if criterion is not None:
        item["criterion"] = criterion
    if options is not None:
        item["options"] = options
    if params is not None:
        item["params"] = params
    return ask_item(item, registry=registry, session=session)


def ask_many(items, params: Optional[dict] = None,
             registry: Optional[dict] = None,
             session: Optional[Session] = None) -> dict:
    """批量：`params` 是全体默认，单题 `params` 可再覆盖。"""
    if not isinstance(items, list) or not items:
        raise BadRequest("items 必须是非空数组")
    if params is not None and not isinstance(params, dict):
        raise BadRequest("params 必须是对象")
    reg = _reg(registry)
    results = []
    for item in items:
        if not isinstance(item, dict) or not params:
            results.append(ask_item(item, registry=reg, session=session))
            continue
        merged = dict(item)
        ip = item.get("params")
        if ip is None:
            ip = {}
        if isinstance(ip, dict):
            p2 = dict(params)
            p2.update(ip)
            merged["params"] = p2
        else:
            merged["params"] = ip  # 让单题层报 bad_request
        results.append(ask_item(merged, registry=reg, session=session))
    return {"v": 1, "results": results}


def _available(prof: dict) -> bool:
    backend = str(prof.get("backend") or "llama-server")
    if backend == "mock":
        return True
    url = str(prof.get("url") or "")
    if not url and backend == "llama-server":
        try:
            port = int(prof.get("port") or DEFAULT_PORT)
        except (TypeError, ValueError):
            port = DEFAULT_PORT
        url = "http://127.0.0.1:%d" % port
    p = Params(backend=backend, url=url, model=str(prof.get("model") or ""),
               timeout_s=3.0, retries=0)
    try:
        return make_backend(p).health()
    except Exception:
        return False


def models_info(registry: Optional[dict] = None, check: bool = True) -> dict:
    """`models` 命令 / MCP `models` 工具的数据。"""
    reg = _reg(registry)
    default = default_name(reg) or ""

    def info_of(prof: dict, probe: bool) -> dict:
        backend = str(prof.get("backend") or "llama-server")
        url = str(prof.get("url") or "")
        if not url and backend == "llama-server":
            try:
                port = int(prof.get("port") or DEFAULT_PORT)
            except (TypeError, ValueError):
                port = DEFAULT_PORT
            url = "http://127.0.0.1:%d" % port
        d = {"backend": backend, "url": url}  # type: dict
        if prof.get("model"):
            d["model"] = str(prof["model"])
        if prof.get("size"):
            d["size"] = str(prof["size"])
        if probe:
            d["available"] = _available(prof)
        return d

    out = {"v": 1, "default": default,
           "registry": str(registry_path()), "models": {}}
    for name in sorted((reg.get("models") or {})):
        try:
            out["models"][name] = info_of(get_profile(reg, name), check)
        except OddsError as e:
            out["models"][name] = {"error": str(e), "available": False}
    needs_builtin = not (default and default in out["models"])
    # 有可用的默认档位时不白探内建（省一次 3s 超时）
    out["builtin"] = info_of(dict(BUILTIN), check and needs_builtin)
    if default and default in out["models"]:
        out["default_available"] = bool(out["models"][default].get("available"))
    else:
        out["default_available"] = bool(out["builtin"].get("available"))
        if not default:
            out["default"] = ""
    return out
