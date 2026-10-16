# llm-odds

本地模型的**校准概率顾问**：给「上下文 + 选项」，拿回概率分布。

- **可换模型** —— 模型注册表 + 后端接口（llama-server 真 logprob / OpenAI 兼容 / mock）
- **可控参数** —— 温度、top_k、超时、模板、前缀、输出形状……三层覆盖（请求 > profile > 默认）、响应回显
- **三种接入** —— Python 库 / CLI（JSON 进 JSON 出）/ MCP（stdio）
- **零依赖** —— 纯标准库，跑在本机 `/usr/bin/python3`

> 状态：**M1 + M3 已落地**（2026-10-16）—— core/backends/CLI/MCP 全部可用，
> 离线测试 125 项全绿；与 `hsbg-helper` 现读法的逐字节对照通过
> （prompt 全等、概率最大差 0.000e+00）。推进计划与决策见
> [`docs/DESIGN.md`](docs/DESIGN.md) §12 / §13 / §15。

## 快速上手

```bash
./bin/llm-odds selftest --mock     # 全链路冒烟（离线，CI 入口）
./bin/llm-odds models              # 看注册表里有哪些档位、谁可用
./bin/llm-odds serve start         # 拉起本机 llama-server（LM Studio 自带的那枚）
./bin/llm-odds ask req.json        # 单题；`-` 从 stdin
./bin/llm-odds ask-many reqs.json  # 批量（一次进程问一串）
./bin/llm-odds mcp                 # stdio MCP server
./run-tests                        # 零依赖测试（-v / -k 过滤）
```

- stdout **只出 JSON**（人话全走 stderr）——宿主可以放心 `json.loads(stdout)`。
- 退出码：`0` 成功 / `2` 契约或参数错 / `3` 模型不可用（含 logprobs 不支持）/
  `4` 超时 / `1` 内部错。

## 契约速览（v1，详见 `schemas/` 与 `examples/`）

请求：

```json
{"v": 1, "task": "choice",
 "evidence": "Turn 8, 11 gold, tier 4. Shop: ...",
 "criterion": "Which purchase gives the strongest immediate board?",
 "options": ["slot1: Murloc Holmes 4/4 (Murloc)", "slot2: ..."]}
```

响应（`ok=false` 时给 `error.code`，见 DESIGN.md §5.3）：

```json
{"v": 1, "ok": true, "task": "choice",
 "probs": {"slot1": 0.62, "slot2": 0.38},
 "ranked": [{"id": "slot1", "p": 0.62}, {"id": "slot2", "p": 0.38}],
 "choice": "slot1", "letter": "A", "confidence": 0.62,
 "warnings": [],
 "model": {"profile": "jevk5-4b", "backend": "llama-server",
           "name": "jevk5-4b-v0.2-Q4_K_M", "size": "4b-v0.2",
           "temperature": 1.532, "top_k": 40},
 "metrics": {"elapsed_ms": 812, "tokens_evaluated": 321,
             "missing_letters": 0, "truncated": 0,
             "cache": "miss", "attempts": 1}}
```

## 模型注册表（换模型零改宿主）

`~/.config/llm-odds/models.json`（`LLM_ODDS_REGISTRY` 或 `--registry` 可覆盖）：

```json
{
  "default": "jevk5-4b",
  "models": {
    "jevk5-4b": {
      "backend": "llama-server",
      "gguf": "~/.cache/lm-studio/models/alibiserikbay/JevK5-GGUF/jevk5-4b-v0.2-Q4_K_M.gguf",
      "temperature": 1.532, "top_k": 40, "port": 8090
    },
    "some-qwen": {
      "backend": "openai",
      "url": "http://127.0.0.1:1234/v1", "model": "qwen3-8b",
      "temperature": 1.0, "top_k": 20
    }
  }
}
```

一条 profile = 一个模型档位；请求 `params` 可临时覆盖任何一项，响应 `model` 块
回显最终生效值（可复现、可审计）。**没有注册表也能跑**：内建默认档位 =
llama-server @ 127.0.0.1:8090 + 从本机 GGUF 自动推校准温度（学 hsbg）。

## MCP（stdio）接入

`tools`：`ask` / `ask_many` / `models` / `serve`。示例配置（Claude Desktop 等）：

```json
{"mcpServers": {"llm-odds": {
  "command": "/Users/edwa2d4th/Desktop/project/llm-odds/bin/llm-odds",
  "args": ["mcp"]}}}
```

## 换模型验收清单（DESIGN.md 附录 B）

1. 字母命中率：正常提问下 `missing_letters == 0`（反复出现 → 调大 `top_k`）；
2. 分布形状：一眼"该赢的选项"概率显著最高（挤平 / 挤歪 → 校准温度没对）；
3. 校准温度：官方有值用官方；没有就重标（M5 工具）；
4. P95 延迟 < 面板 TTL（hsbg 面板 = 12s）的 1/3；
5. 跑一遍 `./bin/llm-odds selftest` + `./run-tests`。

## 目录

```
bin/llm-odds        # bash 包装器（PYTHONPATH=src），同 hsbg 的 ./hsbg
src/llm_odds/       # core + backends + 边缘（见 DESIGN.md §4）
schemas/            # 数据契约（JSON Schema，v1）
docs/DESIGN.md      # 设计稿：目标 / 架构 / 契约 / 迁移计划 / 进展
examples/           # 契约 golden 样例（测试逐字段比对）
tests/              # 零依赖离线测试（./run-tests）
```
