# llm-odds

本地模型的**校准概率顾问**：给「上下文 + 选项」，拿回概率分布。

- **可换模型** —— 模型注册表 + 后端接口（llama-server 真 logprob / OpenAI 兼容 / mock）
- **可控参数** —— 温度、top_k、超时、模板、前缀、输出形状；三层覆盖（请求 > profile > 默认），响应回显
- **三种接入** —— Python 库 / CLI（JSON 进 JSON 出）/ MCP（stdio）
- **零依赖** —— 纯标准库，跑在本机 `/usr/bin/python3`

> 状态：**M1 + M3 已落地、M2 宿主接入已完成**（2026-10-16）——离线测试 125 项全绿；
> 与 `hsbg-helper` 现读法逐字节对照通过（prompt 全等、概率最大差 0.000e+00）。
> 架构 / 契约 / 迁移与推进计划：**[`docs/DESIGN.md`](docs/DESIGN.md)**。

## 快速上手

```bash
./bin/llm-odds selftest --mock     # 全链路冒烟（离线，CI 入口）
./bin/llm-odds models              # 注册表里有哪些档位、谁可用
./bin/llm-odds serve start         # 拉起本机 llama-server（LM Studio 自带的那枚）
./bin/llm-odds ask req.json        # 单题；`-` 从 stdin
./bin/llm-odds ask-many reqs.json  # 批量（一次进程问一串）
./bin/llm-odds mcp                 # stdio MCP server
./run-tests                        # 零依赖测试（-v / -k 过滤）
```

- stdout **只出 JSON**（人话全走 stderr）——宿主可以放心 `json.loads(stdout)`。
- 退出码：`0` 成功 / `2` 契约或参数错 / `3` 模型不可用（含 logprobs 不支持）/
  `4` 超时 / `1` 内部错。

## 契约速览（v1）

请求 `{v, task: choice|noul|score, evidence, criterion, options[], params}` →
响应 `{ok, task, probs, ranked, choice, letter, confidence, noul, score, warnings,
model, metrics}`（失败给 `error.code`，总表见 DESIGN §5.3）。完整定义与 golden
样例：[`schemas/`](schemas/) + [`examples/`](examples/)。

```json
{"v": 1, "task": "choice",
 "evidence": "Turn 8, 11 gold, tier 4. Shop: ...",
 "criterion": "Which purchase gives the strongest immediate board?",
 "options": ["slot1: Murloc Holmes 4/4 (Murloc)", "slot2: ..."]}
```

```json
{"v": 1, "ok": true, "probs": {"slot1": 0.62, "slot2": 0.38},
 "choice": "slot1", "letter": "A", "confidence": 0.62, "warnings": [],
 "model": {"profile": "jevk5-4b", "temperature": 1.532, "top_k": 40},
 "metrics": {"elapsed_ms": 812, "tokens_evaluated": 321, "cache": "miss"}}
```

## 模型注册表（换模型零改宿主）

`~/.config/llm-odds/models.json`（`LLM_ODDS_REGISTRY` / `--registry` 可覆盖）：
一条 profile = 一个模型档位 —— 后端（`llama-server` / `openai` / `mock`）、
`gguf` 或 `url`、温度、top_k、port…… 请求 `params` 可临时覆盖任何一项，响应
`model` 块回显生效值（可复现、可审计）。**没有注册表也能跑**：内建档位 =
llama-server @ `127.0.0.1:8090` + 从本机 GGUF 自动推校准温度。字段总表见 DESIGN §7。

## MCP（stdio）接入

`tools`：`ask` / `ask_many` / `models` / `serve`。宿主配置示例（Claude Desktop 等）：

```json
{"mcpServers": {"llm-odds": {
  "command": "/Users/edwa2d4th/Desktop/project/llm-odds/bin/llm-odds",
  "args": ["mcp"]}}}
```

## 宿主接入

已有参考宿主：`hsbg-helper` 的 `src/hsbg/model_client.py` —— 用 `HSBG_ODDS_BIN`
等环境变量找本插件，以 `ask-many` 子进程调用，失败自动降级（约 250 行 + 19 项
离线测试，可作为其他宿主的模板）。换模型验收清单见 DESIGN 附录 B。

## 目录

```
bin/llm-odds        # bash 包装器（PYTHONPATH=src），同 hsbg 的 ./hsbg
src/llm_odds/       # core + backends + 边缘（见 DESIGN §4）
schemas/            # 数据契约（JSON Schema，v1）
examples/           # 契约 golden 样例（测试逐字段比对）
docs/DESIGN.md      # 设计稿：架构 / 契约 / 迁移 / 进展 / 换模型验收清单
tests/              # 零依赖离线测试（./run-tests）
```
