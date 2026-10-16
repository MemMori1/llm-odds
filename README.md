# llm-odds

本地模型的**校准概率顾问**：给「上下文 + 选项」，拿回概率分布。

- **可换模型** —— 模型注册表 + 后端接口（llama-server 真 logprob / OpenAI 兼容 / mock）
- **可控参数** —— 温度、top_k、超时、模板、前缀、输出形状……三层覆盖（请求 > profile > 默认）、响应回显
- **三种接入** —— Python 库 / CLI（JSON 进 JSON 出）/ MCP（stdio）
- **零依赖** —— 纯标准库，跑在本机 `/usr/bin/python3`

> 状态：**设计稿**（[`docs/DESIGN.md`](docs/DESIGN.md)，2026-10-16）。
> 代码从 `hsbg-helper` 的模型层抽取（`provider.py` / `model_server.py`），
> 按 DESIGN.md §12 的 M0–M5 阶段推进。

## 契约速览（v1，详见 `schemas/`）

请求：

```json
{"v": 1, "task": "choice",
 "evidence": "Turn 8, 11 gold, tier 4. Shop: ...",
 "criterion": "Which purchase gives the strongest immediate board?",
 "options": ["slot1: Murloc Holmes 4/4 (Murloc)", "slot2: ..."]}
```

响应：

```json
{"v": 1, "ok": true, "task": "choice",
 "probs": {"slot1": 0.62, "slot2": 0.38},
 "ranked": [{"id": "slot1", "p": 0.62}, {"id": "slot2", "p": 0.38}],
 "choice": "slot1", "confidence": 0.62,
 "model": {"profile": "jevk5-4b", "backend": "llama-server",
           "temperature": 1.532, "top_k": 40}}
```

## 目录（规划）

```
bin/llm-odds        # bash 包装器（PYTHONPATH=src），同 hsbg 的 ./hsbg
src/llm_odds/       # core + backends + 边缘（见 DESIGN.md §4）
schemas/            # 数据契约（JSON Schema，v1）
docs/DESIGN.md      # 设计稿：目标 / 架构 / 契约 / 迁移计划
examples/           # 契约 golden 样例
tests/              # 零依赖离线测试
```
