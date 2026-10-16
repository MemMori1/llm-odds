# llm-odds · 设计稿（v1）

> 状态：设计稿 · 2026-10-16 · 从 `hsbg-helper` 的模型层抽取。
> 工作名 `llm-odds`（备选：`prob-oracle` / `model-odds`，定稿前可改，见 §13）。

## 0. 一句话

给模型一段上下文和一列选项，拿回**校准过的概率分布**；模型可换（注册表 + 后端接口）、
参数可控（三层覆盖 + 回显）、接入形态可选（库 / CLI / MCP）、零第三方依赖。

## 1. 背景：为什么要拆

hsbg-helper 的模型层（P4 起）实际由四块组成，前三块与「酒馆战棋」毫无耦合，
是纯粹的"本地模型概率读取器"：

| 现址 | 内容 | 与酒馆耦合？ |
|---|---|---|
| `provider.py`（377 行） | JevK5 System One 协议：模板、字母 logprob 读法、温度校准、缺口兜底；`Question/Answer`；`Advisor` 接口 | 无 |
| `model_server.py`（228 行） | 找 llama-server / GGUF、启停、health、PID | 无 |
| `panel.py` 的 `ModelAdvisorRunner` | TTL 缓存 + 后台线程 + 快照防抖 | 无 |
| `advisor.py` 的 `consult()` | 编排：可用性 → 逐题问 → 失败降级 | 一半（evidence/题目是酒馆语义） |

拆出去的动机（按重要性）：

1. **换模型不该改主仓库**：换 GGUF / 换推理后端（llama.cpp → OpenAI 兼容 / MLX）会直接动
   `provider.py` 与 `model_server.py`；主仓库的测试、面板、CLI 都被牵连。
2. **出入参要可控**：温度、top_k、模板、超时、输出形状……这些是"调模型"的旋钮，
   不该烤死在宿主代码里；不同项目可能只要一个子集。
3. **未来项目复用**：概率分析类需求（多项目）要的是"上下文 + 选项 → 分布"这一层，
   而不是酒馆的 evidence 构造。
4. **接法要可换**：插件（子进程 JSON）、MCP（stdio）、将来可能 HTTP daemon ——
   宿主只该依赖一份 **JSON 契约**。

## 2. 目标 / 非目标

### 目标

- **G1 换模型零改宿主**：模型注册表（JSON）+ 后端接口；加一个模型 = 加一条注册表
  +（只有换协议时才需要）一个后端。
- **G2 出入参可控**：请求带 `params`；**请求级 > 模型 profile > 内置默认**三层覆盖；
  响应回显最终生效值与元数据（可审计、可复现）。
- **G3 多形态接入**：Python 库、CLI（JSON in/out）、MCP（stdio）；三形态共用同一核心、
  同一契约。
- **G4 概率质量可复现**：校准温度表、缺字母/截断的确定性策略、确定性 mock 后端、
  缓存键明确。
- **G5 离线可测 + 零依赖**：纯标准库；测试不碰网络（假 `urlopen` 桩 / mock 后端）。

### 非目标

- **N1 不做生成式问答**：v1 只做"从一组选项里读概率"（logprob 读法），
  不做自由文本 → 语义解析。
- **N2 不做训练 / 微调 / 量化**：只使用现成 GGUF / 本地服务。
- **N3 不留任何执行接口**（沿用 hsbg 红线）：只读本地模型、只出概率，
  不碰键鼠、不注入任何宿主。
- **N4 不做业务语义**：evidence 怎么写、题目怎么出（买哪张 / 升不升本）
  永远是宿主的职责。

## 3. 核心概念

| 概念 | 含义 |
|---|---|
| **Task** | 题型：`choice`（N 选 1）/ `noul`（命题真假）/ `score`（档位期望）。沿用 JevK5 官方三型。 |
| **Evidence** | 英文上下文（宿主编排，见 §5.5）。 |
| **Criterion** | 一句话判断标准（如 `"Which purchase best improves my board now?"`）。 |
| **Option** | 选项 = `id + text`。id 由宿主定义（`slot2` / `true` / 档位号），插件只负责传递与回传。 |
| **Distribution** | `{option_id → p}`，∑p = 1；`choice` 题另附 `choice / letter / confidence`。 |
| **Backend** | 产生"答案位置各字母 logprob"的通道：`llama-server` / `openai` / `mock`。 |
| **Model Profile** | 注册表条目：backend + 地址 / 权重 + 校准与默认参数。 |
| **Session** | 常驻进程里的缓存层（TTL、去重）；CLI 单发模式下等价于直通。 |

## 4. 架构

```
                      ┌─────────────── llm-odds ────────────────┐
 宿主（hsbg / 未来项目）│  core/（纯函数，无网络无副作用）           │
   · 组装 evidence     │    types.py     Question / Answer       │
   · 出题 / 收题       │    protocol.py  模板 · 字母 · 选项文本    │
   · 展示 / 降级       │    calibrate.py 温度表 · 缺字母 · 截断    │
        │             │    api.py       ask() / ask_many()      │
        │  JSON 契约   │    session.py   TTL 缓存                │
        ├── 库 import ─┤  backends/（网络 IO，单一职责）            │
        ├── CLI 子进程 ┤    llama_server.py  （n_probs 读法）      │
        ├── MCP stdio ┤    openai_compat.py （logprobs 读法）     │
        └── (v2 HTTP) │    mock.py          （确定性，测试/演示）  │
                      │  registry.py  模型注册表                  │
                      │  server.py    本地推理服务生命周期         │
                      ├──────────────── 边缘 ────────────────────┤
                      │  cli.py    ask / ask-many / models /     │
                      │            serve / mcp / selftest        │
                      │  mcp.py    stdio JSON-RPC（tools）        │
                      └──────────────────────────────────────────┘
```

**分层硬约束**（写代码时不许破）：

1. `core/` 不许 import `backends/` 的具体实现（只依赖 Backend 接口）——
   这是"换模型零改宿主"的边界。
2. `core/` 不许做网络 IO（全部纯函数，可离线单测）。
3. `backends/` 不许认识任何宿主概念（不知道什么是"商店"）。
4. `cli.py` / `mcp.py` 只做协议翻译，不放逻辑（两者调同一个 `api.ask_many`）。

目录树（规划）：

```
llm-odds/
  bin/llm-odds              # bash 包装器（PYTHONPATH=src），同 hsbg 的 ./hsbg
  src/llm_odds/
    __init__.py             # 导出 ask / ask_many / Question / Answer
    types.py  protocol.py  calibrate.py  api.py  session.py
    backends/{base,llama_server,openai_compat,mock}.py
    registry.py  server.py  cli.py  mcp.py
  schemas/                  # 数据契约（JSON Schema，v1）
  docs/DESIGN.md            # 本文件
  examples/                 # 契约样例（golden 测试用）
  tests/                    # 零依赖离线测试（抄 hsbg 的 run_tests.py 风格）
```

## 5. 数据契约 v1（跨项目边界，唯一的强承诺）

> 原则：**宿主永远只依赖这份 JSON，不读插件的 Python 对象**。
> 契约只加字段、不改语义；破坏性变更 = 升 `v`。

### 5.1 请求

```json
{
  "v": 1,
  "id": "buy-2026-10-16T20:31:02",
  "task": "choice",
  "evidence": "Turn 8, 11 gold, tier 4, health 28. Shop: ...",
  "criterion": "Which purchase gives the strongest immediate board?",
  "options": ["slot1: Murloc Holmes 4/4 (Murloc)", "slot2: ..."],
  "params": {"include_raw": true, "timeout_s": 30}
}
```

| 字段 | 必需 | 默认 | 说明 |
|---|---|---|---|
| `v` | 建议 | 1 | 契约版本；省略按 1 处理 |
| `id` | 否 | `""` | 原样回显（宿主对账用） |
| `task` | 是 | — | `choice` / `noul` / `score` |
| `evidence` | 是 | `""` | 英文上下文 |
| `criterion` | 是 | 按题型默认 | 判断标准（一句话） |
| `options` | choice/score 必需 | — | `[str]` 或 `[{id, text}]`；str 自动配 id `"0","1",…` |
| `params` | 否 | `{}` | 见 §7.2 参数总表 |

### 5.2 响应

```json
{
  "v": 1, "id": "buy-…", "ok": true, "task": "choice",
  "probs": {"slot1": 0.62, "slot2": 0.38},
  "ranked": [{"id": "slot1", "p": 0.62}, {"id": "slot2", "p": 0.38}],
  "choice": "slot1", "confidence": 0.62,
  "noul": null, "score": null,
  "warnings": [],
  "model": {"profile": "jevk5-4b", "backend": "llama-server",
            "name": "jevk5-4b-v0.2-Q4_K_M", "size": "4b-v0.2",
            "temperature": 1.532, "top_k": 40},
  "metrics": {"elapsed_ms": 812, "tokens_evaluated": 321,
              "missing_letters": 0, "truncated": 0,
              "cache": "miss", "attempts": 1},
  "raw": {"logprobs": {"A": -0.15, "B": -1.32, "C": -3.21}},
  "error": null
}
```

- `probs` **始终归一化**（∑p = 1）；`raw.logprobs` 仅在 `include_raw=true` 时出现。
- `warnings`（可空数组）：`missing_letters`（有字母差点掉出 top_k）、
  `truncated_options`（选项 > `max_options` 被截）、`low_signal`
  （分布过平 / 无显著胜者 —— v1 先定义，阈值后调）。
- `model` / `metrics` 是**回显与审计**：换模型排查问题先看这两块。
- `cache`：`"hit"` / `"miss"` / `"off"`。

### 5.3 错误模型

失败也返回结构化 JSON（`ok=false` + `error`），错误码：

| code | 含义 | 宿主建议动作 |
|---|---|---|
| `bad_request` | JSON / schema 不符 | 修调用（开发期错误） |
| `unknown_profile` | 注册表里没有该 profile | 提示去 `models` 看一眼 |
| `model_unavailable` | 连不上本地服务 / 服务端 5xx | 降级（hsbg：rules-only） |
| `logprobs_unavailable` | 服务不支持 logprobs（如 LM Studio 自带服务） | 换后端；降级 |
| `timeout` | 超时（含重试后） | 降级 |
| `no_options` | choice/score 没给选项 | 修调用 |
| `internal` | 其它 | 降级 + 日志 |

### 5.4 `ask_many`（批量）

```json
{"v": 1, "params": {"……": "全体默认，单题可再覆盖"},
 "items": [ {"……单题……"}, {"……单题……"} ]}
```

响应：`{"v": 1, "results": [ ……单题响应…… ]}`。

存在的理由：CLI 形态下"一次进程问一串"（面板一轮 3–5 题），避免每题一次进程启动。

### 5.5 两条语言与形状约定

- **英文**：evidence / criterion / 选项文本**必须英文**（ADR-4 的由来）——
  插件不翻译、不改写，原样进 prompt。
- **选项文本前缀**：带不带 `"id: "` 由 `option_prefix` 控制（默认带）——
  官方 JevBench 的映射就是这个形状。

## 6. 后端设计

### 6.1 Backend 接口

```python
class Backend:
    name: str                                  # "llama-server" / "openai" / "mock"
    def health(self) -> bool: ...
    def letter_logprobs(self, *, system: str, user: str, letters: str,
                        params: EffectiveParams) -> Result[Dict[str, float]]
    # Result = {letter: logprob}（未进 top_k 的字母可以缺席）+ metrics
```

`core` 只依赖这个接口；**模板渲染与概率化都在 core**（见 §6.5），后端只管
"把 prompt 变成 logprobs"。

### 6.2 `llama-server` 后端（JevK5 读法，原样搬家）

- system / user 结构 + Qwen3.5 模板**预填空 think 块** —— 答案位置的字母是下一个 token。
- ⚠️ 必须先 `/tokenize`（`parse_special=true`）：字符串直传会把 `<|im_start|>`
  拆成普通字符，模板结构全废（官方注释原话）。
- `/completion`：`n_predict=1`、`n_probs=top_k`、`temperature=0`（读概率用 0，
  校准温度在 core 应用）、`cache_prompt=false`（现状；作为参数可开）。
- 读 `completion_probabilities[0].top_logprobs`；服务不支持时错误码
  `logprobs_unavailable`（LM Studio 自带 HTTP 服务就是这一类）。
- ⚠️ `/health` **只认 GET**（POST 404），且 `HTTPError` 必须先于 `URLError` catch
  （实测踩过，写反会把 404 报成"连不上"）。

### 6.3 `openai` 兼容后端

- `POST {url}/chat/completions`，带 `logprobs: true, top_logprobs: top_k`；
  system/user 交给服务端模板。
- ⚠️ 各家支持不一：支持则读答案位置 token 的 `top_logprobs`；不支持 →
  `logprobs_unavailable`（**不静默降级成生成式** —— N1）。
- 用途：未来换 MLX / vLLM / 远端的同类服务；`model` 参数指定模型名。

### 6.4 `mock` 后端

- `params.mock_probs` 直接给分布（校验归一）；不给则按
  `(task, evidence, options)` 的稳定 hash 生成**确定性**分布。
- 用途：全部单测、`selftest`、宿主集成测试、演示。

### 6.5 概率化与校准（core，全部可覆盖）

公式与 hsbg 现状一致（数值都是实测搬来的，不重学）：

```
p_i = softmax(logp_i / T)              T = 校准温度（profile/params，默认查表）
logp_i 缺字母 → floor = min(seen) − letter_floor_margin（默认 2.0）
选项数 > max_options（默认 16）→ 截断：给被丢选项留 1/(N+1) 份额，其余按比例扣
```

校准温度表（官方 JevK5）：`2b 1.42 / 4b 1.367(v0.3) / 4b-v0.2 1.532 / 9b 1.089`。
换模型 = 换表项或 `params.temperature` 覆盖；**温度错了概率会整体偏平或偏尖**。

## 7. 模型注册表与可调参数

### 7.1 注册表

路径：`~/.config/llm-odds/models.json`（`--registry` 或 `LLM_ODDS_REGISTRY` 可覆盖）。

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

### 7.2 参数总表（三层覆盖：请求 > profile > 内置默认）

| 参数 | 层 | 默认 | 说明 |
|---|---|---|---|
| `backend` | 请求/profile | profile | `llama-server` / `openai` / `mock` |
| `profile` | 请求 | 注册表 `default` | 选哪个模型 |
| `url` | 请求/profile | profile / 8090 | 服务地址 |
| `model` | 请求/profile | — | openai 后端的模型名 |
| `temperature` | 请求/profile | 查校准表 | **校准**温度（不是采样温度） |
| `top_k` | 请求/profile | 40 | n_probs / top_logprobs 深度 |
| `timeout_s` | 请求/profile | 60 | 单次请求超时 |
| `retries` | 请求/profile | 1 | 网络重试次数 |
| `max_options` | 请求/profile | 16 | 字母表上限，超出截断 |
| `option_prefix` | 请求/profile | true | 选项文本是否带 `id: ` 前缀 |
| `include_raw` | 请求 | false | 响应是否带 raw logprobs |
| `template` | 请求/profile | qwen3.5-think-off | 模板名或模板字符串 |
| `system` | 请求/profile | 官方固定句 | 覆盖 system message |
| `letter_floor_margin` | 请求/profile | 2.0 | 缺字母兜底扣分 |
| `cache_ttl_s` | 请求/profile | 0 | 进程内缓存（宿主可另做缓存） |
| `cache_prompt` | 请求/profile | false | llama-server KV 复用 |
| `mock_probs` | 请求 | — | mock 后端直给分布 |

**回显**：响应 `model` 块 = 最终生效的模型相关参数 —— 宿主的日志能完整复现一次提问。

## 8. 本地服务生命周期（server.py）

从 `model_server.py` 泛化：

- 找 llama-server：LM Studio backends 目录
  （`~/.cache/lm-studio/extensions/backends/*/llama-server`）→ PATH `llama-server`
  → profile 显式 `server_path`；
- 找权重：profile 的 `gguf`（支持 `~` 与 glob）；找不到时打全人话报错（学 hsbg 文案）；
- PID / 日志按 profile 隔离：`/tmp/llm-odds-<profile>.pid` / `…-<profile>.log`
  （避免多项目互踢服务）；
- `serve start` 等 `/health` 就绪（≤ 90s）再返回；`stop` 先 SIGTERM 再 SIGKILL；
  `status` 输出 JSON（pid / url / health / model / port）。

## 9. 接入形态

### 9.1 Python 库

```python
from llm_odds import ask
resp = ask(evidence="…", criterion="…", options=["a", "b"],
           params={"profile": "jevk5-4b"})
resp.choice, resp.probs
```

### 9.2 CLI

```bash
llm-odds ask req.json            # 或 `-` 从 stdin
llm-odds ask-many reqs.json
llm-odds models [--json]
llm-odds serve start|stop|status [--profile jevk5-4b]
llm-odds mcp                     # stdio MCP server
llm-odds selftest [--mock]       # 全链路冒烟
```

- stdout **只出 JSON**（人话全走 stderr）——宿主可以放心 `json.loads(stdout)`。
- 退出码：`0` 成功 / `2` 契约或参数错 / `3` 模型不可用（含 logprobs 不支持）/
  `4` 超时 / `1` 内部错。

### 9.3 MCP（stdio）

- 传输：**逐行 JSON-RPC 2.0**（MCP stdio 规范）。
- 实现子集（覆盖宿主需要的一切）：`initialize`（回 `serverInfo` + `capabilities.tools`）、
  `notifications/initialized`（忽略）、`tools/list`、`tools/call`、`ping`。
  暂不做 resources / prompts / 采样。
- tools：

| tool | 入参 | 出参 |
|---|---|---|
| `ask` | `{task?, evidence, criterion?, options?, params?}` | §5.2 响应 JSON（text content） |
| `ask_many` | `{items, params?}` | §5.4 响应 |
| `models` | `{}` | 注册表 + 各模型 `available` |
| `serve` | `{action: "start"\|"stop"\|"status", profile?}` | §8 状态 JSON |

- 示例配置（Claude Desktop / 其它 MCP 宿主）：

```json
{"mcpServers": {"llm-odds": {
  "command": "/Users/edwa2d4th/Desktop/project/llm-odds/bin/llm-odds",
  "args": ["mcp"]}}}
```

### 9.4 （v2 预留）HTTP daemon

`llm-odds daemon --port 8091`：把 `api.ask_many` 挂在 `POST /ask` + `GET /health`。
给"不便拉子进程"的宿主。v1 不做，架构上已留位（api 层就是全部逻辑）。

## 10. 与 hsbg-helper 的集成（第一宿主）

**边界**：hsbg 只依赖 §5 契约（先 CLI 子进程，后补 MCP 常驻）。

- 新增 `src/hsbg/model_client.py`：
  - `OddsAdvisor(Advisor)`：把若干 `Question` 聚合成 `ask-many` 请求 →
    子进程 `llm-odds ask-many -` → 解析回 `Answer`；
  - `available()`：看 `llm-odds models --json` 里默认 profile 的 `available`（带短缓存）；
  - 插件缺失 / 失败 → 抛 `ModelUnavailable` → **沿用现有 rules-only 降级**
    （面板、CLI 文案都不变）。
- `consult()` 编排不变；批量：`Advisor` 增加可选 `decide_many`，`consult` 优先用它
  （一次进程问完 3–5 题）。
- 配置：`HSBG_ODDS_BIN`（默认 PATH 上 `llm-odds` → 兜底
  `~/Desktop/project/llm-odds/bin/llm-odds`）；`HSBG_ODDS_PROFILE`（默认注册表 default）。
- `hsbg serve`：M2 期间保留现实现；M4 改成转发 `llm-odds serve …`。
- hsbg 的 `tests/test_provider.py`（77 项）：读法 / 概率化部分随核心搬走；
  hsbg 留薄壳测试（假 `llm-odds` 脚本 → 断言参数拼装、JSON 解析、失败降级）。

## 11. 测试策略（插件侧）

- **零依赖离线**：抄 hsbg `tests/run_tests.py` 的 runner；不用 pytest。
- **纯函数层**：模板 golden（对齐官方 prompt.py）、选项前缀、字母表、截断分配恒等式
  （∑ = 1）、缺字母 floor、`answer_from_probs`。
- **后端层**：假 `urlopen` 桩喂"服务端原始响应"（照抄 hsbg 现有做法），覆盖：
  正常 / HTTP 404 报错文案 / tokenize 空 / 缺 `completion_probabilities`。
- **契约层**：`examples/*.json` golden —— mock 后端下 `ask(req)` 输出与期望逐字段一致
  （防手滑改字段）。
- **MCP 层**：管道喂 4 条消息（initialize / tools.list / tools.call / ping），
  断言逐行 JSON-RPC。
- **`selftest --mock`**：一条命令跑通全链路（也当 CI 入口）。

## 12. 迁移计划

| 阶段 | 内容 | 验收 |
|---|---|---|
| **M0** | 本设计定稿（含契约 schema） | 决策点（§13）拍板 |
| **M1** | 抽 core + backends + server + CLI（mock / llama-server / 假桩全测） | 离线测试全绿；`selftest --mock` 过；与 hsbg 现状对读法 golden 一致 |
| **M2** | hsbg 接 CLI 形态 + 降级；`consult` 用 `decide_many` | hsbg 全量测试绿；`hsbg advise` / 面板冒烟一致 |
| **M3** | MCP server + 配置示例 | MCP 管道测试过；Claude Desktop 实机接通一次 |
| **M4** | 清理：hsbg `provider.py` 瘦身、`model_server.py` → 转发 | hsbg 无重复实现；文档更新 |
| **M5** | 概率分析增强（按需）：`n_samples` 采样、校准报告、模型 A/B | 用真实场景验收 |

## 13. 开放决策点（需要拍板）

1. **名字**：`llm-odds`（推荐）/ `prob-oracle` / `model-odds` / 其它。
2. **依赖政策**：纯标准库 + 手写 MCP 子集（推荐，和现有生态一致）/ 引官方 `mcp` SDK。
3. **hsbg 集成形态**：CLI 先行、MCP 二步（推荐）/ 直接 MCP 常驻。
4. **HTTP daemon**：v2 再说（推荐）/ v1 就要。

## 14. 风险与对策

| 风险 | 对策 |
|---|---|
| 换模型后"字母纪律"变差（答非字母 / 分布挤平） | `warnings: low_signal`；换模型验收清单（附录 B） |
| 插件没装 / 挂了拖垮宿主 | 错误码分类明确 + 宿主降级路径（hsbg 已有 rules-only）+ `available()` 短缓存 |
| 契约漂移（宿主与插件版本错位） | 契约进 `schemas/` 受版本控制；响应带 `v`；插件对未知字段宽容、缺字段给默认 |
| MCP 生态兼容性 | 只实现实锤子集 + 管道测试；真有宿主需要高级特性再引 SDK |
| 多项目抢同一 llama-server | PID / 日志按 profile 隔离；`serve status` 显示端口与归属 |

## 附录 A：搬运清单（hsbg → llm-odds）

| hsbg 现状 | 去向 |
|---|---|
| `LETTERS` / `SYSTEM_MESSAGE` / `CHAT_TEMPLATE` / `CALIBRATION_TEMPERATURE` / `MISSING_MARGIN` | `core/protocol.py` + `core/calibrate.py`（值不变） |
| `build_payload` / `build_prompt` / `JevK5Prompt` | `core/protocol.py` |
| `JevK5Client`（HTTP / 读字母 / 概率化） | 拆：读字母 → `backends/llama_server.py`；概率化 / 校准 → `core/calibrate.py` |
| `Advisor` / `JevK5Advisor` 抽象 | 插件侧不搬（改为 `Backend` 接口）；hsbg 侧保留 `Advisor`，新增 `OddsAdvisor` 实现 |
| `model_server.py` | `server.py`（泛化 glob / PID 命名 / profile） |
| `panel.ModelAdvisorRunner` 的 TTL 缓存 | `core/session.py`（键 = 请求内容 hash） |
| `tests/test_provider.py` 77 项 | 读法 / 概率化 / 校准部分搬；hsbg 留薄壳（子进程桩） |
| `advisor.py` evidence / questions / consult | 不搬（宿主语义） |

## 附录 B：换模型验收清单（写进 README）

1. 字母命中率：正常提问下 `missing_letters == 0`（反复出现 → 调大 `top_k`）；
2. 分布形状：一眼"该赢的选项"概率显著最高（挤平 / 挤歪 → 校准温度没对）；
3. 校准温度：官方有值用官方；没有就重标（M5 工具）；
4. P95 延迟 < 面板 TTL（12s）的 1/3；
5. 跑一遍 `selftest` + 契约 golden。
