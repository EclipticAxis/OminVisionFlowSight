# CBOR vs JSON Serializer Benchmark (Milestone B4)

> **日期**：2026-07-20
> **范围**：比较 `TargetStateSerializer`（JSON）与 `CBORSerializer`（CBOR）在序列化速度、反序列化速度、载荷大小三个维度上的表现
> **测试环境**：Python 3.11.9 / win64 / cbor2 6.1.3（C 扩展）

---

## 1. 测试方法

### 1.1 三种场景

| 场景 | 描述 | metadata 规模 |
|---|---|---|
| **minimal** | 最小快照，空 metadata，全默认字段 | 0 keys |
| **rich** | 典型生产快照，含嵌套 metadata（source/roi/keypoints/attributes） | 5 keys |
| **large** | 极端情况，100-key metadata，每个 key 含嵌套 dict+list | 100 keys |

### 1.2 三项指标

| 指标 | 单位 | 说明 |
|---|---|---|
| **serialize μs/op** | 微秒/次 | 单次序列化耗时（含 `to_dict()` + 编码） |
| **deserialize μs/op** | 微秒/次 | 单次反序列化耗时（含解码 + `from_dict()`） |
| **payload bytes** | 字节 | 序列化产物的字节大小 |

### 1.3 测量方法

- 每项测量迭代 **20,000 次**
- 预热 100 次让 CPU 缓存稳定
- 使用 `time.perf_counter_ns()` 高精度计时
- JSON payload 大小按 UTF-8 编码字节数计算（`len(json_str.encode("utf-8"))`）
- CBOR payload 大小直接取 `len(bytes)`
- **CBOR/JSON ratio**：< 1.00 表示 CBOR 更优（更快/更小），> 1.00 表示 JSON 更优

---

## 2. 测试结果

| Scenario | Metric | JSON | CBOR | CBOR/JSON ratio |
|----------|--------|------|------|-----------------|
| minimal | serialize μs/op | 12.79 | 12.59 | 0.98x |
| minimal | serialize ops/sec | 78,210 | 79,403 | 0.98x |
| minimal | payload bytes | 211 | 185 | 0.88x |
| minimal | deserialize μs/op | 6.62 | 6.31 | 0.95x |
| minimal | deserialize ops/sec | 151,170 | 158,523 | 0.95x |
| rich | serialize μs/op | 28.98 | 28.56 | 0.99x |
| rich | serialize ops/sec | 34,502 | 35,013 | 0.99x |
| rich | payload bytes | 444 | 397 | 0.89x |
| rich | deserialize μs/op | 8.51 | 8.18 | 0.96x |
| rich | deserialize ops/sec | 117,495 | 122,210 | 0.96x |
| large | serialize μs/op | 1,097.16 | 1,161.34 | 1.06x |
| large | serialize ops/sec | 911 | 861 | 1.06x |
| large | payload bytes | 9,910 | 6,788 | 0.68x |
| large | deserialize μs/op | 80.72 | 117.40 | 1.45x |
| large | deserialize ops/sec | 12,388 | 8,518 | 1.45x |

---

## 3. 分析

### 3.1 载荷大小 — CBOR 全面胜出

| 场景 | JSON bytes | CBOR bytes | CBOR 节省 |
|---|---|---|---|
| minimal | 211 | 185 | **12%** |
| rich | 444 | 397 | **11%** |
| large | 9,910 | 6,788 | **32%** |

**结论**：CBOR 在所有场景下都产生更小的载荷。节省幅度随 payload 复杂度增长——minimal/rich 场景节省 11-12%，large 场景节省高达 32%。

**原因**：
- CBOR 用 1-9 字节变长编码整数（小整数仅 1 字节），JSON 总是文本数字
- CBOR 的字符串长度前缀比 JSON 的引号+转义更紧凑
- CBOR 的 map/list 头部编码比 JSON 的 `{}`/`[]` 更省字节
- 大 payload 中这些差异累积放大

**实际影响**：在网络传输（尤其 UDP 单包 1472 字节 MTU 限制）下，CBOR 的 32% 体积优势意味着 large 场景可以从"需要分片"变为"单包可装下"。

### 3.2 序列化速度 — 小载荷持平，大载荷 JSON 略快

| 场景 | JSON μs/op | CBOR μs/op | 差异 |
|---|---|---|---|
| minimal | 12.79 | 12.59 | CBOR 快 2%（噪声范围） |
| rich | 28.98 | 28.56 | CBOR 快 1%（噪声范围） |
| large | 1,097.16 | 1,161.34 | JSON 快 6% |

**结论**：对于典型 TargetState（minimal/rich），两者速度基本持平（差异在测量噪声内）。对于 large 场景，JSON 略快 6%。

**原因**：
- 两者都先调用 `state.to_dict()`（共享开销），差异仅在编码步骤
- Python 标准库 `json.dumps` 是高度优化的 C 实现
- `cbor2.dumps` 虽然也有 C 扩展，但对嵌套 dict 的遍历开销略高
- large 场景中 100-key 嵌套 dict 的编码开销放大了这一差异

### 3.3 反序列化速度 — 小载荷 CBOR 略快，大载荷 JSON 显著快

| 场景 | JSON μs/op | CBOR μs/op | 差异 |
|---|---|---|---|
| minimal | 6.62 | 6.31 | CBOR 快 5% |
| rich | 8.51 | 8.18 | CBOR 快 4% |
| large | 80.72 | 117.40 | **JSON 快 45%** |

**结论**：对于 minimal/rich 场景，CBOR 反序列化略快 4-5%。对于 large 场景，JSON 显著快 45%。

**原因**：
- `json.loads` 的 C 实现对文本解析高度优化
- `cbor2.loads` 需要处理 CBOR 的类型标签和变长编码，每个字段的解码开销略高
- large 场景中 100 个嵌套 dict 的解码累积放大了每字段的开销差异
- 两者都最终调用 `TargetState.from_dict()`（共享开销），差异仅在解码步骤

### 3.4 综合评估

| 维度 | minimal/rich（典型） | large（极端） |
|---|---|---|
| 载荷大小 | CBOR 胜 11-12% | CBOR 胜 32% |
| 序列化速度 | 持平 | JSON 胜 6% |
| 反序列化速度 | CBOR 胜 4-5% | JSON 胜 45% |
| 人类可读 | JSON 胜 | JSON 胜 |
| 依赖 | JSON 零依赖 | CBOR 需 cbor2 |

---

## 4. 推荐

### 4.1 默认选择：JSON

对于 VisionCore 的大部分场景，**JSON 是更好的默认选择**：

- ✅ 零外部依赖（标准库 `json`）
- ✅ 人类可读（调试/日志友好）
- ✅ 典型 payload 速度与 CBOR 持平
- ✅ 已在 B3 实现（`TargetStateSerializer`）

### 4.2 何时用 CBOR

在以下场景中，**CBOR 更合适**：

| 场景 | 理由 |
|---|---|
| **UDP 网络传输** | 载荷大小是硬约束（MTU 1472 字节），CBOR 的 12-32% 体积优势关键 |
| **高密度文件录制** | 长时间录像产生百万级快照，体积差异累积显著 |
| **带宽受限链路** | 串口/低速无线场景，每字节都珍贵 |
| **跨语言互操作** | CBOR 是 RFC 8949 标准，C/C++/Rust/Go 都有成熟实现 |

### 4.3 何时 NOT 用 CBOR

| 场景 | 理由 |
|---|---|
| **调试/日志** | 人类可读性优先，JSON 胜 |
| **超大 metadata** | JSON 反序列化快 45%，且 to_dict/from_dict 开销主导 |
| **无 cbor2 环境** | 嵌入式/受限环境可能无法安装 cbor2 |
| **简单配置文件** | JSON 可手写编辑，CBOR 不行 |

### 4.4 架构建议

```
                    ┌── JSON (TargetStateSerializer) ──> ConsoleAdapter / 日志 / 调试
                    │
TargetState ────────┼── CBOR (CBORSerializer) ─────────> [未来] UdpAdapter / FileAdapter
                    │
                    └── 原始 dict (to_dict) ────────────> EventBus / 进程内
```

生产者不直接选择序列化格式——它注入 `ProtocolAdapter`，适配器内部选择序列化器：

```python
# 调试配置：人类可读
adapter = ConsoleAdapter(serializer=TargetStateSerializer())

# 生产配置：紧凑传输
adapter = UdpAdapter(serializer=CBORSerializer(), host=..., port=...)
```

---

## 5. 复现

```bash
cd F:/VisionBata
PYTHONPATH=F:/VisionBata python benchmark/benchmark_serializer.py 20000
```

可通过命令行参数调整迭代次数（默认 20000）：

```bash
# 快速测试（1000 次迭代）
python benchmark/benchmark_serializer.py 1000

# 高精度测试（100000 次迭代）
python benchmark/benchmark_serializer.py 100000
```

---

## 6. 文件清单

| 文件 | 说明 |
|---|---|
| `visioncore/protocol/serialization/cbor_serializer.py` | CBORSerializer 类 |
| `visioncore/protocol/serialization/__init__.py` | 更新：导出 CBORSerializer + TargetStateSerializer |
| `tests/test_cbor_serializer.py` | 31 项单元测试 |
| `benchmark/benchmark_serializer.py` | 基准测试脚本 |
| `docs/cbor-benchmark.md` | 本文件 |

---

## 7. 后续里程碑

- **B5**：`UdpAdapter` — 内部使用 CBORSerializer 紧凑传输
- **B6**：InferWorker 接入 ProtocolAdapter（影子旁路发布快照流）
- **B7**：`FileAdapter` — 录制元数据文件 sink（可选 JSON 或 CBOR）
- **C**：健康监控集成 + 自动回退 NullAdapter
