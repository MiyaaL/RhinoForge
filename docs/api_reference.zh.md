# API 参考

简体中文 | [English](api_reference.md)

发行包名为 `rhinoforge`，Python package 仍是 `rpu_backend`。导入后把 PyTorch
`PrivateUse1` 设备注册为 `rpu`，并创建 `torch.rpu` namespace。

支持按配置判定。Class、adapter、registry alias 或 native schema 可导入并不代表模型
受到支持；选择入口前查[模型支持](model_support.zh.md)。

## 顶层 package

`rpu_backend.__all__` 只包含四个名称：

| 名称 | 契约 |
|---|---|
| `__version__` | 已安装 RhinoForge distribution 版本 |
| `RPUCache` | fused causal attention 的 FP16 KV cache |
| `RPUModelForCausalLM` | 已注册 decoder-only CausalLM 配置 loader |
| `reset_graph_cache()` | 清理进程 Graph cache；无 native backend 时为 no-op |

API compatibility level 位于 `rpu_backend.api.API_VERSION`，当前为 `5.0.0`，与
distribution version 独立。

## Causal language model

```python
import torch
from rpu_backend import RPUCache, RPUModelForCausalLM

model = RPUModelForCausalLM.from_pretrained(
    checkpoint,
    dtype=torch.float16,
    device="rpu",
    rpu_execution={
        "prefill": {
            "chunk_size": "auto",
            "padding_rows": "auto",
            "padding_budget": 64,
        }
    },
)
cache = RPUCache.from_model(model, input_ids=input_ids, max_new_tokens=32)
```

### `RPUModelForCausalLM.from_pretrained`

```python
RPUModelForCausalLM.from_pretrained(
    hf_repo_or_path,
    *,
    dtype=torch.float16,
    device=None,
    rpu_execution=None,
    **hf_kwargs,
)
```

Loader 在完整权重加载前读取 Hugging Face 配置，按 `config.architectures[0]` 选择
adapter，并提前拒绝 unsupported profile。当前入口接收 built-in Qwen3、Llama CausalLM
architecture，只接受 `torch.float16`；量化 checkpoint 自带受检查 metadata。

`rpu_execution` 是 read-only per-handle mapping。CausalLM 的 `prefill` 接受：

- `chunk_size`：`"auto"` 或正的 16 倍数；
- `padding_rows`：`"auto"` 或非负准确行数；
- `padding_budget`：非负整数，与精确 `padding_rows` 互斥。

Adapter 可以按配置进一步缩窄。Execution setting 是冷配置：RPU 安装后改变它，需要
创建新模型。

通用 CausalLM、image-text loader 和 `Pi05Policy.from_pretrained` 只支持 built-in model
code；`trust_remote_code` 必须为 `False`，custom Hugging Face code 在配置/权重加载前拒绝。

### `RPUCache`

优先使用 factory：

```python
RPUCache.from_model(
    model,
    *,
    input_ids=None,
    max_new_tokens=...,
    max_seq_len=None,
    batch_size=1,
    device="rpu",
)
```

必须且只能指定一种 sizing：`input_ids + max_new_tokens`，或正的 `max_seq_len`。
Cache 提供 `reset()`、`reset_to_position(pos)`、`get_seq_length()`、
`get_max_length()`、`get_cache(layer_idx)` 和 `to_dynamic_cache(device="cpu")`。
Batch>1 只限明确允许的 Qwen3 路径，并要求各行真实 token 长度相等。

`rpu_backend.api.Qwen3_5Cache.from_config(text_config, max_seq_len, ...)` 增加
Qwen3.5 text 的 recurrent/convolution state。它可完整 reset 到 position 0，但无法通过
rewind 重建非零 recurrent state。

## Image-text model

`rpu_backend.api.RPUModelForConditionalGeneration` 是 built-in Qwen3-VL 路径的
Hugging Face 风格 loader：

```python
from rpu_backend.api import RPUModelForConditionalGeneration

model = RPUModelForConditionalGeneration.from_pretrained(
    checkpoint,
    dtype=torch.float16,
    device="rpu",
    rpu_execution={
        "prefill": {"chunk_size": "auto"},
        "vision": {"chunk_size": "auto"},
    },
)
```

它与 CausalLM loader 一样 CPU-first、profile fail-fast。`vision` 接受 `chunk_size`，
`prefill` 接受前述字段。Class 存在不自动扩展视频或
[模型支持](model_support.zh.md)之外的配置。Qwen3.5 text/vision 与 Gemma4 使用各自
adapter。

## Policy API

`rpu_backend.api` 导出：

| Class | 主要构造与推理方法 |
|---|---|
| `Pi05Policy` | `from_pretrained(...)` / `from_lerobot_policy(...)`；`to("rpu")`；`prepare_graphs(...)`；`predict_action_chunk(...)`；`select_action(...)` |
| `RhinoVLAPolicy` | `from_runtime(...)` / `from_factory(...)` / `from_pretrained(..., runtime_factory=...)`；`prepare_graphs(...)`；`predict(...)`；`predict_action_chunk(...)` |
| `WallOssPolicy` | `from_checkpoint(...)` / `from_pretrained(...)`；`to("rpu")`；`prepare_graphs(...)`；`infer(...)`；`predict_action_chunk(...)` |
| `WallQwen35Policy` | `from_checkpoint(...)`；`to("rpu")`；`predict_action_chunk(...)` / `infer(...)`；`close()` |
| `Lingbot2Policy` | `from_checkpoint(...)`；`to("rpu")`；`prepare_graphs(...)`；`infer(...)`；`predict_action_chunk(...)`；`close()` |
| `HyEmbodiedPolicy` | `from_checkpoint(...)`；`to("rpu")`；`infer(...)`；`predict_action_chunk(...)`；`close()` |

`WallOssActionOutput`、`WallQwen35ActionOutput`、`Lingbot2ActionOutput`、
`HyEmbodiedActionOutput` 是对应结构化结果。带物理单位的 action 字段只表示已应用配置
normalization，不认证机器人安全或坐标系。

`WallQwen35Policy` 是绑定到一个精确本地准入 checkpoint 的 Source-only 受控评估
API。初版范围固定为 FP16 batch 1、精确三个规范相机、初始多模态 prefix 不超过
384 token、Dataset-V2 单次 BICUBIC 图像预处理、robot ID `10070`、normalizer
`x2_normal`、state/action mask `[1]*20+[0]*6`、action shape `[1,32,26]` 和
10 个 Euler step。受控 runtime 会把 base-text prefill 映射到固定 64 行 bucket
`64..384`；retained Prefill 签名还会区分末个 native chunk 的三种拓扑类别
（1 行、2--31 行、至少 32 行），同一类别内的有效 prefix 长度和 M-RoPE 仍按调用刷新。
native pad-zeroing 即使在 bucket 边界也会发出稳定的 fill 节点。优化开启时，Action
也按 64 行分桶（`64..384`），在桶边界插入 action KV，通过 additive mask 屏蔽
真实 prefix 到桶边界的空隙，逻辑 RoPE 位置不变。最多保留六个 Graph，共用固定
SPM 布局；各桶独立、地址稳定的 mask DDR 和 prefix KV 在 capture 外刷新，支持
同桶变长和 A/B/A 跨桶复用。关闭优化仍使用精确 prefix、逐步 Action 路径。
padding 可能改变 FP16 attention 舍入，不承诺逐位一致。
这些只是 Wall profile 的内部优化，不扩展通用 Qwen3.5 API。
统一冷环境开关 `WALL_QWEN35_OPT` 在 `from_checkpoint()` 时绑定：
未设置或 `1` 为 Vision1 + Prefill1 + Action1；`0` 恢复逐图 Vision、
逐步 Action（录制 open-loop 数据集为 3+3+10）。不再提供独立的
`action_execution` 参数。两种配置均使用 FP16 投影/Euler、ACC32 GEMM；
时间/Ada 在 CPU FP32 一次性计算后转 FP16 上传，归一化和输出保持 CPU FP32。
历史 FP32 host profile 不再可选；该精度降级不代表数值/任务认证。
开关覆盖六项 Graph/SDK 预算：开启 32768/8/64 MiB、SDK 65536/16/128 MiB；
关闭 8192/4/32、SDK 65536/8/64。切换需新进程及新 policy，要求匹配的
Python/native 包。优化 Vision/Prefill 及两种 Action 路径均要求每次调用一个
物理 segment，无法满足时拒绝，不静默切段。

`GraphCache(..., require_single_segment=True)` 是构造时固定的逐 cache 策略，使用
SDK 容量的一半（最多 65536 entries / 32 MiB command / 256 MiB instruction），
不扩大其他 Graph 的软预算。要求唯一 segment 覆盖全部节点；空图、提前同步/降级、
嵌套执行、host 边界或资源切段均在提交前拒绝，SDK 硬限制仍保留。
Wall 将三张单帧图像合并为一次 retained Vision Graph 调用；每图为偶数 patch 网格，
不超过 14x14，合计最多 588 patches。dense 运算共享 packed 行，每层 attention
仍按真实图长隔离调用三次。encoder 的两处残差 AllReduce 也按原图边界分别执行，
保留逐图执行的归约几何和浮点累加顺序；dense 计算仍共享一个 packed chunk。
更新此 adapter 时需要重新编译 native 扩展；该源码改动
不改变 numeric-blocked 状态，也不代表已经验证性能提升。
`predict_action_chunk(..., initial_noise=...)` 接受 shape 为 `[32,26]` 或
`[1,32,26]`、可转换成 CPU FP32 且所有元素有限的 tensor，用于精确的跨设备 flow 输入
对齐；它与 `noise_seed` 互斥，两者均省略时保持确定性的 seed-0 行为。Qwen3.5 vision
路径仍为 numeric-blocked，因此 `.to("rpu")` 前必须通过
`from_checkpoint(..., allow_numeric_blocked_vision=True)` 显式 opt-in。
`WallQwen35ActionOutput.actions` 是 shape `[1,32,26]`、物理单位、独立存储的连续
CPU FP32 tensor。runtime 负责冷启动多 handle 配置
`RPU_FUSED_COEXIST_KEEP_PERSISTENT_GEN=1`；若调用方显式配置为假，会在加载权重前
拒绝，且执行必须从新进程开始。板上数值对齐、独立 Graph 生命周期和带阈值的任务
验证仍为 pending。

`RhinoVLAPolicy` 把 checkpoint 组合与 preprocessing 委托给显式 model-repository
runtime factory。Runtime 必须声明 execution capability 和实际消费的 `rpu_execution`；
facade 拒绝静默 mismatch。

字符串 `runtime_factory="module:callable"` 会 import 并执行该 module 中的 Python code。
只使用已安装、已审查的模型集成，不从不可信 TOML/request 复制该值；已导入 callable
同样需要信任。

`HyEmbodiedPolicy.from_checkpoint` 默认不读取 `norm_stats.pkl`。解码物理 action 需要
`norm_stats_path`、显式 `trust_norm_stats_pickle=True` 和准确文件的
`norm_stats_sha256`；同一字节先验证 hash，再 deserialize。

多个 policy 配置仍是 Experimental、Component-only 或 Source-only。只使用 release
指定的 constructor、input schema 和 precision，不能从 method 存在推断相邻配置。

## Error hierarchy

用户可见错误在 `rpu_backend.api`：

```text
RPUBackendError
├── RPUConfigError
├── UnsupportedModelError
├── RPUUnsupportedDtypeError
├── RPUSingleHandleError
├── SPMExhaustionError
└── WeightShapeMismatchError
```

通用捕获 `RPUBackendError`；调用方有特定恢复动作时捕获 subclass。非法 Python 参数仍
使用标准 `TypeError`、`ValueError`。

## Graph API

公开 namespace 为 `rpu_backend.graph`：

```python
from rpu_backend.graph import GraphCache, GraphSignature

cache = GraphCache(max_entries=4)
sig = GraphSignature(
    op_id="my_model_forward",
    shapes=(seq_len, hidden_size),
    dyn_dims=(num_layers,),
    dtypes=(torch.float16,),
)

with cache.capture(sig):
    output = torch.ops.rpu.my_model_forward(...)
```

`GraphCache` 提供 `capture`、`begin_warmup`、`freeze`、`is_frozen`、`lookup`、
`evict`、`clear`、`size`、`max_entries`、`snapshot`、`cache_invariant_ok`。Frozen cache
只允许 lookup，拒绝 online BUILD。应用通常让 adapter 管理 Graph cache；手工 Graph
主要用于 porting。

公开诊断包括 `debug_bucket_counts()`、`debug_branch_counts()`、
`dump_signature_tree()`、`explain_miss(signature)`；`Graph.debug_stats()` 返回最近提交的
聚合 counter。它们解释生命周期，不是数值或支持证据。`Graph.dump_replay_plan()` 返回
pointer-free node/segment 摘要，`Graph.dump_tree()` 按 segment 分组；均不包含 launch
argument word、设备程序或原始 address。
segment 摘要中的 `entries`、`command_bytes`、`instruction_bytes` 来自共享 planner，
表示预留的 batch 资源，而非实测传输量或执行时间。一个缓存 Graph 可以包含多个
物理提交 segment，二者应分别检查。
低层 native module 提供 `graph_segment_budget_abi=1`，受控 runner 据此拒绝
会忽略预算控制项的旧扩展。

`Graph` 是低层 capture object；`get_default_graph_cache()` 返回 thread-local cache。
新 adapter 通常使用显式 per-model `GraphCache`，便于 ownership 与 teardown。

## Adapter registry 与 plugin

Registry 位于 `rpu_backend.runtime.registry`：

```python
from rpu_backend.runtime.registry import get_adapter, list_adapters, register_adapter
```

Built-in binding 在 `rpu_backend.adapters._manifest.BUILTIN_ADAPTERS`，当前包含
`Qwen3ForCausalLM`、`LlamaForCausalLM`、`DINOv3ViTModel`、
`Gemma4ForConditionalGeneration`、`Qwen3_5ForConditionalGeneration`、
`Qwen3VLForConditionalGeneration`。

外部分发包可声明 `rpu_backend.plugins` entry point：

```toml
[project.entry-points."rpu_backend.plugins"]
my_adapter = "my_package.rpu_plugin:register"
```

```python
def register(api):
    api.register_adapter("MyModelForCausalLM", MyModelAdapter)
```

Plugin 在 unknown-architecture lookup 后延迟加载，或由 `discover_plugins()` 显式发现；
不能覆盖 built-in/已注册 binding。Entry point 会执行安装包代码，只启用受信任 plugin。
Registry 是进程全局且无公开 unregister；改变 plugin set 后启动新进程。

## 模型路径 registry

```python
from rpu_backend.model_registry import MODELS, cache_root, model_path
path = model_path("qwen3-0.6b")
```

`cache_root()` 优先使用 `RPU_MODEL_CACHE`，否则为 `~/.cache/rhinoforge/models`。
`model_path(name)` 拼接受检 relative path，unknown alias 抛 `KeyError`。Alias 只是路径
映射，不是支持或资产可用声明。

## Adapter 作者的 weight-layout API

标准 helper 位于 `rpu_backend.runtime.weights`：

- `get_linear_partition(in_features, out_features, dwidth=2, num_cores=8)`；
- `swizzle_linear(weight, partition, num_cores=8)`；
- `swizzle_model_inplace(model, ..., skip_names=...)`。

Swizzle 原地改变 parameter storage，必须只运行一次。Fused subsystem 自有专用 Linear
应进入 `skip_names`，由该 subsystem 的一个权威转换路径处理。

## `torch.rpu` 设备控制

常用函数：

- `is_available()`、`device_count()`、`current_device()`；
- `set_caching_allocator(bool)`、`empty_cache()`、`memory_stats()`；
- `set_ddr_flush(bool)`、`set_ddr_flush_force(bool)`；
- `set_debug_level(0..5)`、`get_debug_level()`；
- `set_cross_layer_batch_prefill(bool)`、`set_cross_layer_batch_size(int)`；
- `shutdown()`。

开发诊断：

- `set_debug` / `get_debug` / `set_profile` / `get_profile` /
  `reset_profile_accumulators`；
- `set_debug_export` / `get_debug_export` / `list_debug_tensors` /
  `get_debug_tensor` / `clear_debug_tensors`；
- `set_spm_debug` / `get_spm_debug` / `spm_alloc_dump`。
- r4 硬件 kernel/DMA Chrome trace：
  `set_hw_perf_trace(enabled, output_dir, max_dumps)`、
  `get_hw_perf_trace()`，以及上下文管理器
  `hw_perf_trace(output_dir, max_dumps=32)`。

Debug tensor 可能含模型输入或 activation。把这些视为敏感应用产物，不要附到公开
issue；API 不暴露 raw Graph register/resource/plan payload。

硬件 trace 配置是进程全局状态，只能在 forward 之间修改。Rhino Launch 会在
`build_batch()` 时固化采集状态，因此每次配置变化都会使全部已注册 Graph 失效。应在
第一次 forward 前启用，最好使用新进程。`get_hw_perf_trace()` 返回 `enabled`、
`output_dir`、`max_dumps` 和 `dump_count`。文件名形如
`rpu_hwperf_*_<graph>_{build,replay,oneshot}_segN.json`；分析一个 Graph 时要合并全部 segment，
稳态结构主要查看 replay 文件。r4 Release trace 保留 kernel/DMA timing 与调度元数据，
但会移除可读 kernel 名、op type 和原始地址。它是敏感应用诊断产物，不是完整的
cache/stall/utilization PMU profiler。

CPU/RPU boundary flush 默认打开，正常推理不要关闭。Chunk size 属于 per-handle
`rpu_execution`，不是进程全局 `torch.rpu` setter。内部 `torch.ops.rpu.*` 是 adapter
使用的 host wrapper interface；未在本页或 model example 中记录的函数视为实现细节。
