# 模型执行与性能采集

简体中文 | [English](model_testing.md)

RhinoForge 提供两层手工执行入口：

- `examples/` 下的脚本是最短、最易读的模型入口；
- `examples/run_model.py` 调用同一批脚本，并从 TOML 应用已记录的运行时环境以及可选的 Torch、RPU 硬件 profile。

这些入口用于板卡冒烟和诊断。一次成功运行只能证明所选本地配置能够执行；它不能替代 CPU/golden 数值对比，也不会扩大[模型支持范围](model_support.zh.md)中的状态。

## 入口清单

| 模型/配置 | 直接入口 | 统一 runner 目标 | 可用范围 |
|---|---|---|---|
| Qwen3 FP16、Llama-3.2-1B | `examples/causal_lm.py` | `causal_lm` | 有匹配的精确发布权重和配置时可运行 |
| Qwen3 14B W8A16 | `examples/causal_lm.py` | `causal_lm` | Source-only 本地转换入口；v1.0.0 未绑定公开不可变派生 checkpoint 身份或 hash |
| Qwen3.5 文本 | `examples/qwen3_5_text.py` | `qwen3_5_text` | 可运行文本入口 |
| Qwen3.5 Vision 2B/4B | `examples/qwen3_5_vision.py` | `qwen3_5_vision` | Experimental/numeric-blocked 图像入口；受控评估必须精确设置 `QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED=1`，不构成支持声明 |
| Qwen3-VL 2B/4B | `examples/qwen3_vl.py` | `qwen3_vl` | 可运行图文入口 |
| Qwen3-VL 8B dense FP16 | `examples/qwen3_vl.py` | `qwen3_vl` | Limited 精确 padded profile 入口；4/4 numeric/task/Graph 已通过，包含 dual-full，base 5/5、long-text 9/9 token 精确一致；相邻长度、视频和量化路径不继承 |
| Qwen3-VL 32B W8A16 | `examples/qwen3_vl.py` | `qwen3_vl` | 仅源码诊断入口；显式 graph-blocked opt-in 只用于复现第三个 decode step 起的已知全零 logits |
| DINOv3 ViT-B | `examples/dinov3.py` | `dinov3` | 可运行视觉入口 |
| SigLIP | `examples/siglip.py` | `siglip` | 可运行组件入口，使用匹配 Pi0.5 包中的视觉塔 |
| Pi0.5 Libero FP16 | `examples/pi05.py` | `pi05` | Supported 精确 profile；9/9 numeric/task 与 27/27 execution-plan record 已通过，另有独立 fresh-process Graph 生命周期门通过 |
| Pi0.5 Libero W8A16 | `examples/pi05.py` | `pi05` | Source-only 本地转换入口；未绑定公开不可变派生 checkpoint 身份或 hash |
| Pi0.5 Libero W4A16 | `examples/pi05.py` | `pi05` | Source-only 本地转换入口；未绑定公开不可变派生 checkpoint 身份或 hash |
| Pi0.5 减步数/闭环 | 公开策略 API | 无 | Source-only 且验证 pending；不提供公开冒烟声明，也不继承精确 FP16 或量化路径的结论 |
| Pi0.5 base | 无 | 无 | 仅源码别名；没有独立任务配置或冒烟声明 |
| Wall-OSS FP16/W8A16/W4A16 | `examples/wall_oss.py` | `wall_oss` | 公开 `wall-x` source-only 接入 |
| Wall Qwen3.5 精确 flow policy | `examples/wall_qwen35.py`；`run_wall_qwen35_openloop.sh` | 无 | 一个本地准入 checkpoint 的 Source-only 受控评估入口：FP16 batch 1、robot ID `10070`、`x2_normal`、固定 mask `[1]*20+[0]*6`、三个规范 Dataset-V2 相机输入、初始 prefix `<=384`、action `[1,32,26]`、10 个 Euler step；板上数值、Graph 生命周期和任务验证仍为 pending |
| Hy-Embodied-0.5-VLA FP16/W16 | `examples/hy_embodied.py` | `hy_embodied` | 精确公开 profile 的可运行归一化 action 入口 |
| Hy-Embodied-0.5-VLA W8/W4 | `examples/hy_embodied.py` | `hy_embodied` | Source-only runtime 转换入口；未绑定公开不可变派生 checkpoint 身份或 hash |
| RhinoVLA | `examples/rhinovla.py` | `rhinovla` | 集成入口；模型仓库提供运行时工厂 |
| Gemma4-E4B 文本 | `examples/gemma4.py` | `gemma4` | 仅源码文本入口；需要兼容的本地权重和依赖集合 |
| GR00T-N1.7-3B | `examples/gr00t.py` | `gr00t` | 仅源码 builder 入口，输入为调用者预处理的张量 |
| LingBot-VLA-V2 | `examples/lingbot2.py` | `lingbot2` | 公开 source-only 接入，图像由调用方提供 |
| Galaxea G0.5 | `examples/g05.py` | `g05` | 配置和集成边界检查；模型仓库提供正式 CPU policy 对象前执行保持 fail-closed |
| InternVLA-N1 + NavDP | `examples/internvla_navdp.py` | `internvla_navdp` | 调用者固定 manifest 和 embedding 的受控 NavDP 组件；不是组合端到端入口 |
| Qwen3 0.6B/1.7B/4B/8B W8A16、Qwen3 32B | `model_support.zh.md` 指定的公开 API | 无 | 仅源码；没有已发布的可运行配置或冒烟配置 |
| 不支持的配置 | 无 | 无 | 不可运行 |

表格只记录真实的公开入口；仅存在源码不会被列为测试路径。

## 仓库内配置模板

下表每个 stem 在 [`examples/configs`](../examples/configs) 下都只有一份 `.toml`。同一文件包含直接模型请求、runner 环境、Torch/硬件 profiling，以及目标接受的全部执行阶段。配置内容不会改变其支持状态。

| 系列 | 配置 stem |
|---|---|
| Qwen3 | `qwen3_0_6b`、`qwen3_1_7b`、`qwen3_4b`、`qwen3_8b`、`qwen3_14b_w8a16` |
| Llama | `llama_3_2_1b` |
| Qwen3.5 文本 | `qwen3_5_0_8b`、`qwen3_5_2b`、`qwen3_5_4b`、`qwen3_5_9b` |
| Qwen3.5 Vision | `qwen3_5_vision_2b`、`qwen3_5_vision_4b` |
| Qwen3-VL | `qwen3_vl_2b`、`qwen3_vl_4b`、`qwen3_vl_8b`、`qwen3_vl_32b_w8a16` |
| DINOv3 / SigLIP | `dinov3_vit_b`、`siglip` |
| Pi0.5 Libero | `pi05_libero`、`pi05_libero_w8a16`、`pi05_libero_w4` |
| Wall-OSS | `wall_oss`、`wall_oss_w8a16`、`wall_oss_w4a16` |
| Hy-Embodied / RhinoVLA | `hy_embodied`、`rhinovla` |
| Gemma4 / GR00T | `gemma4`、`gr00t` |
| LingBot2 / G0.5 | `lingbot2`、`g05` |
| InternVLA-N1 + NavDP | `internvla_navdp` |

量化模板只用于本地 converter/runtime 路径；模板存在不会绑定派生
checkpoint 身份，也不会改变 Source-only 状态。

## 直接示例

所有由统一 runner 管理的直接示例都接受 TOML，并可在不加载权重的情况下验证配置：

```bash
python examples/qwen3_vl.py --config examples/configs/qwen3_vl_2b.toml --check-config
python examples/qwen3_vl.py --config qwen3_vl.local.toml
```

复制 `examples/configs/` 下最接近的文件，再设置本地权重和输入路径。权重获取与哈希方式见[模型资产](model_assets.zh.md)。

通用视觉和 VLM 模板使用 `assets/example.ppm`；这是仓库内的极小合成图，只用于让公开入口不依赖私有图片即可运行。数值或任务验证前必须替换为代表性图像。VLA 相机输入仍由调用方提供，因为相机名和预处理属于精确模型 profile。

Wall Qwen3.5 入口是独立的精确 checkpoint CLI，不是 TOML runner 目标。必须提供
精确三个规范相机和 26 个数值的 state：

```bash
python examples/wall_qwen35.py \
  --checkpoint /path/to/exact/checkpoint \
  --image face_view=/path/to/face.png \
  --image left_wrist_view=/path/to/left.png \
  --image right_wrist_view=/path/to/right.png \
  --instruction "perform the requested action" \
  --state-json /path/to/state-26.json \
  --allow-numeric-blocked-vision
```

该 flag 是 numeric-blocked Qwen3.5 vision 路径的显式受控评估 opt-in，不会提升
Source-only 状态。可选 state mask 和自由度 mask 必须都精确等于
`[1]*20+[0]*6`。
增加 `--check-config` 会检查已提供的命令行路径和输入 shape；不带 request 参数时，
它只执行仓库约定的无依赖 CLI 探测。两种模式都不验证 checkpoint 准入、RPU 执行、
数值对齐、Graph 生命周期或任务质量。

对于固定的 put-spoon-to-bowl episode，仓库根目录 wrapper 对齐 Harrix 的数据、prompt、
分段和 action 解码合同，并且不会向机器人发送命令：

```bash
bash run_wall_qwen35_openloop.sh --check --no-sudo
bash run_wall_qwen35_openloop.sh --max-requests 1
bash run_wall_qwen35_openloop.sh --max-events 1 \
  --torch-profile-dir /tmp/qwen35-openloop-profile
bash run_wall_qwen35_openloop.sh --max-events 1 \
  --hw-perf-dir /tmp/qwen35-openloop-hwperf --hw-perf-max-dumps 32
bash run_wall_qwen35_openloop.sh
```

三图合并的 Vision 路径默认开启，不需要额外 flag。脚本使用所选 Python 环境中
**已安装**的 `rpu_backend`；仅修改源码不会更新该安装包。从当前源码重建并安装后运行：

```bash
source /home/hx/miyaa/work/env.sh
CMAKE_PREFIX_PATH=/home/hx/.local/opt/rhino-launch-kernel-v1.0.0-linux-aarch64 \
  python -m pip install . --no-build-isolation -Cbuild.tool-args=-j2
bash run_wall_qwen35_openloop.sh --max-requests 1
```

启动时会检查 packed Vision adapter 和 native ABI，输出实际 adapter 路径；旧安装包
会在加载 checkpoint 前被拒绝。请求结束时输出的 `Vision Graph: entries=1` 表示
三图共用一个缓存条目；增加 `--torch-profile-dir DIR` 会记录每个请求，首请求包含
初始化/BUILD，不额外插入 warmup/repeat，也不代表 READY 数值或任务质量门禁通过。
`segments.json` 记录 Graph 诊断，`meta.json` 记录
`vision_execution` 与实际安装包、native 扩展和运行时资产 provenance。

统一冷开关为 `WALL_QWEN35_OPT`，默认 `1`，无需逐阶段选项：

```bash
# Vision 1 + Language/Prefill 1 + Action 1
bash run_wall_qwen35_openloop.sh --max-requests 1 \
  --torch-profile-dir /tmp/qwen35-opt-profile

# 本录制数据集的 Vision 3 + Language/Prefill 3 + Action 10
WALL_QWEN35_OPT=0 bash run_wall_qwen35_openloop.sh --max-requests 1 \
  --torch-profile-dir /tmp/qwen35-split-profile
```

优化 Action 将真实 prefix 映射到固定 64 行桶（`64..384`），最多保留六个单段图，
在 capture 外刷新 padding 可见性和真实 RoPE。runner 检查累计 replay 数，并逐请求
记录 `action_prefix_bucket`。测性能时不要提供两种 profiling 目录参数，并复用相同
`--flow-noise` 文件；需覆盖同桶变长和 A/B/A 返回旧桶。缓存桶数不等于每请求提交数。
具体局部生命周期、数值差异和无 profiling 性能记录见
[Action 分桶验证记录](wall_qwen35_action_bucket_validation.md)，不改变 Source-only 状态。

两种配置都使用 FP16 Action 投影/Euler、ACC32 GEMM 和一次性 CPU FP32 时间/Ada
预计算。关闭优化只改变 Graph 组织方式，不恢复历史 FP32 host 路径。
旧 `--language-one-graph`、`--action-execution` 选项已移除。

开关统一覆盖六项 Graph/SDK 预算：开启为 32768 entries / 8 MiB command /
64 MiB instruction，SDK 为 65536 / 16 / 128；关闭为 8192 / 4 / 32，
SDK 为 65536 / 8 / 64。非 Wall 通用默认值不变；切换需新进程。

须检查实际 segment 数，而非缓存条目数。开启时 Vision、Prefill、Action 各要求
一段；关闭时 Action 复用单步图十次，Vision 保留各相机形状。历史 READY 探测核验
三次或十六次物理提交、稳定 replay 和同输入重复数值一致性，不计冷启动 priming/BUILD。
当前逐请求 profiler 包含冷启动工作，不执行该 READY 探测。
其他输入长度可能产生不同的保守 Prefill 分段数，`3+3+10` 对应此 runner 的录制
数据集。metadata 记录开关、图计划与实际预算；旧包在加载前拒绝。
比较时使用相同 `--flow-noise`，这些受控配置不代表发布质量认证。

packed dense 计算共享全部图像行；attention 仍按图隔离，encoder 的两处残差
AllReduce 也按原图边界调用，以保留逐图执行的归约几何。通用和 temporal Vision
路径不变。
Wall runtime 在 Vision/Text 前及 Text 后、Graph 捕获范围之外释放临时 SPM，
保留持久状态，避免 Action priming 遗留的临时区叠加到下一次 packed Vision 分配上。

板上命令默认使用 `sudo`，并 source `/home/hx/miyaa/work/env.sh`。输出包含与参考脚本
兼容的 NumPy 文件名、物理单位指标、精确 profile provenance 和 retained-Graph
诊断。脚本会在 Python 启动前固定多 handle 所需的冷配置
`RPU_FUSED_COEXIST_KEEP_PERSISTENT_GEN=1`。确定性 CPU noise seed schedule 会明确记录
为不与参考 CUDA RNG stream 逐位一致。

Torch profiling 统一使用 `--torch-profile-dir DIR`：请求 CPU 与 `PrivateUse1`
activity，每个推理请求分别导出为
`qwen35_generate_flow_action_batch_*.trace.json.gz`。文件名包含时间戳、PID、
请求序号和纳秒时间戳。固定开启 `record_shapes`、`with_stack`，关闭
`profile_memory`。每个请求新建 profiler，使用 `acc_events=True`，不累积前面
请求的事件。保留 `generate_flow_action_batch` range，以及
`wall_qwen35_preprocess`、`wall_qwen35_vision_text_prefill`、
`wall_qwen35_action_denoise_loop` 和 `wall_qwen35_action_decoder` 阶段。

不再自动插入 profiler 外的 warmup 或同请求 READY 重复探测。
显式 `policy.to("rpu")` 安装不在 Torch trace 内，但首请求的延迟初始化和
Graph BUILD 会记录；后续请求签名变化时也可能 BUILD。trace 仅为诊断证据，
不构成 frozen-READY 或数值一致性声明；`meta.json` 保留逐请求 artifact 清单，
不报告 READY admission。历史同请求重复测量见
[Wall Qwen3.5 READY-profile assessment](wall_qwen35_ready_profile_assessment.md)。
本 Wall runner 已移除旧 `--torch-profile`、`--torch-profile-output`、
`--torch-profile-record-shapes`、`--torch-profile-memory`、
`--torch-profile-with-stack` 参数。

硬件采集统一使用 `--hw-perf-dir DIR`，在 policy 安装前启用 r4 设备 trace。
旧 `--hw-perf`、`--hw-perf-output` 参数已移除。wrapper 将
`LKN_RPU_FREQ_MHZ` 透传给 sudo（默认 800 MHz），保留
`--hw-perf-max-dumps` 限制 segment JSON 文件数（默认 32）。
文件名包含时间戳、PID、BUILD/REPLAY/oneshot 阶段和 segment 序号。
在 Perfetto 中主要查看 `*_replay_segN.json`，并检查推理区域对应的全部 segment。
r4 Release runtime 会按设计脱敏这些 trace。

两个目录参数可同时使用，也可单独使用；未提供对应目录时，不启用该采集。
wrapper 支持等价环境变量 `TORCH_PROFILE_DIR`、`HW_PERF_DIR`，不再读取其他
旧的 profiling 启用、输出、shape、memory 或 stack 环境开关。
目录在 checkpoint 加载或 RPU 初始化前校验，可复用已有目录但不会覆盖 trace。
两种采集都不能与 `--check` 组合。trace 可能包含应用 shape、源码路径和算子元数据，
必须保存在仓库外；采集会扰动延迟，最终 latency 应关闭 profiling 后另跑。

Pi0.5 的 `batch_file` 不随仓库分发。请使用与权重匹配的 LeRobot policy 预处理流水线生成它，再用 `torch.save` 保存张量字典。该字典至少包含权重 `image_features` 配置中每个 key 对应的一份 batched image tensor，以及 `observation.language.tokens` 和 `observation.language.attention_mask`；预处理流水线也可保留该配置拥有的其他字段。相机名称、分辨率和 token 长度属于具体配置，应从权重/配置读取，不要照搬其他配置的 shape。

GR00T 入口同样需要调用者生成的张量映射。`input_ids`、`attention_mask`、`pixel_values`、`image_grid_thw` 和 `state` 必须来自与权重匹配的预处理流水线。NavDP 组件入口需要调用者固定的“路径到 SHA256”JSON manifest，以及包含 `goal_embed [1,1,384]` 和 `rgbd_embed [1,32,384]` 的张量映射。这些都是输入，不是 RhinoForge 分发的模型资产。

Qwen3.5 Vision 默认拒绝执行：公开 example 会在加载模型权重前拒绝，共享 adapter 会在创建 Vision handle 或转换 Vision 权重前拒绝。仅在受控评估时，于新进程启动前显式设置：

```bash
QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED=1 \
  python examples/run_model.py --config examples/configs/qwen3_5_vision_2b.toml
```

该取值不豁免已失败的官方真实图数值门，也不启用视频。Wall Qwen3.5 CLI 通过专用 flag 采用同样的 fail-closed 受控评估原则；它不使用统一 runner，也不表示 pending 的板上门禁已经通过。Qwen3-VL 32B 模板保留显式 opt-in，只用于复现已知 Graph 硬失败；它不代表硬门通过或可运行支持。LingBot2 和 InternVLA/NavDP 模板包含明确的 source-only 或受控评估确认项；删除门禁时必须失败，不能退化为普通配置。G0.5 没有稳定的独立公开 policy 构造器：其示例验证配置后，会明确给出正式模型仓库需要完成的集成步骤并停止。对这些入口执行成功的 `--check-config` 只验证配置结构，不会提升支持状态。

## 完整 TOML runner

每个可运行目标在 `examples/configs/` 下都有一份 TOML。同一文件包含模型/请求字段、公共运行时配置、Torch profile、硬件 profile，以及适用的全部 `rpu_execution` 阶段。完整字段和归属说明见[运行时配置](runtime_config.zh.md#toml-参数目录)。

以 `examples/configs/qwen3_0_6b.toml` 为例：

```bash
python examples/run_model.py --list-targets
python examples/run_model.py \
  --config examples/configs/qwen3_0_6b.toml \
  --check-config
python examples/run_model.py \
  --config examples/configs/qwen3_0_6b.toml
```

配置会把运行时环境、每 handle 执行规划、模型输入和 Torch profiling 合在一起：

```toml
[runner]
target = "causal_lm"

[runner.env]
RPU_LOG_LEVEL = 3
RPU_WARMUP = 1

[runner.torch_profile]
enabled = false
output = "profiles/torch.json"
record_shapes = false
profile_memory = false
with_stack = false

[runner.hw_perf]
enabled = false
output_dir = "profiles/rpu_hwperf"
max_dumps = 32
```

其余 `[model]`、`[generation]` 或 `[request]` 表由选中的直接示例读取。`[runner.env]` 只接受[运行时配置](runtime_config.zh.md)中记录的变量。TOML 布尔值会转换成可移植的环境值 `1`/`0`。runner 会在导入 `torch` 或 `rpu_backend` 前应用它们；每个进程只运行一个模型/配置。凭据、无关的进程加载器变量和 `RPU_KERNEL_LIB_PATH` 都不能写入 TOML；经批准的 Rhino Launch 库和合并算子资产应在部署环境中设置。“运行时配置”页面的**仅诊断变量清单**同样不能写入 runner；开发者必须在 shell 中逐项显式启用，并在分享前检查产生的本地文件。

每 handle 的执行规划放在顶层表中：

```toml
[rpu_execution.prefill]
chunk_size = "auto"
padding_budget = 64
```

统一 runner 验证该表并转交给现有模型入口。可接受的阶段如下：

| 目标 | 接受的 `rpu_execution` 阶段 |
|---|---|
| `causal_lm`、`qwen3_5_text`、`qwen3_5_vision` | `prefill` |
| `gemma4` | `prefill`（仅 `chunk_size`） |
| `qwen3_vl` | `prefill`、`vision` |
| `dinov3`、`siglip` | `vision` |
| `pi05`、`wall_oss`、`gr00t` | `prefill`、`vision`、`action` |
| `rhinovla` | 由可信运行时工厂声明，并通过能力握手检查 |
| `hy_embodied`、`lingbot2`、`g05`、`internvla_navdp` | 无；非空表会被拒绝 |

除 Gemma4 的窄范围外，`prefill` 接受 `chunk_size`、`padding_rows` 和 `padding_budget`；`vision` 与 `action` 接受 `chunk_size`。公开 loader 始终是配置边界的权威检查点，会在安装到 RPU 前拒绝不支持的值。

## Torch profile

设置 `[runner.torch_profile].enabled = true`，可为整个命令导出一份 Chrome trace。若安装的 PyTorch 同时暴露 CPU 与 `PrivateUse1` activity，runner 会请求二者。trace 包含模型初始化和推理；排查设备路径时可筛选 `rpu::` 以及命名的图范围。

`record_shapes`、`profile_memory` 和 `with_stack` 默认均为 `false`。只在需要时单独开启：shape、分配事件、栈帧和源码路径会向 trace 增加应用细节。

可使用 Perfetto 或其他 Chrome trace 查看器打开 JSON。profiling 会增加开销，因此延迟测量应使用关闭 profiler 的运行。

## RPU 硬件 profile

只有在 r4 Rhino Launch 构建提供硬件 trace API 时，才设置
`[runner.hw_perf].enabled = true`。输出目录可以已经存在；每次会话都写入不会覆盖旧文件
的 `rpu_hwperf_*.json`。配置检查会在不初始化设备的前提下接受此表；真实执行时，如果
安装的 RhinoForge extension 早于该集成，runner 会明确失败。上下文会在模型 target
之前进入，并在开启和关闭采集时重置存活的 Graph batch。

## 调试错误或不稳定运行

应先调试正确性，再采集性能 trace。保留一份不可变运行记录，其中包括权重/配置哈希、TOML、输入张量、执行参数和依赖版本。对随机策略，应只采样一次初始噪声或其他随机状态，再把完全相同的张量克隆给 CPU 与 RPU 路径；连续两次 RNG 调用得到的是不同输入。

使用能够区分问题的最小公开检查：

1. 比较第一次调用和相同 signature 的重复调用。只有重复调用变化时，应优先检查 Graph 生命周期、持久 SPM 或 DMA 地址所有权。
2. 保留第一份输出，运行相同 shape 的另一输入，再确认已保留输出的字节和 storage 相互独立。
3. 修改 capture 路径前，检查 `GraphCache.snapshot()`、`cache_invariant_ok()`、`debug_bucket_counts()` 和 `explain_miss(signature)`。仅有干净日志不能证明已经 BUILD 后稳定 REPLAY。
4. 隔离第一个发生偏差的公开组件或算子；在调试完整模型前，先用相同 CPU 输入对比 shape、dtype、layout 和数值。
5. 只按需提高 `torch.rpu.set_debug_level(...)`。使用 `torch.rpu.spm_alloc_dump("label")` 查看分配器摘要；只在受控本地对比中导出调试张量。

调试导出可能包含权重、用户输入派生数据和激活。请将其保存在仓库之外，调查结束后清理，并分享脱敏摘要而非原始张量。正确性和 replay 稳定后，再启动新进程，只开启测量问题所需的 profiler。

## 发布质量的模型检查还需要什么

支持声明必须把所有结果绑定到同一源码提交、Rhino Launch 包、合并算子资产、权重修订版本、输入边界、TOML 和输出哈希。应与 CPU 或批准的 golden 结果对比，覆盖 prefill 与 decode 或完整 policy 调用，warmup 后重复执行，并验证适用的图生命周期。未执行的门禁应记录为待完成，不能从冒烟运行推断。

请使用[模型验证策略](validation_policy.zh.md)中相互独立的硬语义、同 dtype parity、FP32 锚点和代表性任务门禁。单个最差行 cosine 只是诊断证据，不是全仓统一的产品结论。
