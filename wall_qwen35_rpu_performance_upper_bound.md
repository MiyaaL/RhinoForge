# Wall Qwen3.5 在 RPU 上的性能上限分析

对应表格：[wall_qwen35_rpu_performance_upper_bound.xlsx](wall_qwen35_rpu_performance_upper_bound.xlsx)

更新说明（2026-09-08）：源码已将 Wall 三图改为一次 packed Vision Graph 调用，
共享 dense 运算，每层暂时保留三次独立 attention；两次残差 AllReduce 各按原图
边界分三段，保持逐图参考的求和边界，仍在同一 Graph 内。重建安装后直接运行
`bash run_wall_qwen35_openloop.sh --max-requests 1` 即启用，无额外开关；
加 `--torch-profile` 可检查一次 BUILD 后的 REPLAY，但不代表整链数值门槛通过。
下文和工作簿仍是合并前的
三次调用基线账本，未改写为新版本实测。尤其 `1,881 MiB / 14.088 ms` 的 Vision
weight-stream 情景不能直接当作合并版结果；是否减少实际 DDR reread、以及收益大小，
仍需新版本板端正确性、Graph replay 和无 profiler 性能验证。

本次提交中的工作簿 SHA256 为 `d3d30638a20407e0a41895e9c49a5628175f0ec56f405eca05bc53246a8c0f07`；它包含 4 张可见工作表和 210 个公式，不含宏或外部链接。

这份表按当前仓库中的 Wall Qwen3.5 受控 profile、RPU 运行时和八核 SPM/DDR 架构重建了算子账本。你给出的 **FP16 峰值 100 TOPS、DDR 带宽 140 GB/s** 被作为可编辑的情景参数写入“阶段汇总”`B3:B4`。当前 `B3=100` 表示公式使用 **100 TFLOP/s**；这对应芯片规格把一次 MAC 的乘、加分别计作两个 operation。带宽按十进制 GB/s 计算。`B5` 是 launch/replay floor，当前为 0 ms，仅表示理想化 compute+DDR 情景下界，不能当作实测启动开销。

如果芯片规格把一次 MAC 整体记作 1 operation，那么 100 TOPS 等价于本表 `2 FLOP/MAC` 账本中的 **200 TFLOP/s**，应把 `B3` 改为 `200`；本表公式会自动重算（当前总请求仍由 140 GB/s 的 memory proxy 主导）。

## 结果摘要

| 情景 | 严格已知计算量 | 假设的 FP16 weight-stream proxy | 理想化情景下界* | 主导边界 |
|---|---:|---:|---:|---|
| 典型：real prefix 308 → bucket 320 | 1,241,637 MiFLOP | 14,081.53 MiB | 105.468 ms | DDR/memory |
| 上限：real prefix 384 → bucket 384 | 1,417,359 MiFLOP | 14,081.53 MiB | 105.468 ms | DDR/memory |

\* `max(Tcompute, Tmemory, Tlaunch)`；memory 项按“参数字节数 × 假定的重复 stream 次数”计算，并非从 DDR/SPM residency、tiling 和实际 reread 推导出的 mandatory-byte 下界。激活、KV/cache、DMA 编排、SPM 搬运、同步、norm/rope/residual/elementwise、host 和 launch/replay 也未计入，因此这些数字不能保证端到端延迟。

阶段分解（典型情景）为：Vision `236,523 MiFLOP / 1,881 MiB / 14.088 ms`，Text prefill `869,454 / 8,120.53 / 60.821 ms`，Action decoder `135,660 / 4,080 / 30.559 ms`。三段合计是分析用 critical-path proxy，不是公开的端到端 latency。

## 计算口径

- Vision：每个请求调用 3 次，但只保留 2 个 geometry signature（face 一种、两个 wrist 共用一种）；patch token 为 `112+140+140=392`，24 个视觉 block 和 merger 都计入。
- Text：24 层中 18 层 GDN、6 层 full attention；典型执行 bucket 为 `128/128/64`，上限为 `128/128/128`。算术按 bucket 行数计算，模型权重流量按每个 chunk 重复 stream。
- Full attention：逻辑 KV 2 头在 RPU 物理路径复制到 8 头；因此 QKV+gate 的物理输出宽度为 8192，而不是 Thor 表中的逻辑 5120。
- GDN：实际 `conv_dim=6144`，b/a 的每核对齐使物理投影宽度为 8448（逻辑 8224）。GDN delta/BMM 的 `1980/2376 MiFLOP` 是每核 critical-path proxy，单独列出，未混入严格 FLOP 合计；若做芯片算术总量可另乘 8，但仍遗漏部分逐元素/重排工作。
- Action Wall mode：18 层 identity mixer 不走 GDN/linear-attention GEMM；6 层 full attention，`[1,32,26]`，10 个 Euler step，ACC32。Action 的 full O 是 `[32,2048]×[2048,1024]`。

公式为：

```text
MiFLOP = 2 × M × N × K / 2^20
Tcompute(ms) = MiFLOP × 2^20 / (P_TFLOPS × 10^12) × 1000
Tmemory(ms)  = MiB × 2^20 / (BW_GB/s × 10^9) × 1000
```

## 实测/诊断边界

仓库没有可用于发布的 RPU vendor 峰值或持续 DDR 带宽；当前容器也没有 `/dev/rpu*`/`/dev/mem`，所以没有宣称新的整网板测结果。表中历史 r4 HWPerf 仅作诊断参考：Vision 约 30.138 ms、Text prefill 约 98.663 ms、Action 约 51.193 ms。另一个约 445.671 ms 的完整 public API CPU scope 来自已被本次 Graph 设计取代、且没有不可变 source receipt 的历史候选；它包含 host/Graph 调度和 profiler 开销，既不是 clean latency，也不是当前实现的证据。

“阶段汇总”底部的 pure-GEMM 面板来自本地 hxcc 诊断记录：Action ACC32 精确形状的 q/k/v/gate/up 约 3.823 TFLOP/s，o/down 约 3.976 TFLOP/s（Compute critical path；5 warmup + 40 replay）。这是算子级证据，不是全芯片理论峰值，也不能外推到 Vision/Text 或融合后的整网。

若要改做真实校准，只需修改工作簿“阶段汇总”`B3`、`B4`、`B5`；宽表的 per-op roofline 公式会跟随更新。发布前仍需同一 profile 的无 profiler warmup/REPLAY、数值、输入 envelope、生命周期和端到端验证门槛。
