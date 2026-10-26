# 权重与部署文件清单

部署文件发布在 [lightx2v/microduck](https://huggingface.co/lightx2v/microduck/tree/main)，按 `models/microduck/` 与 `models/Wan2.2-TI2V-5B/` 保留目录结构。`fastwam_handoff.py fetch` 使用锁文件中的固定 commit，自动下载并校验。

2026-10-26 在 H200 上逐文件计算 SHA256、核对字节大小。**没有将权重上传 GitHub，也没有移动原训练文件。** 结构化来源记录见 [weight-inventory.json](weight-inventory.json)，部署期预期哈希及 Hugging Face 发布版本见 [model-files.lock.json](../../deploy/microduck/fastwam/model-files.lock.json)。

以下都是 H200 服务器上的路径。项目根：

```text
/mtc/fuhaiwen1/microduck-tinyvla/world-action
```

| 文件/组件 | 字节数 | 十进制大小 | 用途 |
|---|---:|---:|---|
| 已训练导航模型 | 12,042,063,744 | 12.0421 GB | 必须 |
| UMT5 文本编码器 | 11,361,920,418 | 11.3619 GB | 配套必需 |
| Wan2.2 VAE | 2,818,839,170 | 2.8188 GB | 配套必需 |
| 归一化统计 | 27,434 | 0.0000 GB | 配套必需 |
| 训练工件绑定 receipt | 4,148 | 0.0000 GB | 配套必需 |

最小推理集合合计 **26,244,309,246 bytes，26.24GB / 24.44GiB**（已含 tokenizer/架构配置）。原骨干目录合计 34.20GB，其中约 20GB 原始 diffusion 分片不用于此 native 导航权重的加载。官方 ONNX 与仿真资产另外由获取脚本下载。

## 已训练模型和绑定文件

```text
/mtc/fuhaiwen1/microduck-tinyvla/world-action/runs/clear-room-realtime-control-pilot-v1/checkpoints/weights/step_000250.pt
/mtc/fuhaiwen1/microduck-tinyvla/world-action/runs/clear-room-realtime-control-pilot-v1/checkpoints/state/step_000250/navigation_checkpoint.json
/mtc/fuhaiwen1/microduck-tinyvla/world-action/data/teacher-clear-qualification-v1/dataset_stats.json
```

模型 SHA256：`8ea7793290f0a22bff56ec4e44168232564ec5d57a9e21515a8a8c4a989114ea`。

这是包含 `mot` 视频/动作专家与 `proprio_encoder` 的完整任务权重，不是小型 LoRA。虽然运行参数叫 `adapter_model_path`，也不能误认为只需要一个小 adapter。

## 公共骨干组件

```text
/mtc/fuhaiwen1/microduck-tinyvla/world-action/models/Wan2.2-TI2V-5B/config.json
/mtc/fuhaiwen1/microduck-tinyvla/world-action/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
/mtc/fuhaiwen1/microduck-tinyvla/world-action/models/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth
/mtc/fuhaiwen1/microduck-tinyvla/world-action/models/Wan2.2-TI2V-5B/google/umt5-xxl/special_tokens_map.json
/mtc/fuhaiwen1/microduck-tinyvla/world-action/models/Wan2.2-TI2V-5B/google/umt5-xxl/spiece.model
/mtc/fuhaiwen1/microduck-tinyvla/world-action/models/Wan2.2-TI2V-5B/google/umt5-xxl/tokenizer.json
/mtc/fuhaiwen1/microduck-tinyvla/world-action/models/Wan2.2-TI2V-5B/google/umt5-xxl/tokenizer_config.json
```

公共来源固定为 Wan-AI/Wan2.2-TI2V-5B 的 `921dbaf3f1674a56f47e83fb80a34bac8a8f203e`。本次发布镜像这些最小必需文件，保持原始字节和上游许可；获取命令只拉取锁定允许列表。Hugging Face 中同时提供模型卡、文件清单、SHA256SUMS 与第三方声明。

## 不需要交给推理优化同事的文件

- `checkpoints/state/step_000250/pytorch_model/bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt`：72,250,664,839 bytes。
- 同目录 `mp_rank_00_model_states.pt`：14,861,940,593 bytes。
- 原始训练数据、回放缓存、其他候选训练权重、公开 LIBERO 初始化权重、TinyVLA 权重。

这些用于训练/恢复/研究，不能通过遗漏它们而宣称推理模型被压缩。本次交接固定当前保留模型；未采用候选不默认混入包中。需要之后扩展模型选择时，每个候选都应配自己的统计、receipt 和评测。
