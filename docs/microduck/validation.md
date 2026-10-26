# 本次导出验证（2026-10-26）

本页记录交接版本自己的验证；历史 59/60 等能力分数另见 capabilities.md。

- 独立 Docker 已构建：`lightx2v-microduck:fastwam-v1`。
- 固定基础镜像与已安装依赖可查；当前镜像 ID 为 `sha256:f377997ed4d437dc3bb505a75d2e52f33c2a84e9d327a97263f79b1f6b0f5b38`。
- 当前开发环境及独立 Python 3.11 容器分别通过 79 项 CPU 回归；4 项需要额外条件的 integration 测试未纳入此命令。
- 固定官方 BAM/本体下，真实头部相机和状态输入已生成，使用 OSMesa，不依赖桌面显示。
- 最小模型集合逐文件 SHA256 通过，真实 H200 前向成功；没有挂载原始 Wan diffusion 分片或训练状态。
- H200 普通/优化版分别完成 3 输入 × 20 次稳态前向；固定输入最大动作差异 0.0，Profiler trace 已生成（约 51.6MB，不入 Git）。MuJoCo + BAM + 官方 ONNX 的 100 tick 离线闭环已完成，记录/模型输入/动作日志均生成。该 2 秒短集未完成寻球，这是截短 smoke，不计入任务成绩。

## 当前测试范围

模型输入来自官方场景 episode 初始帧与三种颜色指令。它验证部署和固定输入计算，不代表训练任务验收，更不代表新布局或运动过程的全部输入覆盖。未启动训练。

真实 H200 报告在 `evidence/handoff-benchmark-{plain,optimized}.json` 与 `evidence/handoff-offline-smoke.json`；保存逐次计时、GPU/软件版本、模型/输入/源码哈希。`deployment-installed-packages.txt` 为观察到的安装清单，不是独立 pip 安装文件。冷加载、稳态和历史实时延迟分开解释。

本地交接目录与原开发目录隔离，原训练工作树和已有服务未修改。原始权重保留在 H200 原路径；验证目录使用同卷硬链接构造最小模型集合，没有将权重传到本机或 GitHub。Hugging Face 发布记录见下文。

## 固定输入结果

| 变体 | 稳态 p50 | 稳态 p95 | 推理阶段 allocated 峰值 |
|---|---:|---:|---:|
| plain | 500.98ms | 505.39ms | 23.48GiB |
| optimized | 176.36ms | 179.15ms | 23.69GiB |

相同 3 个输入、固定 seed=0、BF16、20 步；这组输入动作最大绝对差异 0.0。p95 改善约 2.82 倍 / 64.6%。本表为模型调用耗时，不包含相机/网络/任务执行器，也不能代替运动中的多观测回归。allocated/reserved 峰值从加载结束后测量，不含加载阶段峰值与驱动内存。没有因此宣布 24GB 卡可部署。

验证目录位于 H200：

```text
/mtc/fuhaiwen1/microduck-tinyvla/world-action/handoff-20261026/
```

其中 `data/results/{plain,optimized}` 保存动作数组和报告，优化版包含 `trace.json`；`data/inputs/startup.npz` 是相同输入。它们不是权重发布目录。测试结束后不保留占用 GPU 的模型服务。

本次没有重新跑完整 60 集实时开发评测或 200 布局独立验收，也未验证新导出版本的 EGL 实时质量。记录相机已成功导出为 MP4（25fps、按真实 camera_ticks 保留缺帧间隔）。OSMesa 短集只用于部署链路检查，不用其速度代替实时指标。

## Hugging Face 发布与下载验证

部署文件已发布至 [lightx2v/microduck 的固定版本](https://huggingface.co/lightx2v/microduck/tree/dad4beaf998780df1f7c17aba46c63ecf2a4e72d)。10 个模型文件共 26,244,309,246 bytes；另有模型卡、清单和许可声明。`fetch` 使用同一完整 commit，并逐文件核对大小、SHA256 及 checkpoint/history/statistics 绑定。

- 上传前已计算全部模型文件的 SHA256；上传后核对 18 个发布文件，大文件使用 Hub 的 LFS SHA256/大小，普通文件下载后重算哈希。
- 使用部署镜像原有的 `huggingface-hub==0.29.2`，在没有 token 的独立 CPU 容器中运行实际 `fetch --component model`。两个绑定 JSON 从远端重新下载，其余 8 个文件用本地硬链接复用；随后全部 10 个模型文件通过完整本地哈希校验。
- 额外匿名读取任务模型、UMT5、VAE 各前 1MiB，验证 HTTP 206、文件总大小及内容与本地一致。这是公开下载可用性抽查，没有声称重新下载了完整 26GB。
- 下载相关回归测试 9 项通过，覆盖固定 revision、完整/部分下载、目录布局与损坏文件拒绝；启动脚本通过语法检查和代理/网络参数转发检查。

完整记录见 [hf-release-verification.json](evidence/hf-release-verification.json)。这次发布没有使用 GPU、启动训练或新增能力验收；官方本体与 ONNX 策略仍由脚本从固定上游获取。
