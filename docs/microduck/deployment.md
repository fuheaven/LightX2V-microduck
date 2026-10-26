# FastWAM MicroDuck 部署

目的：复现原生推理、定位耗时，用相同权重复测 MuJoCo 闭环。全流程不启动训练、不依赖 ROS2、不连接真机。

## 1. 容器

基线为单张 H200、BF16、20 步 flow 采样。最小模型文件共 **26,244,309,246 bytes（26.24GB / 24.44GiB）**，此外还需镜像、缓存、仿真资产与结果空间。文件大小不等于峰值显存；本次实测显存见[验证报告](validation.md)，尚未验证消费卡部署。

```bash
git clone https://github.com/fuheaven/LightX2V-microduck.git
cd LightX2V-microduck
docker build -f deploy/microduck/fastwam/Dockerfile -t lightx2v-microduck:fastwam-v1 .
export FASTWAM_DATA=/absolute/path/to/microduck-data
export GPU_ID=0
```

如服务器访问 PyPI 很慢，可在构建命令加 `--build-arg PYPI_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple`，版本约束保持不变。本次 H200 构建使用此镜像源。

`run.sh` 只暴露指定卡，按调用用户 UID/GID 运行，源码只读挂载 `/workspace`，数据挂载 `/data`，不修改宿主 Python 环境，不开放端口。请选择自己有权使用的空闲卡。CPU 命令使用 `GPU_ID=none`；CPU 渲染还需 `MUJOCO_GL=osmesa`。

固定基础镜像为 PyTorch 2.7.1 / CUDA 12.8，含 digest。直接依赖见 `requirements.lock`，完整安装列表在镜像 `/opt/microduck/installed-packages.txt`。FastWAM 使用 Python 3.11 / Transformers 4.49.0，通过 `PYTHONPATH` 加载共享仿真包，**不要安装主仓库全部 requirements，也不要混入 TinyVLA 的 Transformers 5 环境**。BAM 上游声明 Python 3.12；此镜像只加载仓库中固定的 CPU actuator 源码并检查其 3.11 推理通路，不安装 BAM 训练依赖。

原生默认路径使用 PyTorch 注意力/归一化；导入时可能提示未安装可选 FlashAttention/SageAttention。这些提示不等于加载失败，本交接未要求安装这些扩展。

## 2. 工件

```bash
GPU_ID=none deploy/microduck/fastwam/run.sh \
  python scripts/microduck/fastwam_handoff.py fetch
```

下载固定版本的官方本体、ONNX 策略，以及 Wan 配置、UMT5、tokenizer、VAE。原生 FastWAM 从已训练 checkpoint 加载视频/动作专家，**不需要 Wan 原始 diffusion 分片、完整训练数据、优化器/DeepSpeed 状态或 TinyVLA 权重**。

向项目负责人取得下列三个文件，保持原始字节。当前没有公开权重下载地址，不能用公开 LIBERO 权重替代。

```text
$FASTWAM_DATA/
  models/
    microduck/
      model.pt                    # 原 step_000250.pt，12.04GB
      dataset_stats.json          # 该模型训练集归一化
      navigation_checkpoint.json # 权重、统计与历史输入契约的绑定
    Wan2.2-TI2V-5B/
      config.json
      Wan2.2_VAE.pth
      models_t5_umt5-xxl-enc-bf16.pth
      google/umt5-xxl/{special_tokens_map.json,spiece.model,tokenizer.json,tokenizer_config.json}
  upstream/{microduck_rl,microduck-policies}/
  inputs/
  results/
```

来源、大小、SHA256 见 [weight-inventory.json](weight-inventory.json)。不要重算统计或替换 checkpoint receipt。

```bash
GPU_ID=none deploy/microduck/fastwam/run.sh \
  python scripts/microduck/fastwam_handoff.py verify
```

校验会真实读取文件并计算 SHA256；错误工件直接失败，不静默降级。

## 3. 固定输入测速和 Profiler

```bash
MUJOCO_GL=osmesa GPU_ID=none deploy/microduck/fastwam/run.sh \
  python scripts/microduck/fastwam_handoff.py inputs --output /data/inputs/startup.npz

deploy/microduck/fastwam/run.sh python scripts/microduck/fastwam_handoff.py benchmark \
  --inputs /data/inputs/startup.npz --variant plain --output /data/results/plain

deploy/microduck/fastwam/run.sh python scripts/microduck/fastwam_handoff.py benchmark \
  --inputs /data/inputs/startup.npz --variant optimized --reference /data/results/plain \
  --output /data/results/optimized --profile
```

`inputs` 保存 3 个真实 MuJoCo 相机/状态/指令输入，使用合法的 episode 初始帧历史填充。它用于首次部署检查；新优化还应覆盖运动、遮挡、转头等录制输入，不能仅凭这三个样本宣布全面对齐。

每个输入固定噪声种子，冷调用与预热后至少 20 次稳态测量，CUDA 同步计时。报告包含加载时间、冷调用、逐次耗时、p50/p95、峰值 allocated/reserved 显存、版本与源码/工件哈希。`--reference` 核对输入、模型、精度、采样设置，检查动作 `atol=rtol=1e-2`（BF16 工程门槛，非逐位等价），不通过则非零退出。

报告中的 allocated/reserved 峰值在模型加载后重置计数，属于推理阶段，不包含加载瞬时峰值与驱动上下文。实际显卡容量需求可能更高，不能据此断言 24GB 卡可用。

`--profile` 另跑一次，不污染计时；`trace.json` 可用 Perfetto/Chrome tracing 查看。还可用 `--variant prompt` 和 `context` 分解优化收益。每轮使用新目录，避免覆盖证据。

## 4. MuJoCo 闭环

短集检查模型、官方 ONNX、物理和记录链路：

```bash
MUJOCO_GL=osmesa deploy/microduck/fastwam/run.sh python scripts/microduck/fastwam_handoff.py eval \
  --variant optimized --mode offline --tasks find_red_ball --seeds 2000 \
  --max-ticks 100 --output /data/results/offline-smoke
```

短集超时不能用于判断完整任务能力。完整实时案例：

```bash
deploy/microduck/fastwam/run.sh python scripts/microduck/fastwam_handoff.py eval \
  --variant optimized --mode realtime --scene room_clear \
  --tasks find_red_ball --seeds 2000 --output /data/results/realtime-red-2000
```

换 `--variant plain` 和新目录形成对照。完整开发批次使用 seed 2000–2019 的实际整数列表，并指定 `find_red_ball find_blue_ball find_yellow_ball`。单障碍用 `--scene room_curriculum_one`，仍是实验能力。短 smoke、开发成绩和离线结果均不是正式实时验收。

`report.json` 为批次进度/逐集结果，`resolved_config.json` 为实际模型配置，各 episode 的 `episode.json`、`control.json`、`predictions.json`、`recording.npz` 保存任务、相机、动作与来源 tick。优先查看拒绝计划、回退 tick、观测年龄、实际时间比，再判断模型是否失败。

默认观测时效 500ms。改变 `--max-action-age-ms` 必须记成另一实验条件，不得混入纯加速对照。底层入口 `fastwam_navigation_eval.py --help` 另有取消年龄限制、相机工作进程等诊断选项；有限动作块、episode 和乱序检查仍生效。

### 渲染问题

EGL 需要容器提供 graphics 能力。历史曾遇到 EGL/CUDA 共卡相机严重卡顿；模型单测正常而相机数秒一帧时，应先检查渲染/驱动交互。`MUJOCO_GL=osmesa` 可用于 CPU 渲染诊断，但可能达不到实时要求。不能通过降低相机频率、暂停物理或放宽时限掩盖问题。跨进程/跨主机部署变更必须单独记录。

此交接不重启原训练监控网页，不依赖已有端口或 SSH 隧道。若需浏览图像，可读取 `recording.npz` 的相机帧，或使用[导出视频命令](#5-录像和回归)。

## 5. 录像和回归

```bash
GPU_ID=none deploy/microduck/fastwam/run.sh python scripts/microduck/fastwam_recording_video.py \
  --recording /data/results/realtime-red-2000/EPISODE/recording.npz \
  --output /data/results/realtime-red-2000/head-camera.mp4

GPU_ID=none deploy/microduck/fastwam/run.sh python -m pytest -q \
  tests/test_fastwam_prompt_cache.py tests/test_fastwam_action_graph.py \
  tests/test_fastwam_action_context_cache.py tests/test_fastwam_handoff.py \
  embodied/microduck/tests -m 'not integration' -o cache_dir=/data/cache/pytest
```

`EPISODE` 替换为报告中实际 episode 目录。视频展示记录相机，非实时流，也不是第三人称重渲染；成绩只读评测报告。CPU 契约测试不替代真实 GPU 前向与闭环验证。
