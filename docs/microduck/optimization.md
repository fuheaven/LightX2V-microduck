# FastWAM 优化交接

## 输入、动作与冻结条件

模型 SHA256 `8ea7793290f0a22bff56ec4e44168232564ec5d57a9e21515a8a8c4a989114ea`，BF16、20 步。先做实现优化；不混入重训、减少采样步数、更换模型或修改任务时限的收益。

输入为 4 张 256×256 头部 RGB 的 512×512 网格：来源 tick 减 `[300,150,50,0]`，即 `[6秒前,3秒前,1秒前,当前]`。从左上到右下按旧到新排列，初始缺失历史重复 episode 第一帧。48D 自身状态来自 obs61 的前 48 维，不含教师命令或世界坐标。ToF 由环境提供，**这份权重没有直接消费 ToF**。

输出 `32×5`：`vx[m/s], vy[m/s], yaw_rate[rad/s], head_yaw[rad], stand_probability`。最后一维 ≥0.5 选择站立，每行对应 5 个 50Hz tick，完整预测范围 3.2 秒。实际执行受 500ms 时效限制；晚到计划按来源 tick 选择行，不从第零行重放。

官方 ONNX 输入是 `gyro3 + gravity3 + joint_offset14 + joint_velocity14 + previous_raw_action14 + command13`。FastWAM 控制 command 的部分通道，其他姿态采用本任务默认值；感知和 raw action 历史由环境实时生成。官方策略输出 14D 关节残差，HOME/缩放/滤波/BAM 按既有契约各处理一次。**50Hz 是底层运动控制频率，不是 WAM 的推理更新率。**

## 代码入口

| 工作 | 位置 |
|---|---|
| 完整模型调用 | `lightx2v/models/runners/wan/fastwam_runner.py: FastWAMPolicy.predict_action_chunk` |
| 预处理、VAE、T5、归一化 | 同文件 `build_image_tensor / encode_image_latents / encode_prompt / LinearNormalizer` |
| Flow 循环 | 同文件 `_run_action_denoising` |
| 视频条件预填充、动作专家 | `lightx2v/models/networks/wan/fastwam_model.py` |
| 注意力与条件 K/V 复用 | `lightx2v/models/networks/wan/infer/fastwam/{pre_infer,transformer_infer}.py` |
| CUDA Graph | `lightx2v/models/networks/wan/infer/fastwam/action_graph.py` |
| 历史输入 | `embodied/microduck/src/lightx2v_microduck/{navigation_history,wam_history_policy}.py` |
| 配置和动作约束 | 同包 `wam_policy.py / navigation_action_contracts.py` |
| 来源时钟、动作缓冲 | 同包 `navigation.py / navigation_requests.py` |
| 连续物理与评测 | 同包 `navigation_eval.py / env.py / physics.py` |
| 测速入口 | `scripts/microduck/fastwam_handoff.py benchmark` |

直接用 Python API 时先设置 `LIGHTX2V_IMPORT_MODE=robotics`、`DTYPE=BF16`，按交接脚本调用 `native_navigation_config(..., history_spec=history_spec((300,150,50,0)))`。不要用单帧默认配置加载历史模型。真实闭环使用历史策略适配器；benchmark 的 core 调用接收已经拼好的历史图。

## 优化开关

| 变体 | prompt cache | action context cache | action CUDA Graph |
|---|---|---|---|
| `plain` | 关 | 关 | 关 |
| `prompt` | 开 | 关 | 关 |
| `context` | 开 | 开 | 关 |
| `optimized` | 开 | 开 | 开 |

- 提示词缓存复用相同指令的 UMT5 编码；换指令或 reset 时失效，不跳过新图像。
- 动作条件缓存在一次 plan 内复用不随 flow 步骤变化的条件投影，新观测仍重新计算。
- CUDA Graph 捕获固定形状的动作去噪循环，减少 Python/kernel launch 开销；每次更新图像、文本、状态与噪声输入。

上述优化实现在 LightX2V 原生 FastWAM 中，使用 PyTorch/CUDA。相机处理、异步物理、IPC 和动作时效属于 MicroDuck 执行系统，应分别计量。默认没有量化、蒸馏、减少采样步数或原生 `torch.compile`；实验性 RMS fusion 保持关闭。

## 本次交接实测

独立构建的镜像、单 H200、相同 3 个真实相机/状态/语言输入，每输入 20 次稳态推理：plain p95 **505.39ms**，optimized **179.15ms**，约 **2.82 倍**。在这组输入和固定噪声上，动作最大绝对差异 **0.0**。这不是完整任务成功率，也不替代变化中图像和多 episode 回归。

推理阶段 PyTorch allocated 峰值为 23.48 / 23.69GiB；它不包含所有驱动或加载开销，未验证 24GB 卡。Profiler trace 已实际生成；原始 benchmark JSON 见 evidence 目录。

## 历史闭环结果和下一步

历史相同权重、60 个开发案例、500ms 时效：普通版 0/60，三项优化开启 59/60。按导出的 assessment 逐集 rows 重算，**每集推理 p95 的中位数约 569.4ms → 174.9ms**，约 3.26 倍，降低 69.3%。这是合并效果，不能将全部收益归给某一个开关；也不是纯 kernel 计时。

后续跨主机取消年龄拒绝的普通版复测为 17/60，说明模型并非完全不会执行任务。TTL、部署、服务器负载变化需分开解释。导出的当前执行器含后续修正，历史分数不能直接沿用。

建议优化交付顺序：

1. 同卡跑 plain/optimized，保存冷启动、稳态、显存、逐输入动作与数值对比。
2. 用 Profiler 定位 T5、VAE、条件预填充、动作去噪、同步的占比；Profiler 调用不计入测速。
3. 每项优化单独开关。缓存特别检查连续图像变化、语言切换、reset、跨 episode、采样噪声改变。
4. 固定开发案例做实时 A/B：成功率、回退、过期率、观测年龄 p95、相机延迟、实际时间比。吞吐提升不自动等于闭环改善。
5. 分别报告 CPU 检查、GPU 前向、固定输入对齐、短集运行、完整实时任务、真机验证。

最小部署不携带原始 Wan diffusion 分片或训练状态；这不等于网络压缩或显存下降。量化、卸载、换精度等应另做质量回归。
