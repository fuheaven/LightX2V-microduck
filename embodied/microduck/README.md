# LightX2V × MicroDuck

这是可独立安装、不依赖 ROS2 的 MicroDuck 仿真、数据与评测包，配合仓库内的原生 TinyVLA 后端。当前已经实现真实 MuJoCo+BAM、官方策略、数据采集与分支 DAgger、训练、评测、Web 和 `robotd` 外部动作会话；**尚未训练出达到计划成功率的自主鸭子模型**。

当前状态和证据见 [实施与验收记录](../../docs/microduck/IMPLEMENTATION.md)。完整公开 TinyVLA 权重已与作者原始实现做数值对齐，但公开版本缺少原始归一化与支持示范；不能直接作为作者原样闭环模型运行。这里自行训练的模型使用新统计量，单独标记版本。

## 安装与固定资产

从 LightX2V 仓库根目录执行。推荐 Linux x86_64、Python 3.12；独立仿真环境与官方 GPU RL 环境分别安装。所有大资产保存在外部缓存，不随代码重新分发。

```bash
bash embodied/microduck/scripts/bootstrap.sh /absolute/venv sim
source /absolute/venv/bin/activate
export MICRODUCK_ASSET_ROOT=/absolute/cache/microduck/upstream
export MICRODUCK_RUN_ROOT=/absolute/runs/microduck
export MUJOCO_GL=egl

microduck assets fetch
microduck assets verify
microduck sim --profile official_bam --scene flat --task walk_forward \
  --policy official --steps 250 --output "$MICRODUCK_RUN_ROOT/standalone"
```

默认获取官方 RL 本体与九个 ONNX 策略。资源锁在 `src/lightx2v_microduck/resources/upstream.lock.json`，记录完整 revision、下载摘要、文件清单摘要和许可证来源。资产不完整或摘要不一致会明确报错。

```bash
microduck assets list
microduck assets fetch microduck microduck-ai-world tinyvla tinyvla-author-rl
microduck assets fetch official_simulator-reference tinyvla_simulator-reference
microduck assets fetch community:follow-v4 community:rough-walk-g community:basketball
microduck assets verify microduck microduck-ai-world tinyvla tinyvla-author-rl
```

跟随策略还需要 `pip install -e "embodied/microduck[community]"`。社区标识以 `src/lightx2v_microduck/resources/community-policies.json` 为准。`--all` 会获取全部锁定来源，包括大权重。许可证不明确的生态条目仅登记；官方代码许可与机器人模型资产条款分别保留。

安装原生 TinyVLA 与训练环境，把 bootstrap 的 `sim` 换成 `cpu` 或 `cuda`。`cuda` 在目标机器解析 CUDA wheels 并保存完整安装快照；这里的 CPU 验证不代替 H100/H200 验证。官方批量 RL 使用单独环境：

```bash
bash embodied/microduck/scripts/bootstrap_rl.sh /absolute/official-rl-venv
```

该脚本使用锁定上游的 `uv.lock`，不将其 mjlab/Torch 依赖混入独立推理环境。GPU 仿真与 CPU BAM 的统计一致性仍需在 GPU 节点验证。

## 场景与配置

| 名称 | 作用 |
|---|---|
| `official_bam` | 官方 RL 本体、HOME、61D、CPU BAM 和原始动作后处理；默认动力学基准 |
| `tinyvla_author_compat` | 作者固定旧版 RL 本体及相机/初始化差异；不宣称等同私有 Stage 2.5 |
| `robotd_bam` | 真正的 Rust `robotd` 加本项目 BAM body server，物理持续运行 |
| 上游 `official_daemon` | 未修改的官方 duck-body/robotd，对照其 position-actuator 路径；与 BAM 分开 |

`flat` 用于动作程序；`room` 为固定房间；`room_clear` 保持相同球位而移除静态箱子，用于障碍消融；`room_randomized` 生成可追溯布局，含视野外目标与遮挡候选；`water_court` 导入社区世界并替换为固定官方本体；`ball_apex` 和 `community_follow` 是专用社区任务场景。

Water Court 保留场景与相机选择、支持场景内多本体绑定；不宣称流体物理或已实现多鸭社交。场景真值只供教师与评测使用。`deployment_noise_v1` 是有界相机、IMU、编码器、BAM 摩擦及动作延迟扰动，不是全部官方 GPU 动力学随机化；纹理当前未形成独立测试域。

```bash
microduck sim --scene water_court --task find_red_ball --policy pixel \
  --output "$MICRODUCK_RUN_ROOT/water-court"
microduck sim --scene room_randomized --randomization deployment_noise_v1 \
  --task find_red_ball --policy privileged --output "$MICRODUCK_RUN_ROOT/noise-teacher"
```

`official` 任务控制器可能用真值恢复；`privileged` 使用规划真值；它们是教师上限，结果不会标为像素自主。`pixel` 从 RGB 颜色检测与 ToF 产生搜索/接近/停止意图，执行官方低层 ONNX；仍须以实际测试集判定能力。

`pixel_rgb` 关闭 ToF，作为与当前 TinyVLA 相同 RGB/本体感知输入的消融。`pixel` 比 TinyVLA 多使用 ToF，报告必须保留这个传感器差异，不能将两者直接宣传为同观测条件比较。

## 采集、训练与重放

先验证教师，再扩大数据规模。13 个作者命令程序已固定，腿式 10 项与滚轮 3 项分开建模。下面是计划配额，**不是已完成的数据量**：

```bash
microduck collect --suite skills13 --teacher official --episodes 200 \
  --output "$MICRODUCK_RUN_ROOT/data/skills"
microduck collect --suite visual_goal --teacher privileged --episodes 334 \
  --scene room_randomized --steps 1500 --output "$MICRODUCK_RUN_ROOT/data/goals"
microduck replay --dataset-index "$MICRODUCK_RUN_ROOT/data/goals/index.json" \
  --scene room_randomized --episode 0
```

`episodes` 是每个任务的条数，334×3=1002。`skills13` 自动写入 `legs` 和 `rollers` 子目录。RGB、语言、机器人状态、原始标签和实际执行目标保存在学生数据；教师特权状态及 command13 在独立档案。视觉任务学生 command13 始终中性。记录包含布局、相机、tick、模型及执行配置摘要。

DAgger 以学生真正访问的 MuJoCo 状态克隆分支，教师在分支滚动出动作块，主环境由学生动作推进：

```bash
microduck collect --suite visual_goal --teacher privileged --mode dagger \
  --policy lightx2v_tinyvla --model-path /absolute/trained-checkpoint \
  --backbone-path /absolute/qwen-config-tokenizer --device cuda \
  --scene room_randomized --episodes 20 --output "$MICRODUCK_RUN_ROOT/data/dagger-1"
```

训练前准备固定 Qwen config/tokenizer/processor（无需重复下载其模型权重），方法见 [训练说明](recipes/README.md) 与 [TinyVLA 文档](../../scripts/tinyvla/README.md)。设置 `MICRODUCK_BACKBONE_ROOT`。recipe 的数据目录须对应实际采集目录，使用 `--dataset-index` 可覆盖。

```bash
export MICRODUCK_BACKBONE_ROOT=/absolute/qwen-config-tokenizer
python scripts/tinyvla/fetch_backbone.py --output "$MICRODUCK_BACKBONE_ROOT"
export LIGHTX2V_IMPORT_MODE=robotics
microduck train --recipe tinyvla_microduck \
  --dataset-index "$MICRODUCK_RUN_ROOT/data/skills/legs/index.json"
MICRODUCK_PYTHON=/absolute/venv/bin/python \
  bash embodied/microduck/scripts/train_8gpu.sh tinyvla_legs_bc_8gpu \
  --dataset-index "$MICRODUCK_RUN_ROOT/data/skills/legs/index.json"
```

`tinyvla_microduck` 是腿式 BC 配置；另外提供滚轮 BC、视觉阶段与分支 DAgger 配置。8 卡使用 DDP、BF16、AdamW、全局 batch128；BC 冻结骨干，视觉阶段解冻后两层语言网络。统计量仅从训练集计算，失败教师标签默认不用于训练。checkpoint 包含权重、配置、统计量、执行 manifest 和哈希。

恢复训练支持模型/优化器状态，但不承诺 dataloader/RNG 逐位续跑。没有达标的闭环评测时，模型不能升级为已复现能力。

### FastWAM 视觉导航课程

FastWAM 学习 RGB、语言与机器人状态到高层导航动作，底层仍由官方 ONNX 行走策略执行。
`scripts/microduck/fastwam_navigation_collect.py --scene` 另支持两个实验课程：

| 场景 | 用途 | 障碍物 |
| --- | --- | --- |
| `room_curriculum_one` | 单障碍教师诊断与后续课程数据 | 原随机布局的第一个箱体 |
| `room_curriculum_two` | 两障碍教师诊断与后续课程数据 | 原随机布局的前两个箱体 |
| `room_randomized` | 完整随机障碍任务 | 原有四个箱体 |

课程先调用完整布局生成器，再选择箱体前缀。同一个 seed 的球位置、颜色分配和机器人保持一致，
各课程使用不同场景名、几何摘要与布局标识。seed 使用完整整数，不按奇偶折叠。
采集报告标记 `curriculum_only=true`，并保存场景源码；不会把课程成绩合并成完整障碍任务成绩。

课程入口已通过官方资产上的契约检查，但教师能力和数据尚需单独验收。
先完成独立教师资格检查，再发布对应阶段的数据、统计与模型；课程成功不替代最终
`room_randomized` 保留布局中的搜索、绕障、接近和稳定停止验收。课程目前通过 Python 环境与采集脚本使用，
尚未接入 Web 场景选择。

## 原生服务、评测与比较

原生后端位于 `lightx2v/models/networks/tinyvla` 与 `lightx2v/models/runners/tinyvla`；不是 HF Space 转发，也不导入 LeRobot 或 ROS。独立 runner 不注册为视频生成模型。作者参考服务在单独依赖环境运行。

```bash
python scripts/tinyvla/serve.py --checkpoint /absolute/trained-checkpoint \
  --backbone-path "$MICRODUCK_BACKBONE_ROOT" --device cuda
microduck eval --suite visual_goal --scene room_randomized \
  --policy lightx2v_tinyvla --endpoint http://127.0.0.1:8091 \
  --seeds 200 --seed-start 10000 --output "$MICRODUCK_RUN_ROOT/eval/visual"
microduck eval --suite skills_legs10 --scene flat --policy lightx2v_tinyvla \
  --model-path /absolute/legs-checkpoint --backbone-path "$MICRODUCK_BACKBONE_ROOT" \
  --device cuda --seeds 100 --output "$MICRODUCK_RUN_ROOT/eval/legs"
microduck compare --experiment embodied/microduck/experiments/baselines.yaml
microduck web --runs "$MICRODUCK_RUN_ROOT" --port 8767
```

HTTP 默认端口以 `serve.py --help` 为准。`200` 是每个颜色的 seed 数量，总计600个任务回合；布局去重后仍需至少200个未见几何布局，不把同一布局的三种颜色计为三个场景。比较文件固定场景、seed、传感器和任务；不同 embodiment/checkpoint 分开运行。`vla-ablation.template.yaml` 是需填模型路径的实验模板。

Web 首页 `/` 是能力展厅：先看登记模型的独立验收成绩、成功/失败回放，再体验官方运动与社区实验。原参数配置和分组历史页面保留在 `/experiments`，`/training` 和 `/school` 保持各自入口。2026-09-28 核查的正式离线成绩为 6/13；当前项目用 `ground_lower` 替代原作者的 `ground_touch`，不宣称视觉语言自主规划或实时、真机验收。

实验室先选择“官方策略与预设动作 / 视觉目标任务 / 社区实验”，再展示匹配选项。跟随、球上平衡自动绑定专用场景；Water Court 只代表场景接入，原项目 VLM 大脑未接入。只有 TinyVLA 策略需要配套模型；上层 VLM 技能规划单独配置服务。前后端共用兼容性目录，禁止不匹配组合。首页只启动官方或社区已配置策略，不自动启动训练或 H200 模型服务。

本次新增的固定版本运动对照可独立获取：

```bash
microduck assets fetch community:official-velstand community:duckbatch-128
microduck assets verify community:official-velstand community:duckbatch-128
```

部署与页面验证见 [能力展厅说明](../../docs/microduck/simulator-showcase-2026-09-28.md)；社区调查和本地试验边界见 [社区更新](../../docs/microduck/community-review-2026-09-28.md)。

运行方式默认 `live`：同一 MuJoCo+BAM 环境按墙上时间推进，目标控制频率50Hz；原子发布最新图像，浏览器边运行边显示，可切换视角、停止或刷新重连。显示包含仿真时间、运行速度、超时及接管信息。推理/渲染慢于预算时报告落后，不修改物理步长伪装实时。TinyVLA 在独立线程处理不可变观测，通过200ms时效的动作缓冲器执行；无有效动作时调用本地官方站立/滚轮策略，接管不能计为VLA独立成功。其他本地策略在控制循环中执行并记录超时。浏览器断连不暂停物理，也不会自动重启实验。停止会保存已经执行的部分并标为 `cancelled_by_user`；加载阶段强制停止可能尚无轨迹。这个实时查看模式不代替真正 `robotd_bam` 的协议与健康门槛验证；网页中 `robotd_bam` 仍仅开放给 TinyVLA 视觉目标任务。

```bash
microduck live --scene flat --task walk_forward --policy official \
  --seed 2 --output "$MICRODUCK_RUN_ROOT/live-walk"
```

`offline` 继续提供尽快完成计算的离线评测。历史区有独立类别标签和任务、场景、策略、结果筛选，记录按任务分组，每页最多8条，支持只显示有画面的记录。每条记录可放入左/右比较，显示实际场景、时间与标识，筛选不重置已选的对比对象。新实验结束后自动定位其记录。时间轴按所选记录的实际时长停止，不再固定循环30秒。分数仍读取同一评测 JSON，可查看并排回放和失败详情。可切换第三人称与机器人相机；第三人称只用于审阅，不进入策略或训练数据。`--no-observer` 可关闭离线第三人称录制；无相机记录明确显示无图像。网页不计算另一套分数。服务器默认只监听本机，可通过 SSH 转发端口使用。

每集保存配置、策略 manifest、观测/动作 NPZ、执行轨迹、终止原因和干预计数。正式成功不含跌倒、碰撞、超时或官方控制接管。真值教师、几何跟随与自动恢复分别标识；未完成回合不能计成功。

## 鸭子学校：任务参数学习

Web 首页的“进入鸭子学校”打开 `/school`，与现有 Simulator 共用服务地址。
选择课程示例 → 预览自然语言目标 → 开始教学 → 查看测试结果 → 从技能库实时运行、独立测试或续训。
默认教学预算 10 分钟，课程记录按类别筛选、每页 6 条；刷新浏览器不会重启训练。

五类 `goto/explore/spin/circle/ball` 课程使用仿真真值；`rgb` 课程使用实际头部 RGB 和本体状态。
首版解析器接受中英注册语法，不调用 LLM，也不执行生成的奖励代码。RGB 感知使用固定颜色检测、相机标定和 3.5cm 球半径先验。
训练使用 CEM 优化上层 3×5 命令系数；官方 `walk/stand` ONNX 和 BAM 保持固定，不训练视觉网络或 TinyVLA。

```bash
microduck school preview '走到前方 0.8 米的标记并停下'
microduck school train --instruction '找到红色球，靠近并停下' \
  --budget-s 600 --seed 0 --output /absolute/school/red-v1
microduck school evaluate --artifact /absolute/school/red-v1/skill.json \
  --seed 0 --count 10 --output /absolute/school/red-test
microduck school live --artifact /absolute/school/red-v1/skill.json \
  --seed 2 --output /absolute/school/red-live
microduck school train --instruction '找到红色球，靠近并停下' \
  --seed 0 --resume /absolute/school/red-v1/skill.json \
  --output /absolute/school/red-v2
```

Web 工件保存到 `--runs/_school/<id>/`。`skill.json` 保存参数、CEM 分布/RNG、版本及官方工件指纹；`trials/` 保存候选真实运行结果；`evaluation.json` 保存固定规则与训练控制器的同 seed 对照。`evaluation/` 包含独立测试的状态/动作轨迹，RGB 课另含实际图像。实时模式产生 `live.json`，使用独立物理进程按 50Hz 推进、10Hz 发布最新画面。

训练、验证、测试使用不重叠 seed 区间，且真正改变机器人起点、朝向、目标布局。测试结果不用于 CEM 选优。
取消和预算耗尽保存已完成的代；续训创建新版本，不覆盖父工件。不完整的代在续训时重新评估。工件 SHA、控制器源码、官方策略或本体不匹配时明确拒绝。
未完成测试不显示通过；“通过小规模测试”要求至少 10 集且成功率 ≥80%，同时报告规则基线。没有提升时可能仍选中规则初始参数，界面明确标注。
首版场景是平地和球/标记，不包含避障、遮挡课程；所有工件仍无真机验证。详细方案见 `docs/microduck/school-plan.md`。

## robotd 与 ROS2

```bash
microduck assets fetch microduck
microduck robotd build
microduck realtime --backend robotd_bam --scene room --task find_red_ball \
  --policy lightx2v_tinyvla --model-path /absolute/goal-checkpoint \
  --backbone-path "$MICRODUCK_BACKBONE_ROOT" --device cuda \
  --seconds 30 --output "$MICRODUCK_RUN_ROOT/realtime"
```

需要 Rust toolchain。构建在单独副本上应用可审计 external-policy 扩展，不修改缓存中的原始 runtime。开启本地 Unix 会话后，`robotd` 在 50Hz tick 接收/消费 raw14 缓冲，HOME 只加一次，raw-action history 在缩放前记录。非法维度、旧 episode、旧 tick、过期或未来动作会拒绝；默认有效年龄200ms，耗尽时切回官方策略并记录接管。

原始 `official_daemon` 对照直接启动未修改的 `body_server.py` 与 `robotd --sim`，单独构建原始 Rust binary：

```bash
python -m lightx2v_microduck.upstream_daemon smoke \
  --asset-root "$MICRODUCK_ASSET_ROOT" --output "$MICRODUCK_RUN_ROOT/official-daemon" \
  --seconds 2 --build
```

该命令检查真实15关节 RemoteIo、8×8 ToF、物理推进和 daemon health；并非完整任务或外部VLA验收。它运行上游 position servos（kp/200增益），不运行 BAM 电机模型，报告分别标记动力学。

模型推理、HTTP 等待与物理不在同一个阻塞循环。`--latency-ms`、`--drop-every`、`--reorder-every` 注入执行扰动。`robotd_bam` 当前仅支持无离线随机化的官方 BAM 配置。软件 EGL 的相机延迟也会触发真实时效检查；不能通过暂停世界掩盖算力不足。

视觉模型的 command13 中性输入与 daemon 一致。作者预设 BC 数据带任务时钟 command13，当前 daemon 观测尾部为本机零命令；因此不能把离线 BC 的技能结果直接外推为实时技能复现。需另行统一任务输入或训练中性命令版本。

可选 ROS 包在 `lightx2v_ros/src/microduck_interfaces` 与 `microduck_bridge`。在 ROS Jazzy 环境中构建这两个包，启动 `microduck.launch.py`；提供原子图像/61D、动作块、执行反馈、取消和可选 rosbag。桥接 Python 环境必须能导入独立包及仿真依赖，远程推理节点无需 Torch。具体启动参数见 launch 文件。已验证 colcon 构建和消息序列化，尚未验收 ROS+训练模型完整闭环。

## 验证与证据

```bash
LIGHTX2V_IMPORT_MODE=robotics PYTHONPATH=embodied/microduck/src:. \
  python -m pytest -c embodied/microduck/pyproject.toml embodied/microduck/tests tests/tinyvla -q
python -m lightx2v_microduck.validation --asset-root "$MICRODUCK_ASSET_ROOT" \
  --output "$MICRODUCK_RUN_ROOT/rust-contract"
python -m lightx2v_microduck.acceptance --runs "$MICRODUCK_RUN_ROOT" \
  --output "$MICRODUCK_RUN_ROOT/acceptance.json"
```

资产依赖用例需预先下载对应固定工件。全权重参考对齐和性能命令见 TinyVLA 文档。CPU 单步训练、数值对齐、协议健康与硬件任务成功是不同证据；所有当前机器人结果保持 `hardware_unvalidated`。

正式 `success_rate` 只汇总主赛道合资格回合；纯教师或额外ToF组为 `null`，辅助策略实际结果保存在 `system_success_rate`。acceptance 模块还检查样本量、模型身份、完整回合、训练/验证布局排除和轨迹摘要；未见布局证据格式见其模块文档。`--require-pass` 可用于行为验收CI，缺证据明确不通过。
