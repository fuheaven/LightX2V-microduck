# 来源与许可证

| 组成 | 来源和固定点 | 处理方式 |
|---|---|---|
| LightX2V | ModelTC/LightX2V，完整 commit 见 source-provenance.json | 保留 Apache-2.0、上游源码与文件声明 |
| FastWAM 参考路线 | https://github.com/yuantianyuan01/FastWAM；参考源码快照在历史评测中单独记录 | MIT，保留 [FastWAM 作者许可](licenses/FastWAM-LICENSE)；本部署使用 LightX2V 原生实现，不要求额外安装作者训练仓库；不含其原始模型权重 |
| MicroDuck runtime 契约 | pollen-robotics/microduck `5620aa214e6e304eec00d5c376ac81d3001fb7a3` | 关节/观测/动作契约按来源注明，代码 Apache-2.0 |
| MicroDuck RL 本体 | pollen-robotics/microduck_rl `2b581c641406a48346e696212930ea881c222c52` | 代码与机器人 3D 资产许可不同；资产上游标为 CC BY-NC-SA，不包含在 Git 分发中；下载后保留原条款 |
| 官方 ONNX 策略 | pollen-robotics/microduck-policies `088524a64e2557dc453256b6071dbb9d23888802` | 按 upstream.lock.json 下载并校验，保留上游模型说明 |
| BAM | Rhoban/bam `62bd8ce12154340be97e06f7f41a0ca8f116d967` | Apache-2.0，保留固定 CPU Python 源码（third_party/bam），获取时校验 archive SHA256，构建时再校验逐文件哈希；保留声明/README/元数据 |
| Wan2.2 TI2V 5B | Wan-AI/Wan2.2-TI2V-5B `921dbaf3f1674a56f47e83fb80a34bac8a8f203e` | 公共骨干按固定版本获取，模型卡标注 Apache-2.0 |
| 本项目训练的导航 checkpoint | clear-room-realtime-control-pilot-v1 / step 250；[lightx2v/microduck](https://huggingface.co/lightx2v/microduck) | 由负责人指定仓库发布；尚未单独指定权重许可证，不因代码仓库 LICENSE 自动扩展权重许可；归一化与 receipt 保持原字节 |

第三方代码和资产仍适用其各自条款。共享 MicroDuck 包中保留的 TinyVLA 参考代码有独立 NOTICE；本部署不加载 TinyVLA。此文档没有将机器人非商业资产改为 Apache 许可。

`examples/microduck/visualization/` 新增本项目仿真实验的录屏、缩略图和报告，包含官方 MicroDuck 本体的渲染画面。相关官方资产来源与 CC BY-NC-SA 条款见上表；本仓库代码许可不改变这些素材的来源与适用限制。原始 MJCF/mesh 和 ONNX 仍按固定上游单独获取。

可追溯锁：`embodied/microduck/src/lightx2v_microduck/resources/upstream.lock.json`、`deploy/microduck/fastwam/model-files.lock.json`、`third_party/bam/source-manifest.json`。官方资产由固定源获取，不通过 Git 复制未知或未授权的资源包。
