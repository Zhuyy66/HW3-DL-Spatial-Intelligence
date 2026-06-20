# HW3 期末作业 架构规划与执行方案

> **基础信息**
> 课程：深度学习与空间智能 | 作业：HW3 | 队伍规模：2 人 | 开工日：2026-05-28（周四）| 截止：2026-06-24 23:59（周三）
> 工作目录：`/root/Test/Zhr/DL/HW3` | 可用磁盘：567 GB | 实验室 8×4090 共享
> 关键 GPU 实时观测（2026-05-27 14:13）：**3 号卡 49 GB 几近全空、5 号卡 49 GB 约 30 GB 空闲**，其余 24 GB 卡均仅剩 3–5 GB 余量

---

## 一、总体策略与分工建议

经反复权衡可交付性与时间窗口，本次作业最稳妥的执行策略不是"先做完题目一再做题目二"，而是**双轨并行、两人对口主线、关键里程碑互锁**。理由有三：题目二（ACT + CALVIN）属于 LeRobot 官方文档明确支持的轻量管线，单卡 5 GB 残余显存即可启动，可作为"必交付的稳态主线"持续推进；题目一（3DGS + threestudio + Zero123 + 融合）依赖大显存窗口和多次 prompt / seed 重试，是"上限加分项"但不可作为唯一依靠；而两题完全解耦（不同的 conda 环境、不同的数据流、不同的评测体系），天然适合两人各管一题、交叉评审。

**两人分工建议**如下：成员 A 担任题目一负责人，主导 COLMAP 拍摄、3DGS 训练、threestudio/Zero123 资产生成与 Blender 融合渲染；成员 B 担任题目二负责人，主导 LeRobot 环境搭建、CALVIN 数据子集切分、ACT 训练与跨环境评测脚本编写。报告写作两人共同完成，但各自主笔自己负责的章节；GitHub 仓库与权重打包由一人统一收口（建议成员 A，因题目一产物形态更复杂）。每天晚上 30 分钟同步进度，使用共享文档（推荐飞书或腾讯文档）记录每日 blocker。

由于课程文档没有提供测试/推理脚本，且 2 人组队即可拿到"加分项"，整体策略上不再追求两题都达到 SOTA，而是确保两题**都有完整、可复现、有曲线可看、有视频/数值结果可展示**的成体系交付。题目一的融合渲染策略首选 **Mesh 导出 + Blender 合成**（文档明确认可的路径），统一 Gaussian 表示作为 stretch goal；题目二的评测策略首选 **包装 CALVIN 官方 `evaluate_policy.py` 的 CustomModel wrapper**，因为这是仓库唯一明确文档化的接口。

---

## 二、技术架构规划

### 2.1 题目一架构：多源 3D 资产生成与场景融合

题目一的核心架构思想是**"显式表示统一为渲染坐标系，而非数据结构"**。即不强求把 3DGS 高斯球、SDS 隐式场和单图重建 Mesh 全部转成同一种数据结构，而是统一到同一个世界坐标系与同一条相机轨迹下分 pass 渲染，再做视频合成。这条路线在作业 PDF 第 4 点已经被明确认可（"导出为带纹理的 Mesh 后在 Blender 中结合"），可解释性强，工程风险低。

**物体 A（真实多视角重建）的链路**为：手机环绕拍摄 → ffmpeg 抽帧并过滤模糊帧 → COLMAP SfM 恢复稀疏点与位姿 → 3DGS `convert.py` 转换数据格式 → 3DGS `train.py` 训练高斯场。训练时建议按官方加速版 rasterizer 配合 `--optimizer_type sparse_adam` 启动（约 2.7× 提速），同时打开 `--antialiasing` 以提升最终漫游视频的边缘质量。如果手机拍摄过程中曝光不一致，开启曝光补偿相关参数；如果出现 floater 或道路平面发虚，再叠加深度正则。训练分两阶段：第一阶段 7k iteration 快速验证管线与相机分布；第二阶段 30k iteration 在 3 号或 5 号大卡上做最终质量训练。

**物体 B（文本到 3D）的链路**为：在 threestudio 中以保底配置（DreamFusion + Stable Diffusion 1.5，约 6 GB 显存）启动，验证通路 → 选定 prompt 后以冲高配置（Magic3D coarse-to-fine 或 ProlificDreamer，需 15–30 GB 显存）跑出最终资产 → 用 `--export` 导出带纹理 obj+mtl。Prompt 设计的关键不是"复杂华丽"，而是"主体明确、风格统一、不易触发 Janus"。建议选择"单一主体、外形对称性不过强"的物体，例如"a vintage wooden treasure chest with brass fittings"或"a stylized ceramic owl figurine, smooth glaze"。如果出现多面脸或漂浮几何，按 threestudio README 推荐顺序处理：先换 seed，其次开启 prompt debiasing 或 Perp-Neg，最后才考虑切换到 D-SDS。

**物体 C（单图到 3D）的链路**为：手机拍摄一张正面照 → 用 `rembg` 或 SAM 去除背景得到 RGBA 前景 → 在 threestudio 中以 `configs/stable-zero123.yaml` 启动 SDS → `--export` 导出 Mesh。这里必须使用 Stable Zero123 而非原始 Zero123 或 Zero123-XL，因为 Stability AI 模型卡明确说明 Stable Zero123 在数据渲染与条件策略上更优。如果拍摄物体不是严格正面视角，需要在配置中显式设置 `elevation` 与 `azimuth`。

**背景场景**建议从 Mip-NeRF 360 中选择 `counter` 作为首选——这是一个室内桌面场景，与上述三个物体的尺度最匹配，相机轨迹设计也最容易把三个物体"放在桌面上"形成自然空间叙事。`garden` 作为备选，仅在三个物体都偏户外尺度时考虑。

**融合渲染**采用以下流程：在 Blender 中导入背景场景的 COLMAP 相机轨迹（通过插件或 `cameras.json` 解析），将 B、C 两个 Mesh 在同一世界坐标系下放置、缩放、对齐；对于物体 A（高斯表示），如果时间充裕可以尝试通过 SuGaR 或类似工具转 Mesh 后一起进 Blender，否则采用"双 pass 合成"——背景 + B/C 在 Blender 渲染前景 RGBA pass，物体 A 通过 3DGS 渲染同一相机轨迹下的背景 pass，最终在 ffmpeg 或 DaVinci 中做 alpha 叠加。这条双 pass 路线虽然不优雅，但是当时间紧张时的可靠 fallback。

### 2.2 题目二架构：ACT 跨环境泛化实验

题目二的核心架构思想是**"固定一切、只换数据范围"**。作业明确要求 A-only 模型与 ABC 联合模型使用相同的网络结构和超参数，因此整个实验设计必须以"配置文件版本控制"为骨架——同一份 ACT 配置，仅替换 `dataset.episodes` 字段或数据加载器的环境过滤逻辑。这样做的副产品是：所有实验对比都自动具备可比性，报告里画 Loss 曲线、汇总指标表都不会因为"忘了改某个参数"而失效。

**数据准备链路**为：从 ModelScope 镜像或 HF 镜像下载 `calvin_task_ABC_D` 的 4 个 LeRobot shard → 用 `pandas` 或 `datasets` 库直接读取 parquet 文件中的 `meta` 字段，扫描出环境标签字段（HF 页面未直接展示该字段名，需本地审计）→ 切出 A-only 训练子集、ABC 联合训练子集、D 测试集 → 在本地存为独立的 LeRobotDataset 目录或通过 episode index 过滤实现。考虑到 567 GB 磁盘空间，建议先下载一个 shard 做格式探查，确认环境字段后再决定是否下载全部 shard。如果总数据量过大，可以选择按 episode 数量降采样（例如每个环境取 200–500 个 episode，足以训练 ACT）。

**训练链路**为：在 LeRobot 中以官方 ACT 配置为基础，定义一份 `act_calvin_base.yaml`，固定网络结构（80M 参数、6 层 transformer encoder/decoder、chunk size 100 等）、优化器（AdamW，lr 1e-5）、batch size、epoch 数；A-only 训练用 `dataset.repo_id` 指向 A 子集，ABC 训练指向 ABC 子集，其他不变。建议在 6 号或 7 号卡（约 5 GB 残余）以 batch size 8 启动 baseline，确认稳定后视 GPU 空闲情况尝试单卡 batch size 16 或 multi-GPU。WandB 或 SwanLab 日志记录从第一次训练就启动，避免后期补图表。**注意**：LeRobot multi-GPU 不自动缩放 lr 和总步数，做对比实验时务必固定"有效 batch size × 总 epoch 数"作为对比基线，否则曲线会被"看似 ABC 训练更久所以更好"误导。

**评测链路**是题目二最大的工程不确定项。因为老师没有提供脚本，必须自建。建议采用"包装 CALVIN 官方 `evaluate_policy.py`"的方案：单独建立 `env_hw3_calvin_eval` 环境（Python 3.8 + PyBullet + 原 CALVIN 仓库），实现一个 `LeRobotACTWrapper(CustomModel)` 类，包含 `__init__()`、`reset()`、`step(obs, goal)` 三个方法。其中 `step` 内部完成 CALVIN observation → ACT 输入格式的预处理（图像 resize、proprio 拼接）、调用本地 ACT checkpoint 推理（用 torch 加载即可，不必启动整个 LeRobot 训练框架）、再把 ACT 输出的 action chunk 转回 CALVIN 期望的 action 空间（绝对笛卡尔 vs 关节，需要在数据审计阶段就确认）。这个 wrapper 一旦写好，A-only 与 ABC 两套权重的评测只需替换 checkpoint 路径，大幅降低实验比较成本。

### 2.3 共享基础设施

**Conda 环境拆分**为 4 套，避免互相污染：

```
env_hw3_recon         Python 3.10 + CUDA 12.1 + PyTorch 2.4 + COLMAP + 3DGS
env_hw3_gen3d         Python 3.10 + CUDA 12.1 + PyTorch 2.4 + threestudio + Stable Zero123
env_hw3_robot         Python 3.12 + PyTorch 2.10+ + LeRobot + WandB
env_hw3_calvin_eval   Python 3.8 + PyTorch 1.13 + PyBullet + 原 CALVIN 仓库
```

**磁盘策略**：所有 conda 环境、pip 缓存、HuggingFace cache、模型权重统一指向 `/root/Test/Zhr/DL/HW3`。具体配置如下：在 `~/.bashrc` 或 conda 环境的 `activate.d` 中导出 `HF_HOME=/root/Test/Zhr/DL/HW3/.hf_cache`、`TRANSFORMERS_CACHE=$HF_HOME`、`TORCH_HOME=/root/Test/Zhr/DL/HW3/.torch`、`PIP_CACHE_DIR=/root/Test/Zhr/DL/HW3/.pip_cache`，conda 包路径通过 `conda config --add pkgs_dirs /root/Test/Zhr/DL/HW3/.conda_pkgs` 配置。首次拉齐模型后设置 `TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 DIFFUSERS_OFFLINE=1` 防止训练时回连。

**国内网络加速**：HuggingFace 通过 `HF_ENDPOINT=https://hf-mirror.com` 切到镜像；ModelScope 直接用其原生 API；GitHub 仓库 clone 可通过 `https://gh-proxy.com/` 加速；pip 用清华源 `https://pypi.tuna.tsinghua.edu.cn/simple`；conda 建议先 `conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main`。这些配置在每个环境激活脚本中固化，避免反复设置。

**GPU 占用策略**：基于 5/27 实时快照，3 号卡（49 GB 全空）应在 5/28 第一时间占住用于 3DGS / threestudio 的训练。每个长时任务用 `tmux` 会话和 `nohup` 启动并固定 `CUDA_VISIBLE_DEVICES`，避免被人误踩。ACT 训练放在 6/7 号卡上的"残余显存"区间运行，batch size 设为 8。所有长时任务都在 WandB 或 SwanLab 上挂日志，方便远程查看。

**版本管理**：从 5/28 起每天提交 commit，按 `topic1_fusion/`、`topic2_act/`、`env/`、`reports/`、`assets/` 五个一级目录组织。每周末打 tag（`week1`、`week2`、…），便于回退和最终复盘。

---

## 三、拍摄建议（Task 1 关键产出，建议 5/30 周六完成）

拍摄是题目一最早的不可回退步骤——如果 COLMAP 重建失败，需要重新拍摄；而 COLMAP 失败的根因往往出在拍摄环节本身。建议两位成员在 5/30 周六上午联合完成 A、C 两个物体的拍摄，因为白天光线最稳定、两人配合可以避免拍摄者的影子或反射进入画面。

### 3.1 物体 A（用于多视角重建 + 3DGS）

**物体选择标准**。理想的物体应当具备四个特征：表面有丰富纹理（避免纯色塑料、纯色陶瓷、反光金属、玻璃、透明物体），尺寸适中（约 15–30 cm 高，便于环绕拍摄又不至于太小导致细节糊掉），形状非平面（COLMAP 在纯平面物体上几乎必然失败），轻度反光或哑光表面最佳。推荐候选：旧的复古玩具、带纹理的盆栽（叶片有结构）、外表有图案的茶罐或调味罐、布艺玩偶、有雕刻纹路的木制摆件、带贴标的酒瓶（贴标有文字纹理）。**避雷清单**：纯色马克杯（无纹理）、不锈钢水壶（强反光）、透明水瓶、镜面物体、纯黑物体（细节丢失）、毛绒玩具的极长毛发（COLMAP 容易把毛认作错配）。

**拍摄环境**。光线优先选择阴天的室外或室内大面积漫射光（朝北窗户附近、或开启多盏白光均匀照明的房间），避免强烈直射阳光产生硬阴影。背景选择有纹理的桌面（木纹桌、麻布桌布、有图案的地毯均可），**绝对避免纯色单一背景**（白桌布、黑桌布会让 SfM 失去背景特征点支撑）。物体周围 1.5–2 米内不要有其他移动物体（人、宠物、风扇等）。如果是在桌面拍摄，建议把物体放在桌子中央，留出 360° 环绕空间。

**拍摄方法**。强烈推荐**视频路线**而非纯照片路线，原因是视频帧间相机姿态变化连续、便于 COLMAP 的 sequential matching、且后期可以根据需要灵活抽帧。具体参数：手机切到 1080p @ 30fps（不要用 4K，文件过大且 COLMAP 默认会缩到 1.6K 以下）；**锁定曝光、锁定对焦、锁定白平衡**（iPhone 长按对焦框出现 AE/AF 锁、Android 在专业模式手动设置）；持手机以约 1 米半径**沿圆形轨迹围绕物体走动**，**而非站在原地旋转手机**——这是 COLMAP 文档反复强调的关键点，原地旋转会让 SfM 无法三角化深度。建议拍摄 3 圈：第一圈眼平视（与物体等高）、第二圈俯视约 30°（手举高一些）、第三圈仰视约 20°（蹲下来）。每圈大约 30–45 秒，三圈合计约 2 分钟。

**视频处理**。回到服务器后用 ffmpeg 抽帧：`ffmpeg -i video.mp4 -vf "fps=4" frames/%04d.jpg`，4 fps 抽帧会得到约 500 帧，再手动用脚本根据 Laplacian 方差过滤掉模糊帧（运动模糊或对焦失败的），目标是保留 150–250 张清晰图像送入 COLMAP。**不要把全部 500 帧都送进去**——COLMAP 的 exhaustive matching 在几百张图上会从几小时增长到几天，而 sequential matching 配合 ~200 帧是性价比最高的设置。

### 3.2 物体 C（用于单图到 3D + Stable Zero123）

物体 C 的拍摄逻辑完全不同。这里只需要**一张高质量的正面照**，但要求比物体 A 更严苛，因为 Stable Zero123 对输入图像的"主体清晰度"非常敏感。

**物体选择**。优先选**轮廓鲜明、对称性不过强、有明确"正面"语义**的小物件：一只单独的玩具公仔（不是双胞胎那种过于对称的）、一个手办、一个有把手或喷嘴的茶壶、一只单独的运动鞋、一个有正面 logo 的杯子。避免太对称（容易触发 Janus 多面脸）、避免太复杂（多个并列主体）、避免太薄（书本、纸片——背面没有可推理的几何）。

**拍摄要点**。在均匀漫射光下，把物体放在**纯白或纯灰背景**前（白墙、白纸板均可，与物体 A 完全相反，因为这里要方便后续抠图），相机正对物体的"正面"——眼平视、距离约 0.5–1 米，让物体占画面中央约 60–70% 面积，避免广角变形（建议用主摄而不是超广角镜头）。拍约 5–10 张同角度照片以备选最佳一张。后期处理：用 `rembg` 或 Photoshop 抠掉背景保存为 RGBA PNG（512×512 或 1024×1024 正方形），确保主体居中、Alpha 通道边缘干净。

### 3.3 背景场景的获取

背景场景**不需要自己拍摄**——作业明确要求从 Mip-NeRF 360 数据集中选取。建议从官方 Google Drive 或 HF 镜像下载 `counter` 场景（约 5–8 GB），它已经包含完整的 COLMAP 输出，可以直接喂给 3DGS 的 `train.py`，无需自己跑 SfM。这一项的工作量主要在"训练 3DGS 到收敛"而非数据准备。

---

## 四、四周周度计划（5/28 周四 — 6/24 周三）

| 周次 | 日期范围 | 核心目标 | 题目一里程碑 | 题目二里程碑 | 报告进度 |
|---|---|---|---|---|---|
| **第 1 周** | 5/28 (四) – 6/3 (三) | 环境搭建 + 数据采集 + Smoke Test | 3 套环境装好、物体 A/C 拍摄完成、COLMAP 跑通一次、3DGS 跑出 toy 结果 | LeRobot 环境装好、CALVIN 数据下载与字段审计完成、ACT 跑出第一条 loss 曲线 | GitHub 仓库初始化、README 骨架 |
| **第 2 周** | 6/4 (四) – 6/10 (三) | 核心训练全面铺开 | 物体 A 3DGS 训练到 30k iter、背景场景 counter 训练完成、threestudio 装好并跑通 DreamFusion smoke test | A-only ACT 训练完成（含完整 loss 曲线）、CALVIN 评测 wrapper 写好并能跑通一集 rollout | 任务背景、方法原理章节初稿 |
| **第 3 周** | 6/11 (四) – 6/17 (三) | 生成资产 + 联合训练 + 评测 | 物体 B（文本→3D）完成、物体 C（单图→3D）完成、Mesh 导出全部就绪 | ABC 联合 ACT 训练完成、A-only 与 ABC 在环境 D 上 zero-shot 评测完成、动作分块鲁棒性分析 | 实验结果章节初稿、关键图表导出 |
| **第 4 周** | 6/18 (四) – 6/24 (三) | 融合渲染 + 报告终稿 + 复现验证 | Blender 融合渲染完成、多视角漫游视频成片输出、3 种生成方式对比表 | 评测结果交叉验证、超参表与指标表定稿 | 报告终稿、GitHub README 完整、权重打包上传、复现命令端到端验证 |

### 第 1 周（5/28–6/3）目标说明

第 1 周是整个项目的"工程地基"，质量决定后三周的速度。核心可交付物是：**两套环境可重复构建（提供 `environment.yml`）、物体 A 和 C 拍摄数据已上传服务器、3DGS 在 Mip-NeRF 360 `counter` 上跑出第一张评测图、ACT 在 CALVIN A 子集上跑出第一条 loss 曲线**。如果到 6/3 这四个里程碑没有全部达成，第 2 周必须立即调整任务优先级（具体看每日计划）。

### 第 2 周（6/4–6/10）目标说明

第 2 周转入"核心训练"。题目一方面，物体 A 的 3DGS 训练完成（含 7k 和 30k 两个 checkpoint），背景场景 counter 训练完成，并在 Blender 中验证相机轨迹可以正确导入；threestudio 完成安装并跑通一个 toy DreamFusion 配置（不追求质量，只确认链路通）。题目二方面，A-only 模型训练完成（含 WandB 全程日志），评测 wrapper（`LeRobotACTWrapper`）写好并能在环境 D 上跑通一集完整 rollout（不追求成功率，只确认接口通）。报告启动：任务背景、数据集描述、方法原理这三章可以基本完稿，因为这些内容与最终实验结果无关。

### 第 3 周（6/11–6/17）目标说明

第 3 周是"生成+评测"的高强度周。题目一方面，物体 B 通过 threestudio 的保底配置（DreamFusion + SD 1.5）跑出 mesh，如果时间允许再用 Magic3D 或 ProlificDreamer 跑冲高配置；物体 C 通过 Stable Zero123 跑出 mesh；两者都导出 obj+mtl 并验证 Blender 可导入。题目二方面，ABC 联合训练完成，A-only 和 ABC 两套权重都在环境 D 上完成 zero-shot 评测，整理出成功率、动作 L1 误差等核心指标。报告启动实验结果章节，所有曲线图表、对比表导出到 `reports/figures/`。

### 第 4 周（6/18–6/24）目标说明

第 4 周是"收尾周"，前 4 天专注融合渲染与报告写作，后 3 天专注复现验证与提交准备。题目一的融合渲染（Blender 中三物体放置、相机轨迹设计、双 pass 渲染、ffmpeg 合成）在 6/18–6/20 完成；报告终稿、GitHub README、权重云盘上传在 6/21–6/22 完成；6/23 全天做端到端复现验证（按 README 命令重新跑一遍训练和评测，确认每条命令都能从干净环境复现）；6/24 留作 buffer day 处理突发问题，**严禁把 6/24 视为常规工作日**。

---

## 五、第一周日度计划（5/28 周四 — 6/3 周三）

> 每日计划按"上午、下午、晚上"分时段安排；标注 **A** = 成员 A（题目一），**B** = 成员 B（题目二），**A+B** = 两人协作。
> 每日 22:00 之前完成 git commit 与进度同步。如某天里程碑未达成，次日开工前 30 分钟在共享文档中调整后续计划。

### 5/28（周四）Day 1：项目启动与基础设施

**上午（A+B 协作，约 3 小时）。** 完成项目启动会，包括：在飞书或腾讯文档建立共享进度表（按四周排期填入里程碑、每日 blocker、GPU 占用日志三栏）；在 GitHub 创建 Public 仓库 `HW3-DL-Spatial-Intelligence`（建议先 private 一周再转 public），按下文目录结构提交骨架；在服务器 `/root/Test/Zhr/DL/HW3` 下 `git clone` 并初始化 `.gitignore`（排除 `*.ply`, `*.ckpt`, `data/`, `.hf_cache/`, `wandb/` 等大文件目录）；统一确定 GPU 占用约定（推荐 3 号卡作为题目一长任务卡，6 号或 7 号卡作为题目二日常训练卡，使用 `tmux` + `nohup` + 明确的 session 命名 `hw3-t1-gs`、`hw3-t2-act` 等）。

**下午（A）。** 开始安装 `env_hw3_recon` 环境：创建 conda env（Python 3.10），按 3DGS 官方仓库 `environment.yml` 修改 CUDA 版本至 12.1 后安装 PyTorch 2.4（用清华源），编译 `submodules/diff-gaussian-rasterization` 和 `submodules/simple-knn`。COLMAP 安装优先尝试 conda-forge 路线（`conda install -c conda-forge colmap`），若 CUDA 支持有问题再考虑源码编译。下午目标是 `python train.py --help` 能正常打印帮助信息。

**下午（B）。** 开始安装 `env_hw3_robot` 环境：创建 conda env（Python 3.12），按 LeRobot 官方文档安装（`pip install lerobot` + ffmpeg + torchcodec）。验证安装：`python -c "from lerobot.common.policies.act.modeling_act import ACTPolicy; print('ok')"`。同步在 `~/.bashrc` 中固化 HF_ENDPOINT、HF_HOME、PIP_CACHE_DIR 等环境变量。

**晚上（A+B）。** 各自 commit 当天工作；A 整理拍摄方案（确认拍摄对象、约定 5/30 上午拍摄时间）；B 浏览 HF 上 `calvin_task_ABC_D` 数据集页面，阅读 LeRobotDataset 的 parquet schema 文档，确认明天的数据下载与字段审计计划。

**Day 1 完成标志**：GitHub 仓库已创建并 push 骨架；两套 conda 环境均能成功激活并导入核心包；共享进度表运行起来。

### 5/29（周五）Day 2：环境深化与依赖打通

**上午（A）。** 完成 `env_hw3_recon` 的 smoke test：从 3DGS 官方 release 下载预训练好的 garden 场景 `.ply` 文件（约 200 MB），用 `view.py` 或 SIBR viewer 验证渲染（如果服务器无显示器，用 `python render.py` 渲染几张图查看 PSNR）。同时开始安装 `env_hw3_gen3d`（threestudio），按官方 README 安装依赖时注意 PyTorch 与 CUDA 版本对齐到 12.1。

**上午（B）。** 在 `env_hw3_robot` 中通过 ModelScope 或 HF 镜像下载 `calvin_task_ABC_D` 的第一个 shard 到 `/root/Test/Zhr/DL/HW3/topic2_act/data/`（约 50–150 GB，视 shard 大小，先下一个看实际大小再决定后续）。下载完成后用 `datasets.load_dataset` 或直接 pandas 读取 parquet 文件，**重点审计 meta 字段**：环境标签字段的名字是什么（推测可能是 `env`、`scene_id`、`environment` 等）、动作空间是绝对笛卡尔还是关节、proprio state 维度多少、相机有几路、图像分辨率多少。把这些信息记录到 `topic2_act/data/data_audit.md`。

**下午（A）。** 下载 Mip-NeRF 360 的 `counter` 场景（约 5 GB，从官方网站或 GitHub mirror），解压到 `topic1_fusion/data/mipnerf360/counter/`。先用 3DGS 的 `convert.py`（如果数据已经是 COLMAP 格式则跳过）确认目录结构，再用**短训练**（`--iterations 7000`）验证整条管线在 3 号卡上能正常运行，预计耗时 15–30 分钟。日志记入 WandB 项目 `hw3-topic1`。

**下午（B）。** 根据上午的数据审计结果，编写 `topic2_act/scripts/split_env_a.py`：从 LeRobotDataset 中按环境标签过滤出 A 子集，输出为新的 LeRobotDataset 目录或一份 episode index 列表。同时编写 `topic2_act/configs/act_calvin_base.yaml`，确定 ACT 的网络结构、optimizer、batch size、epoch、log frequency 等参数。

**晚上（A+B）。** A 把 3DGS counter smoke test 的渲染结果截图上传仓库；B 把数据审计结果提交。两人讨论确认 5/30 周六的拍摄安排（地点、时间、物体清单、备用方案）。

**Day 2 完成标志**：3DGS 在 counter 上跑通 7k iter；CALVIN 第一个 shard 下载完成并完成字段审计；`env_hw3_gen3d` 安装进行中。

### 5/30（周六）Day 3：拍摄日 + 数据探索

**上午（A+B，约 3 小时）。** 拍摄日。两人协作完成物体 A 和物体 C 的拍摄，按第三章拍摄建议执行。拍摄完成后立即把视频和照片传到服务器（用 `scp` 或 `rsync`），存放路径 `topic1_fusion/data/raw_capture/object_A_video.mp4` 和 `topic1_fusion/data/raw_capture/object_C_*.jpg`。**当天必须验证文件可读、画面无明显失焦**，发现问题立即补拍，不要等到周一。

**下午（A）。** 用 ffmpeg 从物体 A 视频中按 4 fps 抽帧（`ffmpeg -i object_A_video.mp4 -vf "fps=4" -q:v 2 frames/%04d.jpg`），得到约 400–500 帧。编写或复用 `scripts/filter_blurry.py`，用 Laplacian 方差过滤模糊帧，目标保留 150–250 张清晰图像到 `topic1_fusion/data/object_A/images/`。开始 COLMAP SfM：先用 GUI 或命令行的 `feature_extractor` + `sequential_matcher` + `mapper`（**不要用 exhaustive_matcher**），预计耗时 30–90 分钟。

**下午（B）。** 完成 `env_hw3_gen3d` 安装的剩余部分（如果上午没完成）。同时根据数据审计结果，正式启动 A 子集的训练数据准备：如果一个 shard 不够覆盖环境 A 的全部 episode，下载第二个 shard；如果环境 A 占比很大、磁盘吃紧，则按 episode 数量降采样到 200–500 个。

**晚上（A）。** 检查 COLMAP 输出：稀疏点云 `sparse/0/points3D.bin` 是否生成、注册的相机数是否接近输入图像数（理想 90%+）、相机轨迹可视化是否呈合理环形（用 `colmap gui` 或 Python 脚本绘制）。**如果注册率低于 70%，必须排查原因**（最常见是图像模糊太多、纹理太少、或环绕角度不够）；必要时回到 ffmpeg 步骤调整抽帧策略或决定补拍。

**晚上（B）。** 提交 A 子集准备脚本和数据审计文档。开始阅读 CALVIN 官方仓库的 `evaluate_policy.py` 源码，理解 `CustomModel` 接口的 `reset()` 和 `step()` 应当返回什么。

**Day 3 完成标志**：拍摄数据全部上传、视频抽帧完成、COLMAP 稀疏重建质量验证通过；ACT 训练数据准备脚本提交。

### 5/31（周日）Day 4：3DGS 真训练 + ACT 训练启动

**上午（A）。** 用 3DGS 的 `convert.py` 把物体 A 的 COLMAP 输出转成训练所需格式，启动 7k iter smoke 训练（3 号卡，预计 20–40 分钟），看 PSNR 和重建出的 `.ply` 是否合理。如果效果可接受，立即追加启动 30k iter full 训练（预计 1.5–3 小时），训练同时启动 WandB 日志。

**上午（B）。** 在 `env_hw3_robot` 中启动 A-only ACT 训练的 **smoke test**（5 个 epoch，batch size 8，6 号或 7 号卡）：仅验证训练能跑、loss 在下降、checkpoint 能保存、WandB 日志能上传。预计 30–60 分钟。这一步**不追求收敛**，只追求"能跑通"。

**下午（A）。** 等待 30k iter 训练时利用空闲时间：在 3DGS 的 counter 场景上启动 full 30k 训练（如果显存允许另起 5 号卡上跑，否则等物体 A 训完再排队）。同时开始编写 `topic1_fusion/scripts/render_trajectory.py`，用 3DGS 的 render API 沿自定义相机轨迹渲染视频，为后续融合做准备。

**下午（B）。** 根据 smoke test 结果调整超参（如果 loss 不下降，检查数据归一化、动作空间映射、相机字段匹配），启动**正式的 A-only 训练**：建议 50 个 epoch（按 LeRobot 文档 5 epoch ≈ 30 分钟 估算，50 epoch ≈ 5 小时，可以挂晚上跑）。WandB 项目 `hw3-topic2`。

**晚上（A+B）。** A 检查物体 A 的 30k 训练结果，渲染几张视角的图查看几何质量。B 检查 A-only 训练前 10 epoch 的 loss 曲线，确认下降趋势健康；如果发现严重问题（loss 爆炸、不下降），停下来 debug 而不是让它继续浪费 GPU。

**Day 4 完成标志**：物体 A 的 3DGS 30k 训练在运行或已完成；A-only ACT 正式训练在运行（挂夜）。

### 6/1（周一）Day 5：评测脚手架与生成框架准备

**上午（A）。** 检查物体 A 和 counter 场景的 30k 训练成果。如果质量不达标（PSNR 过低、明显 floater），分析原因并决定是否重训（调整 densify 参数或重新筛选输入图像）。如果质量可接受，开始尝试在 Blender 中**导入背景场景的相机轨迹**（用 `cameras.json` 或 transforms 文件解析），为后续融合渲染打基础。这一步是题目一的工程难点之一，预计需要 2–4 小时摸索。

**上午（B）。** 检查 A-only 训练的最终结果（应在夜里跑完）。开始搭建 `env_hw3_calvin_eval` 环境：clone CALVIN 官方仓库到 `topic2_act/calvin_official/`，按其 README 创建 Python 3.8 conda env，安装 PyBullet、egl-probe 等渲染依赖。**这一步很容易踩坑**（PyBullet 在无 X server 环境下的渲染配置），建议预留 2–4 小时。

**下午（A）。** 开始安装并调试 threestudio。`env_hw3_gen3d` 应该已经安装好，这一步重点是下载 Stable Diffusion 1.5、Stable Zero123 等预训练模型到 `HF_HOME` 目录，确认离线模式（`HF_HUB_OFFLINE=1`）下能成功加载模型。启动一个最小化的 DreamFusion 配置（如 `configs/dreamfusion-sd.yaml`，prompt 用官方示例的 "a hamburger"）做 smoke test，**不追求质量、只追求跑通 100 步**，预计 20–40 分钟。

**下午（B）。** 在 `env_hw3_calvin_eval` 中阅读并改写 `evaluate_policy.py`，开始编写 `topic2_act/eval/lerobot_act_wrapper.py`，初步实现 `LeRobotACTWrapper(CustomModel)` 的骨架——`__init__` 加载 checkpoint，`reset()` 初始化内部状态，`step(obs, goal)` 暂时返回零动作（占位）。本步骤目标是确认接口可调用，不需要正确的推理。

**晚上（A+B）。** A 提交 threestudio smoke test 和 Blender 探索笔记。B 提交 wrapper 骨架。讨论第 2 周计划：根据当前进度判断 6/2、6/3 是否还要补任何 Week 1 残留任务。

**Day 5 完成标志**：背景场景训练完成、threestudio smoke test 通过；A-only ACT 训练完成、CALVIN eval 环境搭建完成、wrapper 骨架可调用。

### 6/2（周二）Day 6：连通性验证

**上午（A）。** 编写 `topic1_fusion/scripts/sample_camera_trajectory.py`，从 counter 场景的 COLMAP 相机中插值或自定义一条新的环绕相机轨迹，用 3DGS 的 render API 渲染出一段 10 秒、30 fps 的环绕视频（约 300 帧）。这是后续融合渲染的"骨架轨迹"。同时确认 Blender 中可以正确导入这条轨迹（验证坐标系一致）。

**上午（B）。** 完成 `LeRobotACTWrapper` 的真实推理逻辑：`step()` 内部加载 ACT checkpoint，把 CALVIN obs（dict 含 `rgb_static`, `rgb_gripper`, `robot_obs`）转成 ACT 输入格式（resize 图像到 224×224 或配置指定尺寸、拼接 proprio），调用 `policy.select_action()` 得到 action chunk，按 CALVIN 的 action space（绝对笛卡尔 7-dim 或关节 7+1-dim）映射回去。**这一步是题目二的关键工程瓶颈**，预计 2–4 小时。

**下午（A）。** 把 6/1 的 DreamFusion smoke test 跑到完整训练完（约 3–5 小时在 3 号卡），导出 `--export` 的 obj+mtl。确认 Mesh 可以在 Blender 或 MeshLab 中正常打开、有纹理（不是纯色）。这一步验证了"文本→3D→Mesh→Blender"整条链路可达。

**下午（B）。** 用 `LeRobotACTWrapper` 在 CALVIN 环境 D 上跑**一集** rollout（不是完整评测，只是一个 episode），验证 wrapper 不报错、action 能被环境接受、机器人能动起来（不要求任务成功）。这一步打通了"训练→评测"整条链路。

**晚上（A+B）。** 两人对比当前进度与 Week 1 目标，确认四个核心里程碑（环境齐备、拍摄完成、3DGS 跑通、ACT 跑通）是否全部达成。如果有 gap，决定 6/3 是补漏还是已经可以进入 Week 2 计划。

**Day 6 完成标志**：题目一"文本→Mesh→Blender"链路验证通过；题目二"checkpoint→wrapper→CALVIN rollout"链路验证通过。

### 6/3（周三）Day 7：Week 1 收尾与 Week 2 启动

**上午（A）。** 整理 Week 1 题目一产物：物体 A 的 3DGS 模型（`output/object_A/point_cloud/iteration_30000/point_cloud.ply`）、counter 背景的 3DGS 模型、相机轨迹脚本、threestudio smoke test 的 obj 模型。在 `topic1_fusion/README.md` 中记录每个产物的生成命令和验证方式。

**上午（B）。** 整理 Week 1 题目二产物：A-only 模型 checkpoint（`outputs/act_calvin_a_only/checkpoints/last/`）、`LeRobotACTWrapper`、`split_env_a.py`、`data_audit.md`。在 `topic2_act/README.md` 中记录每个产物的生成命令和验证方式。

**下午（A+B）。** 在主 `README.md` 中合并两条主线的说明，更新到当前进度。开始撰写报告骨架（`reports/main.tex`），按 NeurIPS 模板创建章节：Abstract、Introduction、Related Work、Methods（含两个子节）、Experiments（含两个子节）、Discussion、Conclusion。**报告写作越早启动越好**，因为前几章（背景、相关工作、数据集）不依赖最终结果，可以利用 Week 2 的训练等待时间陆续完成。

**晚上（A+B）。** Week 1 复盘会议（约 1 小时）：列出实际进度与计划的偏差、识别 Week 2 需要重点关注的风险点、调整 Week 2 计划草稿。把 Week 1 的所有 commit 打 tag `week1`。**把当前进度同步给我，以便我制定 Week 2 的精确日度计划。**

**Day 7 完成标志（Week 1 整体）**：两套主线均完成"环境就绪 + 数据齐备 + smoke test 通过"；GitHub 仓库结构完整且 commit 历史清晰；报告骨架已建立；Week 1 进度同步给规划助手。

---

## 六、关键风险与应对预案

整个项目最高优先级的风险有四个，需要在 Week 1 内就建立缓解机制。

**第一，GPU 资源被同事抢占。** 当前 3 号卡 49 GB 全空是难得的窗口，但实验室共享意味着随时可能被占用。应对：5/28 第一时间用 `tmux + nohup` 占住 3 号卡运行一个长任务（即使只是 3DGS 的 toy 训练，也能占住卡）；同时与课题组同学口头沟通，明确说明你接下来三周需要长期使用大显存窗口，争取协调；准备好"降级方案"——如果 3、5 号卡都不可用，3DGS 改用 `--iterations 7000` 短训练，threestudio 用最低 6 GB 显存配置（SD 1.5 + DreamFusion 仅 64×64 渲染）。

**第二，COLMAP 重建失败或质量不佳。** 这是题目一最早可能失败的环节，且失败原因往往在拍摄环节而非软件本身。应对：5/30 拍摄当天就完成第一次 COLMAP 验证，发现问题立即补拍；准备一个"备用物体"，即在拍摄日同时拍 2 个不同物体，万一一个失败可以无缝切换；如果 6/1 仍然失败，启用应急方案——用 Mip-NeRF 360 的现成场景中的某个物体作为"物体 A"（这虽然不完全符合题目要求的"自己拍摄"，但可以在报告中说明"由于拍摄物体反光过强 COLMAP 无法重建，故改用 X"）。

**第三，CALVIN 环境标签字段不存在或与预期不符。** 如果 HF 上的 `calvin_task_ABC_D` 没有显式的环境标签，A 子集切分将非常困难。应对：数据审计是 5/29 的高优先级任务；如果确实没有环境标签字段，备用方案是直接通过 episode 数量比例切分（CALVIN 原始数据中 A、B、C、D 各占 25%，对应到 LeRobot shard 后顺序可能保留），或者参考 CALVIN 官方仓库的环境分组定义反查 episode ID；最坏情况下，可以联系老师询问数据切分方式（**这是合理且必要的求助**）。

**第四，融合渲染在 Week 4 才发现不可行。** Blender 导入 COLMAP 相机轨迹涉及坐标系转换（OpenCV → Blender 的 z 轴翻转、scale 处理），如果到 Week 4 才发现导入失败，时间将非常紧张。应对：把"Blender 导入背景相机轨迹"作为 Week 1 Day 5 的探索任务，**Week 2 Day 1**（6/4）必须验证通过；如果 Blender 路线确实不可行，启用 fallback——所有融合在 Python + 3DGS 的 render API 中完成（把 Mesh 也转成点云形式作为"伪高斯球"插入到 3DGS 的场景中渲染），这条路虽然质量较低但完全代码可控。

此外还有两个次级风险值得提及：threestudio 的 SDS 训练 seed 敏感、单次结果质量不稳定（缓解：Week 3 每个物体跑 3–5 个 seed，挑最好的）；以及 ACT 在环境 D 上 zero-shot 成功率可能很低甚至为零（缓解：作业要求是"对比 A-only 与 ABC 联合在 D 上的表现"，即使两者都很低，**只要能分析出差异**就是合格的报告内容，不必追求绝对成功率）。

---

## 七、立即可执行的开工动作（5/27 周三晚 — 5/28 周四上午）

为了 5/28 周四上午能立即进入 Day 1 上午的项目启动会，建议今晚或明早完成以下五个动作：

第一，与队友确认分工：你担任成员 A（题目一）还是成员 B（题目二）。建议根据两人的偏好——更喜欢 3D 视觉/拍摄/Blender 的人做 A，更喜欢机器人/RL/数据工程的人做 B。第二，在飞书或腾讯文档建立共享进度表，按本计划的"四周周度计划"表格复制为初始版本。第三，在 GitHub 注册 organization 或个人账号下创建仓库 `HW3-DL-Spatial-Intelligence`（先 private），邀请队友加入。第四，登录服务器 `tmux new -s hw3-setup`，在 `/root/Test/Zhr/DL/HW3` 下 `mkdir -p {topic1_fusion,topic2_act,env,reports,assets}` 建好目录骨架。第五，若手机相册中已经有合适的物体素材（按第三章选择标准），可以提前先传一张到服务器评估是否符合 Stable Zero123 输入要求，节省 5/30 的拍摄判断时间。

完成 Week 1 后，请把以下三类信息同步给我，以便制定 Week 2 的精确日度计划：①Week 1 四个核心里程碑的达成情况；②实际遇到的 blocker 或与预期偏差较大的环节；③Week 1 截至 6/3 晚的 GPU 实时占用情况（再贴一次 `nvidia-smi`）。

祝项目顺利。