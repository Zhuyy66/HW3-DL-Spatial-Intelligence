# HW3 期末作业 · 执行方案 Week 3（6/11 周四 — 6/17 周三）

> 课程：深度学习与空间智能 | 队伍：2 人（成员 A 主管题目一，成员 B 主管题目二）
> 工作目录：`/root/Test/Zhr/DL/HW3` | 远程仓库：`github.com/Zhuyy66/HW3-DL-Spatial-Intelligence`
> 截止：2026-06-24 23:59 | 本方案覆盖：Week 3（6/11–6/17）日度计划
> 关键约束变化：剩余任务由原 Week 2–4（三周）压缩为 **Week 3–4（两周）**

---

## 一、整体计划调整说明（先读）

### 1.1 为什么压缩到两周仍然可行

时间窗少了一周，但题目一的实际进度领先约一周，两者基本抵消。截至 Week 1 收口（6/14）的真实状态：

**题目一已完成（含原 Week 2/Week 3 部分内容）：** 物体 A（绿植）COLMAP 注册率 100% + 3DGS 30k（test PSNR 31.33）；背景 counter 3DGS 30k（test PSNR 29.26，质量好）；**物体 B（文本→3D）hamburger DreamFusion 10000 步已完成并导出 OBJ/MTL+纹理**（这本是原 Week 3 任务）；counter 相机轨迹 + Blender 初步验证；garden 预训练参考。

**题目二已完成（仅工程链路）：** 双环境（py3.12 训练 / py3.8 评测）、官方 `xiaoma26/calvin-lerobot` 数据（splitA/B/C/D = 6089/6115/5666/5124）、A-only smoke500 50ep 健康 baseline、`LeRobotACTWrapper` 双环境桥接 + 单集 rollout smoke 全部通过。

### 1.2 两周倒排框架

| 周 | 主题 | 题目一（成员 A） | 题目二（成员 B） | 报告 |
|---|---|---|---|---|
| **Week 3**（6/11–6/17） | 核心实验与产物完成 | A 线全面入库；物体 C 生成；融合渲染成片；三方式对比数据齐备 | dataloader 提速 + 步数标定；全量 A-only 与 ABC 训练；环境 D zero-shot 评测；动作分块分析 | Intro / Related Work / Method / Dataset 完稿 |
| **Week 4**（6/18–6/24） | 收尾 / 终稿 / 复现 / 提交 | 融合视频精修；题目一图表定稿 | 评测交叉验证；超参表与指标表定稿 | Experiments / Discussion / Conclusion + 全部图表；README、权重上传；端到端复现；6/24 buffer |

### 1.3 关键路径与风险定位

题目二是关键路径（严格依赖链：**dataloader 提速 → 步数标定 → 全量 A-only → 全量 ABC → D 评测 → 分析**），且评测侧压着 EGL 未解风险，必须最早动手、最早决策。题目一剩余任务（物体 C、融合、对比、入库）相对独立、风险中等，与题目二并行推进。

---

## 二、贯穿 Week 3 的前置决策（开工前统一口径）

以下五项决定整周排期能否成立，须在 Day 1 内全部敲定，后续日度计划默认遵循。

### 决策 1：全量训练按"固定步数预算"，A-only 与 ABC 严格相同

作业要求 A-only 与 ABC "使用相同网络结构和超参数"，其本质是**两次实验之间一致**，而非等于某个特定数字。继续沿用 Day 4 的结论：**以相同的总梯度步数（batch 固定为 8 时即相同样本数）作为对比基准，而非相同 epoch。** 理由有二：其一，按 epoch 对齐会让 ABC 获得约 3 倍于 A-only 的梯度步数，把"数据更多"与"算力更多"混为一谈，污染结论；其二（时间价值），按固定步数训练，两个模型的 wall-clock 几乎相同，与数据集大小无关——这是两周能装下两个全量训练的前提。

**步数预算建议：先暂定 120,000 步，Day 1 优化后微调。** 依据：smoke500 上 120K 步≈32 epoch，已越过主要收益区（epoch 20≈loss 0.071，epoch 50≈0.051）；全量数据多样性远高于 500 集，120K 步足够。微调规则：Day 1 实测优化后的 s/step 后，若提速效果好（<0.15 s/step）可上探到 150K；若效果差（仍 >0.3 s/step）下调到 80K–100K 以确保两个模型都能在 Week 3 跑完——**无论怎么调，A-only 与 ABC 必须同步数。** 同时在 Day 1 查清服务器上 LeRobot 0.4.0 ACT 的训练步数默认值，作为参考锚点（不照搬，以实测为准）。

### 决策 2：评测采用"双指标"，稳健主指标不依赖仿真

| 指标 | 类型 | 依赖 | 风险 | 用途 |
|---|---|---|---|---|
| **Action L1（D 离线）** | open-loop | splitD 离线数据（已确认 5124 ep） | 低，一定拿得到 | **主指标**，A-only vs ABC 核心对比 |
| **Success Rate（D rollout）** | closed-loop | CALVIN 仿真 + EGL | 中高，EGL 未修复 | 辅助指标，能跑则锦上添花 |

作业要求"成功率**或**动作误差"，因此即使 rollout 因 EGL 失败，open-loop Action L1 也独立满足要求。**动作分块鲁棒性分析**可完全基于 Action L1 展开：对比 chunk 内不同时间步位置的 L1 误差曲线、A-only 与 ABC 在 D 上的误差分布差异，论证 action chunking 在视觉分布偏移下的表现，不依赖成功率。

### 决策 3：EGL 路线 Day 4 一次性决策，不等管理员

主路线修复严格 NVIDIA EGL；若 Day 4 当天无法自行修复（需管理员配合），**直接切换到已验证可跑的 direct-cameras 路线**并在报告中透明声明替代评测路线，不再等待。核心是 A-only 与 ABC 用**同一评测路线**，对比即成立。时间不允许把关键路径押在外部依赖上。

### 决策 4：ABC 联合数据集加载方案 Day 2 确认

splitA/B/C 的 episode index 都从 0 开始，不能直接拼接。两条可选路线：multi-dataset 加载（LeRobot 多 `repo_id`/`MultiLeRobotDataset`），或合并为单数据集并 remap episode index。Day 2 二选一并验证读取正确（抽样核对三个 split 的样本都被覆盖、无 index 冲突）。

### 决策 5：物体 B 定稿用 hamburger，不重跑

hamburger 已有完整产物（10000 步、OBJ/MTL、纹理、test 视频、法向图），质量达标，且与 counter 厨房台面背景语义自洽（汉堡放厨房合理）。时间压缩下不再为换自选 prompt 重跑——若报告需要，在方法章节说明"采用 threestudio 官方示例 prompt 验证文本→3D 全链路"即可。**省下的时间投入物体 C 和融合渲染这两个真正的缺口。**

---

## 三、Week 3 日度执行方案

> 每日列出【工作】【交付物】【验收/风险】。两条线并行，成员 A／B 各自推进，报告写作穿插于训练等待时间。
> 服务器正式命令前统一 `source scripts/activate_cuda_driver_shim.sh`。ACT 训练/评测为 IO 受限任务，放 24GB 卡；大卡（49GB）留给题目一的 3DGS/threestudio。每次启动前 `nvidia-smi` 按空闲选卡并在共享进度表登记。

### Day 1 · 6/11（周四）：A 线入库 + B 线提速攻坚 + 物体 C 输入准备

**工作**
- **成员 B（关键路径起点）：** 攻坚 dataloader IO 瓶颈。当前 `data_s≈0.31` vs `updt_s≈0.075`（实测 0.397 s/step），GPU 约 80% 时间空等数据。优先手段：提高 `num_workers`、开 `persistent_workers` 与 `prefetch_factor`，确认 parquet 图像解码未卡在单进程。优化后在全量 splitA 上跑短测（数千步）实测新的 s/step。同步查清 LeRobot 0.4.0 ACT 训练步数默认值。
- **成员 A（入库优先）：** 按 Week 1 同步报告"待入库清单"补 `topic1_fusion/README.md`——Week 1 产物索引（哪些进 Git、哪些走外部存储），小型配置/训练命令/相机轨迹脚本/Blender 验证脚本入库；飞书截图原图导出到 `reports/figures/`；WandB 链接（topic1: `wandb.ai/zhuyy666-fudan-university/hw3-topic1`）写入 README。
- **成员 A（物体 C 启动）：** 准备物体 C 输入图——若尚无照片，按"轮廓清晰、有正面语义、与厨房台面尺度协调"标准选一个小物件（杯/罐/水果类）拍一张正面照（纯白/灰背景、主摄非广角、主体占画面 60–70%）；用 `rembg` 或 SAM 抠背景为干净 RGBA（512²或 1024²，主体居中）。
- **报告：** 启动 Introduction、Related Work 写作（不依赖最终结果）。

**交付物**
- 优化后的 dataloader 配置 + 实测 s/step 数字（更新到训练 runbook）。
- `topic1_fusion/README.md` Week 1 产物索引初版 + 入库的小型脚本/配置。
- 物体 C 干净 RGBA 输入图（`topic1_fusion/data/object_C/input_rgba.png`）。
- 报告 Intro/Related Work 草稿段落。

**验收/风险**
- 验收：优化后 s/step 明显低于 0.397；A 线 README 能让他人看懂哪些产物在哪、如何复现。
- 风险：若 dataloader 提速效果有限（仍 >0.3），立即按决策 1 下调步数预算，不在优化上无限投入；物体 C 输入图若主体抠图边缘脏，会直接拖累 Zero123 质量，务必当天确认干净。

### Day 2 · 6/12（周五）：启动全量 A-only + 物体 C 生成 + ABC 加载方案

**工作**
- **成员 B：** 据 Day 1 实测 s/step 最终敲定步数预算（决策 1），用 `episodes_A_full.json`（6089 集）启动**全量 A-only 正式训练**（24GB 卡，tmux+nohup，WandB 项目 hw3-topic2）。确认日志**单独记录 Action L1 分量**（非仅混合 loss），否则后续画 Action L1 曲线无数据。同步确认 ABC 联合加载方案（决策 4）并写好 ABC 训练配置（仅数据范围不同，结构/超参/步数与 A-only 完全一致）。
- **成员 A：** 物体 C 用 `configs/stable-zero123.yaml` 启动 SDS 生成（大卡）；若输入非严格正面，显式设置 `elevation`/`azimuth`。生成中若出现 Janus/漂浮几何，按 README 顺序处理（换 seed → prompt debiasing/Perp-Neg）。
- **报告：** Method 章节题目一部分（三种 3D 生成路径原理）。

**交付物**
- 全量 A-only 训练启动并健康运行（前若干 epoch loss 正常下降），WandB run 链接。
- 最终步数预算数字 + ABC 训练配置文件（已验证加载正确）。
- 物体 C SDS 训练启动（或首轮结果）。
- 报告 Method（题目一）草稿。

**验收/风险**
- 验收：A-only 训练 loss 健康、Action L1 已单独记录、checkpoint 正常落盘；ABC 加载抽样核对三 split 全覆盖、无 index 冲突。
- 风险：A-only 启动后前 10 epoch 若 loss 异常立即停训 debug，不浪费 GPU；物体 C 是题目一最大内容缺口，若首轮质量差当天就换 seed 重试，不拖到后面。

### Day 3 · 6/13（周六）：启动全量 ABC + 物体 C 导出 + 融合坐标对齐

**工作**
- **成员 B：** A-only 训练完成后（或并行用另一张 24GB 卡），启动**全量 ABC 联合训练**（同步数预算，配置仅数据范围不同）。A-only 完成后建立稳定软链接 `.../a_only/last` 指向最终 checkpoint（评测命令引用软链接而非易变 step 路径）。
- **成员 A：** 物体 C 训练完成后 `--export` 导出带纹理 mesh（obj+mtl），在 Blender/MeshLab 确认可打开、有纹理。**启动融合渲染坐标系对齐**——这是题目一最大工程风险：在 Blender 中导入 counter 背景的 COLMAP 相机轨迹，处理 OpenCV→Blender 坐标转换（z 轴翻转、scale），验证物体 B/C 两个 mesh 能在统一世界坐标系下正确放置。
- **报告：** Dataset 章节（CALVIN 数据描述 + splitA/B/C/D 统计 + 6089 交叉验证、Mip-NeRF 360 counter、object_A 采集）。

**交付物**
- 全量 ABC 训练启动并健康运行，WandB run 链接。
- A-only `last` 软链接 + A-only 最终 checkpoint 就位。
- 物体 C 带纹理 mesh 导出完成。
- Blender 相机轨迹导入验证（坐标系一致的截图/说明）。
- 报告 Dataset 草稿。

**验收/风险**
- 验收：ABC loss 健康；物体 C mesh 有纹理非纯色；Blender 中 B/C mesh 放置位置与背景尺度合理。
- 风险：Blender 坐标系对齐若失败，启用 fallback——在 Python+3DGS render API 中完成融合（mesh 采样为点云作"伪高斯球"插入 3DGS 场景渲染），质量略低但完全代码可控；此 fallback 决策不晚于 Day 4。

### Day 4 · 6/14（周日）：EGL 决策 + A-only 评测 + 融合渲染推进

**工作**
- **成员 B（评测路线决策日）：** 按决策 3 一次性决定 EGL 路线——能自修则修，否则切 direct-cameras 并记入报告口径。完成 **A-only 在环境 D 的评测**：主指标 open-loop Action L1（splitD 离线数据，稳健），辅指标 closed-loop success rate（能跑则跑）。评测只需向 worker 传 A-only checkpoint 路径，其余复用 Day 6 已验证的桥接（client 薄、worker 厚、reset 清 action 队列）。
- **成员 A：** 推进融合渲染——将物体 A（高斯）、B（hamburger mesh）、C（mesh）按合理比例/位置插入 counter 背景。物体 A 若能经 SuGaR 类工具转 mesh 则一起进 Blender，否则用双 pass 合成（背景+B/C 前景 RGBA pass，物体 A 走 3DGS 同轨迹背景 pass，ffmpeg alpha 叠加）。
- **报告：** 补充 Method 题目二部分（ACT 原理、动作分块、双环境评测架构 + 协议图）。

**交付物**
- EGL 路线决策记录（写入审计文档与报告口径）。
- A-only 评测结果：D 上 Action L1（必有）+ success rate（可选），存入 `topic2_act/eval/results/`。
- 融合渲染首版（三物体已就位的若干帧或短片）。
- 报告 Method（题目二）草稿。

**验收/风险**
- 验收：A-only 至少拿到 D 上 Action L1 数值；融合首版三物体位置/光照不明显穿帮。
- 风险：rollout 若机器人行为离谱而桥接保真度测试通过，第一嫌疑是 action 空间错配（绝对笛卡尔 vs rel_actions），核对训练 action key 与 env action 模式一致——但有 Action L1 兜底，不阻塞。

### Day 5 · 6/15（周一）：ABC 评测 + 三方式对比数据 + 融合成片

**工作**
- **成员 B：** ABC 训练完成后建立 `.../abc/last` 软链接，完成 **ABC 在环境 D 的评测**（同 A-only 流程与指标，仅换 checkpoint）。汇总 A-only vs ABC 的 D 上 Action L1（及 success rate），整理成对比表。
- **成员 A：** 整理题目一**三方式对比数据**——物体 A（多视角重建）/ B（文本生成）/ C（单图生成）在几何准确度、纹理细节、计算耗时上的差异（耗时数据来自各自训练日志，质量来自 render 对照）。完成融合**多视角漫游渲染视频**成片（基于 counter orbit 轨迹）。
- **报告：** 开始 Experiments 章节框架，导出已有曲线/对照图到 `reports/figures/`。

**交付物**
- ABC 评测结果 + A-only vs ABC 对比表（D 上 Action L1 为主）。
- 题目一三方式对比表（几何/纹理/耗时）。
- 融合漫游视频成片（mp4）。
- 报告 Experiments 框架 + 首批图表入库。

**验收/风险**
- 验收：A-only 与 ABC 对比表数值完整可追溯到日志；三方式对比表三个物体齐全；融合视频三物体在多视角下空间关系稳定。
- 风险：若 ABC 评测显示与 A-only 差异极小甚至 ABC 更差，**这是合格的可分析结果**（作业要求"对比"而非"ABC 必须更好"），重点转向现象分析（为何动作分块在视觉偏移下表现如此）；融合视频若某视角穿帮，裁剪轨迹避开即可。

### Day 6 · 6/16（周二）：动作分块鲁棒性分析 + 题目一图表定稿 + 报告整合

**工作**
- **成员 B：** 完成**动作分块（action chunking）鲁棒性分析**——基于 D 上 Action L1，分析 chunk 内不同时间步位置的误差变化、A-only 与 ABC 误差分布对比，论证 chunking 在跨环境视觉分布偏移下的鲁棒性。整理题目二超参详表（网络结构、batch、lr、optimizer、步数预算、chunk size 等）与指标表。
- **成员 A：** 题目一全部图表定稿（3 物体 render 对照、融合视频关键帧、三方式对比表、PSNR/loss 曲线），统一导出到 `reports/figures/`；补 `topic1_fusion/README.md` 的训练/复现命令。
- **报告：** 两人整合各自 Experiments 章节内容，填入图表与表格。

**交付物**
- 动作分块鲁棒性分析文字 + 支撑图（误差曲线/分布）。
- 题目二超参表 + 指标表。
- 题目一全部图表定稿入库。
- 报告 Experiments 章节基本成形。

**验收/风险**
- 验收：动作分块分析有数据支撑而非空泛描述；两条线图表都能追溯到 WandB/log/manifest。
- 风险：本日为 Week 3 内容收口日，若任一线仍有训练/评测未完成，优先保证"有完整可分析结果"而非追求最优数值。

### Day 7 · 6/17（周三）：Week 3 收口 + Week 4 启动 + 进度同步

**工作**
- **两人：** Week 3 复盘——核对两条线核心产物完成度（见第五节验收清单）；将 Week 3 所有 commit 打 annotated tag `week3`，注释中如实记录两线真实状态。
- **成员 B：** 确认 A-only/ABC 两套权重、评测结果、所有曲线齐备；整理待上传云盘的权重清单。
- **成员 A：** 确认融合视频、三物体 mesh/ply、三方式对比齐备；整理待上传云盘的产物清单。
- **报告：** 确认 Intro/Related Work/Method/Dataset 已完稿，Experiments 主体成形，列出 Week 4 待补的 Discussion/Conclusion 与剩余图表。

**交付物**
- Week 3 验收清单逐项确认结果。
- `week3` annotated tag。
- 两条线待上传云盘权重/产物清单。
- 给规划助手的 Week 3 进度同步（见第七节）。

**验收/风险**
- 验收：题目二两套全量模型 + D 评测对比 + 动作分块分析齐备；题目一物体 C + 融合视频 + 三方式对比齐备；报告四章完稿。
- 风险：诚实记录未达成项，转入 Week 4 倒排；不为收口粉饰任一线状态。

---

## 四、两人分工与协作

| 维度 | 成员 A（题目一） | 成员 B（题目二） |
|---|---|---|
| 主线 | 物体 C 生成 → 融合渲染 → 三方式对比 → A 线入库 | dataloader 提速 → 步数标定 → 全量 A-only/ABC → D 评测 → 动作分块分析 |
| GPU | 大卡（49GB）跑 3DGS/threestudio/Zero123 | 24GB 卡跑 ACT 训练与评测（IO 受限，不占大卡） |
| 报告主笔 | Method/Experiments 题目一部分、题目一全部图表 | Method/Experiments 题目二部分、超参表/指标表 |
| 收口 | 题目一产物 + GitHub 题目一目录 | 题目二代码/权重 + GitHub 题目二目录 |

协作要点：每日结束 30 分钟同步进度与 blocker（共享进度表 + GPU 占用登记栏）；报告 `reports/main.tex` 两人共同维护，各自主笔自己章节；GitHub 最终收口由成员 A 统一（题目一产物形态更复杂）。

---

## 五、Week 3 验收清单（Day 7 逐项核对）

**题目二（关键路径）**
- [ ] dataloader 提速完成，实测 s/step 记录在案
- [ ] 步数预算敲定，A-only 与 ABC 严格同步数
- [ ] 全量 A-only 训练完成，WandB 曲线 + checkpoint + `last` 软链接
- [ ] 全量 ABC 训练完成，WandB 曲线 + checkpoint + `last` 软链接
- [ ] Action L1 已单独记录（非仅混合 loss）
- [ ] A-only 在 D 的评测（Action L1 必有，success rate 可选）
- [ ] ABC 在 D 的评测，A-only vs ABC 对比表
- [ ] 动作分块鲁棒性分析（有数据支撑）
- [ ] EGL 路线决策已记录入报告口径

**题目一**
- [ ] 物体 C（Stable Zero123）生成 + 带纹理 mesh 导出
- [ ] 融合渲染：三物体插入 counter + 多视角漫游视频成片
- [ ] 三方式对比表（几何/纹理/耗时）
- [ ] A 线全面入库：README 产物索引 + 脚本/配置 + 图表 + 外部产物清单
- [ ] object_A 外观局限已在报告中透明说明（或最短路径修复）

**报告与版本**
- [ ] Intro / Related Work / Method / Dataset 完稿
- [ ] Experiments 章节主体成形 + 首批图表入库
- [ ] `week3` annotated tag 已打，注释记录真实状态

---

## 六、风险登记与应对

| 风险 | 概率 | 影响 | 应对 |
|---|---|---|---|
| dataloader 提速不及预期（仍 >0.3 s/step） | 中 | 全量训练超时 | 下调步数预算至 80K–100K，A=ABC 同步数；ACT 仍放 24GB 卡 |
| 严格 EGL 无法自修 | 中高 | closed-loop 评测受阻 | Day 4 切 direct-cameras + 报告透明声明；open-loop Action L1 为主指标兜底 |
| Blender 融合坐标系对齐失败 | 中 | 融合视频受阻 | fallback：Python+3DGS render API 中以"伪高斯球"融合，代码可控 |
| 物体 C 质量差（Janus/漂浮） | 中 | 题目一缺口 | 换 seed → prompt debiasing/Perp-Neg；Day 2 启动留足重试窗口 |
| object_A 外观过拟合（train 37.7 vs test 31.3） | 已知 | 资产观感 | 点云清理/输入帧筛选，或报告中作为"多视角重建局限"透明分析 |
| ABC 评测与 A-only 差异极小或更差 | 中 | 叙事预期 | 作业要求"对比"非"更优"，转向现象分析即合格 |
| GPU 被同事抢占 | 中 | 训练中断 | tmux+nohup 占稳 + 占用登记；24GB 卡间按空闲切换 |
| A 线产物始终未入库 | 已知高 | 最终复现/README 断裂 | Day 1 即作最高优先级补入库，不拖延 |

---

## 七、Week 3 结束需同步给规划助手的信息

为制定 Week 4 精确日度计划，Day 7 请同步：

1. **题目二实测数据：** 优化后 s/step、最终步数预算、A-only 与 ABC 的最终 loss/Action L1、D 上评测对比数值、EGL 路线最终走向。
2. **题目一完成度：** 物体 C 是否达标、融合视频是否成片、三方式对比是否齐全、A 线入库进度。
3. **报告进度：** 四章完稿情况、Experiments 缺口、还差哪些图表。
4. **未达成项与偏差：** Week 3 哪些验收项未完成、原因。
5. **GPU 实时占用：** 6/17 晚的 `nvidia-smi`。

拿到这五类信息后，Week 4 计划将围绕"报告终稿 + 全部图表 + README/权重上传 + 端到端复现 + 6/24 buffer"精确排期，并把两条线最后的收尾与提交准备对齐到截止时间。
