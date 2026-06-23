# A 同学 5/29 执行手册

适用日期：2026-05-29

今天 A 的目标只有三件事，不要扩散：

1. 把 `env_hw3_recon` 验证到能正常跑 3DGS 命令。
2. 把 `env_hw3_gen3d` 安装推进到依赖开始稳定安装。
3. 把 `counter` 场景跑通一次 `7k` iteration 的 3DGS 短训练。

## 上午：`env_hw3_recon` smoke test

先进入重建环境：

```bash
conda activate env_hw3_recon
cd /root/HW3/topic1_fusion/code/gaussian-splatting
```

先跑核心检查：

```bash
python train.py --help
python render.py --help
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "import diff_gaussian_rasterization, simple_knn; print('core 3dgs extensions ok')"
colmap -h | head
```

再跑可选检查：

```bash
python -c "import fused_ssim; print('fused_ssim ok')"
```

通过标准：

- `train.py --help` 和 `render.py --help` 正常输出。
- `diff_gaussian_rasterization` 和 `simple_knn` 可导入。
- `colmap -h` 能打印帮助。
- `fused_ssim` 缺失不阻塞今天任务，训练会自动退回普通 `ssim`。

如果想补装 `fused_ssim`，在 `gaussian-splatting` 根目录执行：

```bash
pip install -e submodules/fused-ssim
```

如果有预训练 `garden` 模型和对应数据，直接跑：

```bash
bash /root/HW3/topic1_fusion/scripts/run_garden_smoke_render.sh \
  /path/to/garden_model \
  /path/to/garden_dataset \
  3
```

如果现在还没有下载好 `garden`，这一步可以用 `counter` 的短训练代替，不阻塞今天主线。

## 上午并行：启动 `env_hw3_gen3d`

安装说明已经整理在 [env_hw3_gen3d_setup.md](/root/HW3/env/env_hw3_gen3d_setup.md)。

建议顺序：

```bash
conda activate env_hw3_gen3d
cd /root/HW3/topic1_fusion/code/threestudio

python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt -c constraints_hw3.txt
pip install -e .
```

今天的目标不是把 DreamFusion 跑起来，而是把环境安装推进到：

- `pip install -r requirements.txt -c constraints_hw3.txt` 已开始且没有版本方向性错误。
- `torch==2.4.0`、`cuda 12.1` 保持一致。

## 下午：`counter` 场景 7k 短训练

期望数据位置：

```text
/root/HW3/topic1_fusion/data/mipnerf360/counter/
```

目录里至少要有：

- `images/`
- `sparse/0/`

如果 `counter` 已经是 COLMAP 结构，就不用跑 `convert.py`。先检查目录：

```bash
find /root/HW3/topic1_fusion/data/mipnerf360/counter -maxdepth 2 -type d | sort
```

然后直接启动短训练：

```bash
conda activate env_hw3_recon
bash /root/HW3/topic1_fusion/scripts/run_counter_7k.sh \
  /root/HW3/topic1_fusion/data/mipnerf360/counter \
  /root/HW3/topic1_fusion/outputs/counter_7k \
  3
```

训练结束后渲染测试集：

```bash
conda activate env_hw3_recon
cd /root/HW3/topic1_fusion/code/gaussian-splatting
CUDA_VISIBLE_DEVICES=3 python render.py \
  -m /root/HW3/topic1_fusion/outputs/counter_7k \
  --iteration 7000 \
  --skip_train
```

重点查看：

- `test/ours_7000/renders/`
- `test/ours_7000/gt/`

## 晚上：今天需要提交什么

至少留出以下产物：

1. `counter_7k` 的训练目录。
2. 一张或几张 `renders` 截图，放到 `topic1_fusion/assets/day2_counter_smoke/`。
3. 记录今天命令和结果的简短备注。

建议备注模板：

```text
Date: 2026-05-29
GPU: 3
env_hw3_recon: pass / blocked
env_hw3_gen3d: installing / blocked
counter_7k: pass / blocked
Main blocker:
Next action for 5/30:
```

## 卡住时的优先级

1. 先保 `counter 7k` 跑通，这是今天最重要的可视化产物。
2. 再保 `env_hw3_recon` 能稳定启动命令。
3. `env_hw3_gen3d` 今天只要求进入正确安装轨道，不要求完全装完。
