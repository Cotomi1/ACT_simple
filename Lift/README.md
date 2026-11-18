# ACT Simple：Panda Lift

在 Windows 上用 robosuite 的默认 Panda Lift 场景验证 LeRobot ACT 的训练和推理。

先按[仓库根目录 README](../README.md) 创建并激活 `act_simple` 环境。以下命令在 `Lift/` 目录执行。

运行 `python environment.py` 打开 MuJoCo 原生 viewer，可用鼠标查看场景，关闭窗口即退出。项目没有键盘遥操作或人工采集。

| 文件 | 用途 |
| --- | --- |
| `environment.py` | 创建默认 Panda Lift、打开 viewer，并提供训练和推理共用的观测定义。 |
| `data.py` | 读取官方示范 HDF5，按原始模拟器状态渲染相机图像并转换为本地 LeRobot 数据。 |
| `train.py` | 训练 LeRobot ACT，支持完整 checkpoint 续训和从本地权重开始新训练。 |
| `infer.py` | 加载 LeRobot checkpoint，在同一仿真环境运行一回合并报告是否抬起方块。 |

数据来自 [robomimic 官方 Lift PH v1.5](https://huggingface.co/datasets/robomimic/robomimic_datasets/blob/main/v1.5/lift/ph/demo_v15.hdf5)，共 200 条轨迹、9,666 步，采样频率为 20 Hz。Git LFS 拉取完成后可直接使用 `data/panda_lift/`；如需重新转换，脚本会优先读 `data/demo_v15.hdf5`，原始文件缺失时从 Hugging Face 下载，也可用 `--source` 指定本地文件。

每步输入为 7 个关节位置、2 个手指位置和两张 84×84 RGB 图像（`agentview`、`robot0_eye_in_hand`），训练目标为 7 维世界坐标系 OSC 增量动作。转换时用同一帧的模拟器状态和动作配对；推理时把平移和旋转转入默认 Panda 控制器的基座坐标系。

```powershell
# 转换全部轨迹；仅在 data/panda_lift 不存在时运行
python data.py

# 只转换前 2 条，检查数据格式和相机；输出必须是新目录
python data.py --episodes 2 --output data/lift_smoke

# 准备好后手动启动训练；首次训练可能下载默认 ResNet18 的初始化权重
python train.py --steps 10000 --batch-size 8 --device cpu

# 加载共享的第 10,000 步 checkpoint，在 viewer 中运行一回合
python infer.py --device cpu
```

要推理自己的模型，给 `infer.py` 传入对应的 `pretrained_model/` 目录。

### 续训本地 checkpoint

完整续训需要保留同一编号目录下的 `pretrained_model/` 和 `training_state/`。`--resume` 恢复权重、优化器、随机数状态和已完成步数；有学习率调度器时也会恢复它的状态。以下命令从第 10,000 步继续到第 15,000 步，实际再更新 5,000 次：

```powershell
python train.py --resume outputs/checkpoints/010000 --output outputs/lift_continue --steps 15000 --device cpu
```

续训沿用保存的数据路径和批量大小，默认把结果保存到 checkpoint 所在的训练目录。如果旧的默认数据路径已失效，会尝试使用 `Lift/data/panda_lift/`；其他数据可通过 `--dataset` 指向搬迁后的位置。中断后可以省略 `--steps`，继续到原目标步数；已达到原目标时，需要传入更大的总步数。如果取出了单个 checkpoint，或从较早的 checkpoint 分支训练，请用 `--output` 指定新目录。

如果只保留了 `pretrained_model/` 中的 `config.json` 和 `model.safetensors`，可以从这些权重开始新训练。优化器和步数重新初始化，沿用权重内的观测和动作归一化统计。数据应使用同样的相机、状态定义、动作含义和频率。

```powershell
python train.py --weights outputs/checkpoints/010000/pretrained_model --output outputs/lift_finetune --steps 5000 --device cpu
```

### `train.py` 参数

| 参数 | 用法 |
| --- | --- |
| `--dataset PATH` | 转换后的 LeRobot 数据目录，包含 `meta/info.json`。默认用 `Lift/data/panda_lift/`；续训优先沿用保存的路径。 |
| `--output PATH` | 保存 checkpoint 的目录。新训练默认用 `Lift/outputs/act_lift/`，要求目录尚未创建；续训默认用编号 checkpoint 所在的训练目录。 |
| `--steps N` | 优化器更新的**总步数**。新训练默认 `10000`；续训默认沿用原目标，且目标必须大于已完成步数。 |
| `--batch-size N` | 每步使用的样本数。新训练默认 `8`；续训沿用原设置，可按显存调整。 |
| `--device {cpu,cuda}` | 训练设备。新训练默认自动选择；续训优先使用 checkpoint 保存的设备。训练在 5090 D v2 上时可显式传 `cuda`。 |
| `--resume CHECKPOINT` | 从包含权重和训练状态的编号目录恢复。也可传入其中的 `pretrained_model/`。 |
| `--weights MODEL` | 只加载本地权重来启动新训练。可传编号目录、`pretrained_model/` 或其中的 `model.safetensors`，与 `--resume` 互斥。 |
| `-h` / `--help` | 显示命令行帮助。 |
