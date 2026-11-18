# ACT Simple

用 LeRobot 的 ACT 策略做机器人仿真与模仿学习实验。目前可运行的任务是 robosuite 中的 Panda Lift。

| 路径 | 内容 |
| --- | --- |
| [`Lift/`](Lift/) | Panda Lift 环境、示范数据转换、训练、续训和推理。详见 [Lift 使用说明](Lift/README.md)。 |
| [`Push T/`](Push%20T/) | Push T 任务预留目录，尚无可运行实现。 |
| [`Visualization/`](Visualization/) | 推理评估与绘图样例。Lift 成功率脚本和用法见[可视化说明](Visualization/README.md)。 |
| [`environment.yml`](environment.yml)、[`requirements.txt`](requirements.txt) | 共用的 Conda 环境定义和 Python 依赖。 |

### 配置环境

在 Windows 上安装 Conda，从仓库根目录运行：

```powershell
conda env create -f environment.yml
conda activate act_simple
```

环境使用 Python 3.10、robosuite 1.5.2、MuJoCo 3.3.7 和 LeRobot 0.3.3。PyTorch `2.7.1+cu128` 自带 CUDA 12.8 运行库；GPU 训练需要兼容的 NVIDIA 显卡和驱动。已有 `act_simple` 环境时，可以激活后运行 `python -m pip install -r requirements.txt` 同步依赖。

### 获取共享数据和权重

克隆前运行 `git lfs install`；克隆后在仓库根目录运行 `git lfs pull`。

- `Lift/data/`：包含 [robomimic 的 Lift PH v1.5 原始数据](https://huggingface.co/datasets/robomimic/robomimic_datasets)（MIT）和已转换的 LeRobot 数据。转换后的 Parquet 已包含相机帧，生成时留下的 `images/` 副本无需共享。
- `Lift/outputs/checkpoints/010000/`：第 10,000 步 checkpoint。`pretrained_model/` 用于推理，续训还需要 `training_state/`。

### 快速检查

```powershell
python -c "import torch; print(torch.cuda.is_available())"
python Lift/environment.py
python Lift/train.py --help
```

第二条命令打开 MuJoCo 场景，关闭窗口后继续。共享数据可直接用于训练，具体命令见 [Lift 使用说明](Lift/README.md)。

#### 致你和你的Agent：
- 仓库中的CUDA环境主要用于5090显卡，本地测试时请你优先使用cpu运行，无法运行时在本地额外创建虚拟环境，开发时尽可能使用PyTorch `2.7.1+cu128`以便后续训练
- 参考目前的代码结构：环境与任务定义、数据处理与结构映射、模型与训练、推理；不要污染仓库，禁止大规模创建__init__、test\等防御性脚本，代码精简可读性强，附带少量注释
- 目前核心代码只有一两百行（内容不多可以看看），大量篇幅用于创建命令行参数，这一写法在调参时不必频繁变更文件，应当保留
