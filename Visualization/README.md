# Visualization

绘图代码放在 `Scripts/`，图表放在 `Plots/`。在仓库根目录激活 `act_simple` 后运行：

```powershell
python Visualization/Scripts/lift_success.py
```

脚本默认使用共享的第 10,000 步 Lift 权重，运行 50 个回合，每回合最多 200 步。它用种子 `42` 初始化仿真，每回合重置场景和 ACT 动作队列，按默认 Lift 奖励判断成功。相机在后台渲染，柱状图写到 `Plots/lift_success.png`。

可以通过位置参数换权重，通过 `--episodes`、`--steps`、`--seed`、`--device` 和 `--output` 调整实验。已有环境缺少绘图库时，在仓库根目录执行 `python -m pip install -r requirements.txt`。
