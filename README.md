# EKF-LSTM Slung-Load Swing-Angle Reproduction

基于 **EKF + LSTM** 的无人机吊载摆角估计半物理仿真复现实验。

本仓库整理了复现实验使用的代码、PX4 SITL / Gazebo Classic 仿真数据、训练与独立测试结果，以及完整的数据来源和复现说明。实验首先使用扩展卡尔曼滤波器（EKF）依据飞行状态估计吊载摆角，再由长短期记忆网络（LSTM）学习并补偿 EKF 的系统性误差。

> 当前整理版本采用仓库基线流程进行离线验证，目标是确认 EKF 与 EKF-LSTM 在独立飞行数据上的效果。论文方向的进一步改进应建立在该基线结果之上。

## Repository layout

```text
.
├── README.md
├── results/                     # 训练曲线、测试对比图与结果汇总表
├── code/                        # 复现实验脚本、核心算法与仿真配置
└── datasets/                    # 仓库原始数据与新生成的仿真数据
```

每个部分均附带独立 README 和 SHA-256 文件清单，用于说明文件用途并验证复制完整性。

## Environment

- Windows + WSL2
- Ubuntu 20.04
- PX4 SITL
- Gazebo Classic
- ROS / MAVROS
- Python / PyTorch
- Sampling rate: 50 Hz
- Sequence length: 400 samples (8 s)
- LSTM input dimension: 13
- LSTM architecture: 2 layers, 64 hidden units per layer
- Loss: mean squared error (MSE)

## Dataset split

| Split | Flights | Purpose |
|---|---:|---|
| Training | 10 | Train the new-data-only model and participate in mixed training |
| Validation | 4 | Select the best epoch without tuning on the test set |
| Test | 4 | Final independent evaluation |

训练、验证和测试飞行相互独立。每次新仿真飞行均提供 CSV 数据和配套 JSON 元数据。仿真采用约 0.5 kg 吊载、2 m 绳长和 0.05 kg 绳质量；真值摆角由 Gazebo 中无人机与吊载连接位置的相对关系计算。

## Estimation pipeline

```text
Flight log / simulation state
             |
             v
    Extended Kalman Filter
             |
             v
  LSTM error compensation
             |
             v
 Corrected swing-angle estimate
             |
             v
Comparison with Gazebo ground truth
```

输出包含两个正交水平平面内的摆角分量：`xi` 与 `zeta`。

## Experiments

- **New-data-only model:** trained for 50 epochs; epoch 39 selected using validation data.
- **Mixed-data model:** trained for 35 epochs on repository and new simulation data; epoch 33 selected using validation data.

混合训练曲线根据已保存的 1–35 轮训练记录重新绘制；整理仓库时没有重新训练，也没有重新生成测试数据。

## Contents

### Results

`results/` 包含训练损失曲线、EKF/EKF-LSTM/真值对比图、详细误差指标及可编辑的 Excel 汇总表。

### Code

`code/` 包含数据采集、训练与统一测试脚本，EKF/LSTM 核心实现，轨迹协议，以及 PX4 SITL / Gazebo 场景和吊载模型配置。

### Datasets

`datasets/` 包含原仓库训练/测试数据，以及本次独立生成的训练、验证和测试飞行数据。详细来源和字段说明见该目录的 README。

## Interpretation boundary

本仓库可用于验证完整的数据生成、训练、验证和独立测试流程，并量化比较 EKF 与 EKF-LSTM。单次训练结果不代表模型对所有轨迹、载荷参数和环境均具有普遍优势；进一步研究应增加随机种子、未见轨迹、不同绳长/载荷质量和真实飞行试验。

## Integrity

各子目录中的 `文件清单_SHA256.csv` 记录文件相对路径、大小和 SHA-256，可用于下载后的完整性校验。

## License

本整理仓库暂未声明独立许可证。原始仓库文件仍应遵守其原有许可证与署名要求；使用或再发布前请核对原项目许可条款。
