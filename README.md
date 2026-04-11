# KITTI-360 Satellite-to-Frontview Generation with Stable Diffusion

基于 Stable Diffusion 的卫星图到前视图生成框架，完全移除 IPM（逆透视映射）中间步骤，直接使用卫星图和相机位姿，通过坐标位置编码和 self-attention 保持空间一致性。

## 核心创新

1. **直接端到端生成**：移除传统 IPM 中间步骤，卫星图 + 位姿 → 前视图
2. **坐标位置编码**：为卫星图特征添加前视图 token 在 BEV 图上的相对位置编码
3. **空间一致性保持**：通过 self-attention 保持卫星图内部的空间几何关系
4. **几何读写对齐**：通过前视 token 的 BEV 坐标与卫星 token 建立读写对齐

## 项目架构

```
├── configs/              # 配置文件（训练、推理）
├── data/                 # 数据加载模块
│   ├── Kitti360dDataset   # 完整的 KITTI-360 数据加载器
│   └── Kitti360Dataset    # 简化的数据加载器（预留）
├── models/               # 模型定义
│   ├── encoders/         # 卫星图编码器
│   │   ├── SatelliteConditionEncoder  # 卫星图条件编码器
│   │   └── RelativeCoordinateEncoder  # 相对坐标编码器
│   ├── unet/             # U-Net 模块
│   │   └── RelativePositionAttention  # 相对位置注意力机制
│   ├── sd_model.py       # 基础 SD 模型包装
│   └── sd_trainer.py     # 完整的训练器
├── utils/                # 工具函数
│   ├── geometry/         # 几何计算（来自原始项目）
│   │   ├── differentiable_projection.py  # 可微分投影
│   │   ├── kitti_transforms.py          # KITTI 变换
│   │   ├── pose_encoding.py             # 位姿编码
│   │   ├── bev_to_camera_warp.py        # BEV 到相机图
│   │   └── ...
│   └── pos_embed.py      # 位置编码
├── metrics/              # 评估指标
│   ├── psnr.py
│   ├── ssim.py
│   └── lpips.py
├── scripts/              # 训练和推理脚本
│   ├── train.py          # 主要训练脚本
│   ├── infer.py          # 推理脚本
│   └── train_sd.py       # 原始训练脚本（已保留）
├── requirements.txt      # 依赖包
├── README.md            # 项目说明
└── .gitignore           # Git 忽略文件
```

## 快速开始

### 1. 安装依赖

```bash
# 创建并激活虚拟环境（可选但推荐）
python -m venv venv
source venv/bin/activate  # Linux
# 或 .\venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

确保 KITTI-360 数据集位于 `/media/zhimiao/Lenovo/KITTI-360` 或更新配置文件 `configs/train.yaml` 和 `configs/inference.yaml` 中的路径。

### 3. 训练

```bash
# 使用默认配置（batch_size=2, 50 epochs）
python scripts/train.py

# 自定义配置（推荐）
python scripts/train.py --data_dir "/path/to/your/data" --batch_size 4 --epochs 100
```

**训练参数说明：**

- `--data_dir`: KITTI-360 数据根目录
- `--batch_size`: 每个 GPU 的批次大小（建议 1-4）
- `--epochs`: 训练轮数
- `--lr`: 学习率（默认 1e-4）
- `--warmup`: 预热轮数（默认 5）
- `--seed`: 随机种子（默认 42）
- `--device`: 使用的设备（"cuda" 或 "cpu"）

### 4. 推理

**单张图片推理：**

```bash
python scripts/infer.py --checkpoint "./output/checkpoints/checkpoint_epoch_50.pt" \
    --drive_dir "/media/zhimiao/Lenovo/KITTI-360/2013_05_28_drive_0003_sync" \
    --frame_id 0 \
    --output_dir "./inference_results"
```

**整个 drive 推理：**

```bash
python scripts/infer.py --checkpoint "./output/checkpoints/checkpoint_epoch_50.pt" \
    --drive_dir "/media/zhimiao/Lenovo/KITTI-360/2013_05_28_drive_0003_sync" \
    --output_dir "./inference_results"
```

**验证集推理：**

```bash
python scripts/infer.py --checkpoint "./output/checkpoints/checkpoint_epoch_50.pt" \
    --data_dir "/media/zhimiao/Lenovo/KITTI-360" \
    --output_dir "./inference_results"
```

## 关键组件详解

### SatelliteConditionEncoder

卫星图条件编码器，包含：

- **Patch 嵌入**：将 512x512 卫星图分割为 16x16 补丁
- **坐标编码**：为每个补丁添加相对位置编码
- **Transformer 层**：使用相对位置注意力保持空间一致性

```python
encoder = SatelliteConditionEncoder(
    embed_dim=768,       # 嵌入维度
    patch_size=16,       # 补丁大小
    num_layers=12,       # 层数
    num_heads=12,        # 注意力头数
    use_relative_pos=True  # 使用相对位置编码
)

# 输入：卫星图 (B, 3, 512, 512)
# 输出：特征嵌入 (B, 1024, 768)
embeddings = encoder(sat_images)
```

### SDTrainer

完整的训练器，包含：

- **学习率调度**：余弦衰减 + 预热
- **梯度累积**：支持小显存训练
- **Checkpoint 管理**：自动保存和恢复
- **混合精度**：支持 FP16 训练（预留）

## 训练策略

### 冻结策略

1. **基础冻结**：只训练卫星编码器和顶部 U-Net 块（默认）
2. **解冻策略**：训练初期固定基础层，后期可解冻部分中层
3. **微调策略**：所有层参与训练（适合数据充足时）

### 学习率

- **预热期**：前 5 轮使用较小学习率
- **稳定期**：使用余弦衰减到 1e-6
- **权重衰减**：1e-4，L2 正则化

## 性能预估

### 硬件要求

- **GPU 显存**：至少 12GB（Batch size 2），24GB 更好
- **训练时间**：50 轮，约 12-24 小时（取决于 GPU）
- **显存优化**：使用 `--gradient_accumulation` 和 `--batch_size=1`

### 评估指标

| 指标 | 期望 | 描述 |
|------|------|------|
| PSNR | 25-30 dB | 峰值信噪比（越高越好）|
| SSIM | 0.85-0.95 | 结构相似度（越高越好）|
| LPIPS | <0.1 | 感知损失（越低越好）|
| 几何一致性 | >0.95 | 边界匹配度（越高越好）|

## 常见问题

### 1. 数据加载失败

**Q**：`FileNotFoundError: calibration file not found`
**A**：确保 `--data_dir` 正确，且包含完整的 KITTI-360 目录结构。

### 2. 显存不足

**Q**：`CUDA out of memory`
**A**：减小 `--batch_size` 或增加 `--gradient_accumulation` 步数。

### 3. 训练不稳定

**Q**：Loss 波动大
**A**：调整学习率，增加 warmup 轮数，或检查数据质量。

## 未来优化方向

1. **多尺度 reading block 强化**：在更多 U-Net 层进行几何条件读取
2. **VQGAN 替换**：使用更轻量级的图像编码器
3. **多视图融合**：支持多卫星视角
4. **轻量化部署**：使用 ONNX 和 TensorRT 优化

## License

MIT
