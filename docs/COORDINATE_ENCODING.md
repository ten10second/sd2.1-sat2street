# 透视 token 位置编码的实现原理

本文档详细解释了"透视 token 在 BEV 图上的相对位置"如何作为卫星图特征的额外位置编码，以及它们在 self-attention 中如何融合和关联。

## 核心概念

### 问题
如何在不使用 IPM（逆透视映射）的情况下，让卫星图特征保持空间一致性？

### 解决方案
> **给卫星图加入另一个位置编码——透视平面上每个 token 在 BEV 图上的相对位置，然后做 self-attention，最后通过 Stable Diffusion 输出。**

---

## 实现原理

### 1. BEV 空间坐标 vs 图像网格坐标

卫星图有两种坐标系统：

#### 图像网格坐标（传统方法）
```
卫星图像 (512x512)
┌─────────────────┐
│ (0,0)  (1,0)    │ ← 图像像素坐标
│ (0,1)  (1,1)    │   - 只表示在图片上的位置
│ ...              │   - 没有物理意义
└─────────────────┘
```

#### BEV 空间坐标（我们的方法）
```
BEV 空间 (以米为单位)
┌─────────────────┐
│ (-51.2, +51.2)  │ ← BEV 物理坐标
│ (-51.2, +51.0)  │   - 表示在真实世界中的位置
│ ...              │   - 卫星图中心在 (0, 0)
└─────────────────┘
     ↑
     车的位置 (IMU)
```

**关键区别**：
- BEV 空间坐标是**固定的物理位置**
- 两个 patch 在 BEV 空间中距离近，在真实世界中也近
- 这样 self-attention 可以利用真实的空间关系

---

### 2. 坐标编码流程

```
输入: 卫星图像 (B, 3, 512, 512)
    ↓
[步骤 1] Patch Embedding
    - 将 512x512 分割成 16x16 的 patch
    - 共 32x32 = 1024 个 patches
    - 输出: (B, 1024, 768)
    ↓
[步骤 2] BEV 坐标计算
    - 为每个 patch 计算它在 BEV 空间的物理坐标 (x, y)
    - 单位：米
    - 输出: (B, 1024, 2)
    ↓
[步骤 3] 坐标编码
    - 将 BEV 坐标编码为 embedding
    - 使用 MLP: Linear → LayerNorm → GELU → Linear → LayerNorm
    - 输出: (B, 1024, 768)
    ↓
[步骤 4] 特征融合
    - Patch embedding + 坐标 embedding
    - x = patches_flat + coord_emb
    - 输出: (B, 1024, 768)
    ↓
[步骤 5] 相对位置 Self-Attention
    - 使用 BEV 坐标计算相对位置
    - 对于每对 patch (i, j):
      relative_pos = (x_i - x_j, y_i - y_j)
    - 将相对位置编码为 bias，加到 attention weights 上
    - 输出: (B, 1024, 768)
    ↓
[步骤 6] 最终 LayerNorm
    - 输出: (B, 1024, 768)
```

---

### 3. 相对位置 Self-Attention 的工作原理

#### 标准 Self-Attention
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V
```

#### 我们的相对位置 Self-Attention
```
1. 计算 Q, K, V  (和标准一样)

2. 计算 BEV 相对位置
   对于每个 patch 对 (i, j):
   Δx = x_i - x_j  (BEV x 坐标差，单位：米)
   Δy = y_i - y_j  (BEV y 坐标差，单位：米)
   relative_pos_ij = (Δx, Δy)

3. 编码相对位置
   relative_bias_ij = MLP(relative_pos_ij)

4. 加到 attention weights 上
   Attention(Q, K, V, relative_bias) = softmax( (QK^T + relative_bias) / sqrt(d) ) V
```

**为什么这样做？**

- 如果两个 patch 在 BEV 空间很近，它们的相对位置 bias 会让 attention 更强
- 这样保持了真实物理世界的空间关系
- 不管卫星图怎么旋转或裁剪，BEV 空间坐标是固定的

---

### 4. 关键代码详解

#### SatelliteConditionEncoder._compute_patch_bev_coords()

```python
def _compute_patch_bev_coords(self, B: int, H: int, W: int) -> torch.Tensor:
    """
    计算每个卫星 patch 的 BEV 空间坐标（米）。
    这是关键：每个卫星 patch 在 BEV 空间有固定的物理位置，
    而不只是图像上的网格位置。
    """
    patch_h = H // self.patch_size  # 32
    patch_w = W // self.patch_size  # 32

    # 计算 patch 中心的像素坐标
    patch_pixel_h = torch.arange(patch_h) * self.patch_size + self.patch_size / 2
    patch_pixel_w = torch.arange(patch_w) * self.patch_size + self.patch_size / 2

    # 创建网格
    w_grid, h_grid = torch.meshgrid(patch_pixel_w, patch_pixel_h, indexing='xy')

    # 转换为 BEV 空间坐标（米）
    # 卫星图中心在 (0, 0)
    # x 轴向东为正，y 轴向北为正
    half_sat = self.sat_size / 2.0  # 256 像素
    x_meters = (w_grid - half_sat) * self.sat_resolution  # (w_grid - 256) * 0.2
    y_meters = (half_sat - h_grid) * self.sat_resolution  # (256 - h_grid) * 0.2

    return coords  # (B, 1024, 2)
```

#### RelativePositionAttention._compute_relative_positions()

```python
def _compute_relative_positions(self, coords: torch.Tensor) -> torch.Tensor:
    """
    计算所有 patch 对在 BEV 空间中的相对位置。
    对于每对 patch (i, j)，计算 (x_i - x_j, y_i - y_j)
    其中 x, y 是 BEV 坐标（米）。
    """
    # coords: (B, N, 2)
    B, N, _ = coords.shape
    # (B, N, 1, 2) - (B, 1, N, 2) -> (B, N, N, 2)
    relative_pos = coords.unsqueeze(2) - coords.unsqueeze(1)
    return relative_pos
```

---

### 5. 完整的数据流示例

假设我们有：
- 卫星图: (1, 3, 512, 512)
- Patch size: 16
- Embedding dim: 768

```
步骤 1: Patch Embedding
输入: (1, 3, 512, 512)
输出: (1, 1024, 768)

步骤 2: BEV 坐标计算
输入: patch 位置
输出: (1, 1024, 2)
  - patch 0: (-51.2, +51.2) 米 (左上角)
  - patch 1: (-48.0, +51.2) 米
  - ...
  - patch 511: (+51.2, +51.2) 米 (右上角)
  - ...
  - patch 1023: (+51.2, -51.2) 米 (右下角)

步骤 3: 坐标编码
输入: (1, 1024, 2)
输出: (1, 1024, 768)

步骤 4: 特征融合
x = patch_emb + coord_emb
输出: (1, 1024, 768)

步骤 5: 相对位置 Self-Attention
对于每个 head:
  - Q, K, V: (1, 12, 1024, 64)
  - Attention weights: (1, 12, 1024, 1024)
  - 加上相对位置 bias
  - 输出: (1, 1024, 768)

步骤 6: 最终输出
输出: (1, 1024, 768) 用于 Stable Diffusion 的条件
```

---

### 6. 为什么这样做有效？

#### 优势 1: 空间一致性
```
BEV 空间
┌─────────────────┐
│  A        B     │  A 和 B 在 BEV 空间距离近
│                 │  → 它们之间的 attention 会强
│        C        │
└─────────────────┘

标准 Attention (只看内容):
- A 可能关注 C 而不是 B

我们的方法 (看 BEV 相对位置):
- A 会更关注 B，因为它们在物理空间更近
```

#### 优势 2: 视角无关性
```
不管卫星图怎么旋转或裁剪，
BEV 空间坐标是固定的！

原始卫星图: A 在左上角
裁剪后的卫星图: A 在中间

对于 A 来说，它的 BEV 坐标不变，
它和其他 patch 的相对关系也不变！
```

#### 优势 3: 不需要 IPM
```
传统方法:
卫星图 → IPM → 前视图
  (几何投影)

我们的方法:
卫星图 + 相机位姿 → Stable Diffusion → 前视图
  (BEV 位置编码保持空间一致性)
```

---

### 7. 总结

**核心思想的实现**：

1. **卫星图的每个 patch 有固定的 BEV 空间坐标**
   - 不是图像网格坐标，而是真实物理坐标（米）

2. **将 BEV 坐标编码为位置 embedding**
   - 加到 patch embedding 上

3. **Self-attention 使用 BEV 坐标计算相对位置**
   - 让 attention 尊重真实的空间关系
   - 保持空间一致性

**这样就实现了**：
"给卫星图加入另一个位置编码——透视平面上每个 token 在 BEV 图上的相对位置，然后做 self-attention，最后通过 Stable Diffusion 输出。"
