# 代码需求文档 v1

## 项目名

`Satellite-to-Perspective Position-Conditioned Reading Module for Stable Diffusion`

## 文档目标

实现一个插入到 Stable Diffusion U-Net 中的“卫星到前视的位置条件读取模块”。

该模块不直接负责主去噪，而是：

- 从冻结的卫星 token 中读取与前视位置几何对齐的地图特征。
- 用 GeoRoPE 作为几何寻址机制。
- 将读取到的地图特征以门控残差方式注入当前 U-Net feature。

---

## 1. 当前版本范围

### 1.1 V1 范围

V1 仅实现最小可跑通方案，验证以下核心假设：

- 冻结卫星 encoder 提供的 token 能作为稳定地图记忆。
- 前视位置的归一化 ego-centric BEV 坐标可作为查询锚点。
- GeoRoPE 能让读取模块按几何位置从卫星 token 中软读取内容。
- 读取结果通过门控残差注入 U-Net 后，能改善前视生成。

### 1.2 V1 不做内容（延期到 V2）

- 几何锚点层间 refinement。
- depth distribution / uncertainty。
- ray 显式编码进入 GeoRoPE。
- polar 坐标主通路。
- 前视 latent feature 参与 query 构造。
- 卫星 token 更新。
- query-key-value 双向联合 attention。
- 卫星图像 grid 坐标辅通路。

---

## 2. 总体架构

### 2.1 模块定位

该模块是 U-Net 若干层中的条件读取器，不是主去噪器。

- U-Net 主干职责：正常处理 noisy latent。
- Reading 模块职责：
  - 给每个前视位置生成 query。
  - 从卫星 token 中按 GeoRoPE 寻址读取地图特征。
  - 将读取特征注入 U-Net 当前层。

### 2.2 数据流

输入：

- `front_feat`（当前 U-Net feature）
- `sat_tokens`（冻结遥感 encoder 输出）
- `front_bev_xy`（前视位置归一化 ego-centric BEV 锚点）
- `sat_xy`（卫星 token 归一化 ego-centric 坐标）

输出：

- `front_feat_out`

---

## 3. 坐标系统定义

### 3.1 统一坐标系

所有几何坐标使用 `ego-centric map frame`：

- 原点：卫星图中心（车辆位置）。
- 朝向：与目标前视相机 heading 对齐。
- 全部坐标先转换到该坐标系，再做归一化。

### 3.2 卫星 token 坐标

每个卫星 token 对应地图平面坐标：

$$
\mathbf{p}^{sat}_i = (x_i, y_i)
$$

归一化后：

$$
\hat{\mathbf{p}}^{sat}_i = (\hat{x}_i, \hat{y}_i)
$$

建议归一化：

$$
\hat{x}_i = x_i / R_x,\quad \hat{y}_i = y_i / R_y
$$

使坐标大致落在 `[-1, 1]`。

### 3.3 前视查询锚点

每个前视位置对应 BEV 锚点：

$$
\mathbf{p}^{front}_j = (x_j, y_j),\quad \hat{\mathbf{p}}^{front}_j = (\hat{x}_j, \hat{y}_j)
$$

要求：前视锚点与卫星坐标必须使用同原点、同方向、同尺度归一化。

---

## 4. 卫星特征分支

### 4.1 卫星 encoder

- 使用遥感数据集预训练模型提取卫星特征。
- 参数默认冻结。
- 输出多尺度卫星特征（V1 先接入一个尺度）。

### 4.2 卫星 token

输入卫星图后得到：

- `sat_tokens: [B, Ns, Cs]`
- `sat_xy: [B, Ns, 2]`

其中 `Ns` 为卫星 token 数，`Cs` 为通道维度。

### 4.3 Adapter

卫星 encoder 通道与 U-Net attention 通道可能不一致，需要可训练 adapter：

- 通道对齐。
- 可选 layer norm。

输出：

- `sat_feat: [B, Ns, Cmodel]`

---

## 5. 前视侧输入

### 5.1 当前 U-Net feature

- `front_feat: [B, Cf, H, W]`
- 展平后 `Nf = H * W`

### 5.2 前视位置锚点

- `front_bev_xy: [B, Nf, 2]`

V1 中 query 仅由该位置锚点生成，不使用前视内容 feature 构造 query。

---

## 6. Reading Block 定义

### 6.1 模块名称

建议类名：`SatelliteReadingBlock`。

### 6.2 模块职责

对每个前视位置：

1. 根据 `front_bev_xy` 生成 query。
2. 对卫星 token 生成 key/value。
3. 对 query/key 施加 continuous ego-centric xy GeoRoPE。
4. 用 attention 从卫星 token 读取内容。
5. 将读取特征回注到当前 U-Net feature。

---

## 7. Q / K / V 定义

### 7.1 Query

输入：`front_bev_xy: [B, Nf, 2]`

先用小 MLP 生成位置 embedding：

$$
\mathbf{e}^{front}_j = \mathrm{MLP}(\mathbf{front\_bev\_xy}_j)
$$

再投影为 query：

$$
\mathbf{Q}_j = W_q \mathbf{e}^{front}_j
$$

### 7.2 Key / Value

由卫星 token（adapter 后）生成：

$$
\mathbf{K}_i = W_k \mathbf{h}^{sat}_i,\quad \mathbf{V}_i = W_v \mathbf{h}^{sat}_i
$$

---

## 8. GeoRoPE 定义

### 8.1 V1 类型

V1 使用 `xy-separable continuous GeoRoPE`：

- `x` 通道做 1D rotary。
- `y` 通道做 1D rotary。
- 不做 polar。
- 不做 uncertainty-aware expected RoPE。

### 8.2 head 维度切分

设每个 head 维度为 `d_head`：

$$
d_{geo}=d_{head}//2,\quad d_{base}=d_{head}-d_{geo}
$$

再将 `d_geo` 平分：

$$
d_x=d_{geo}//2,\quad d_y=d_{geo}-d_x
$$

每个 head：

$$
Q=[Q_{base}\|Q_x\|Q_y],\quad K=[K_{base}\|K_x\|K_y]
$$

其中 `Q_x/K_x` 用 `x` 旋转，`Q_y/K_y` 用 `y` 旋转，`base` 不旋转。

### 8.3 旋转角

对归一化坐标 `\hat{x}, \hat{y}`：

$$
\theta^x_m = \hat{x}\cdot \omega_m,\quad \theta^y_m = \hat{y}\cdot \omega_m
$$

频率：

$$
\omega_m = base^{-2m/d}
$$

默认：`rope_base = 10000.0`。

### 8.4 应用位置

GeoRoPE 仅作用于 `Query` 和 `Key`，不作用于 `Value`。

输出：$\tilde{Q}, \tilde{K}$ 用于 attention score 计算。

---

## 9. Geometry Bias

### 9.1 V1 启用

V1 添加弱几何 bias 到 attention logits。

### 9.2 定义

$$
B^{geom}_{ji} = -\lambda_{geo}\left[(\hat{x}_j-\hat{x}_i)^2+(\hat{y}_j-\hat{y}_i)^2\right]
$$

### 9.3 作用

$$
S_{ji}=\frac{\tilde{Q}_j\tilde{K}_i^\top}{\sqrt{d_{head}}}+B^{geom}_{ji}
$$

### 9.4 超参

建议初值：`lambda_geo = 1.0`。

---

## 10. Attention 读取

权重：

$$
A_{ji}=\mathrm{softmax}_i(S_{ji})
$$

读取结果：

$$
M^{read}_j=\sum_i A_{ji}V_i
$$

得到：

- `read_tokens: [B, Nf, Cmodel]`
- reshape 后 `read_feat: [B, Cread, H, W]`

---

## 11. 回注到 U-Net

### 11.1 策略

V1 使用门控残差注入（`gated residual injection`）。

### 11.2 公式

先投影：

$$
\hat{F}_{read}=W_o(F_{read})
$$

由当前 `front_feat` 生成 gate：

$$
\gamma=\sigma(\mathrm{MLP}(\mathrm{GAP}(F_{in}^{fv})))
$$

输出：

$$
F_{out}^{fv}=F_{in}^{fv}+\gamma\odot\hat{F}_{read}
$$

其中 `gamma` 为每通道 gate，`shape: [B, C, 1, 1]`。

### 11.3 gate 生成方式

流程：`GAP -> bottleneck MLP -> sigmoid`

建议：

- `Linear(C, C//4)`
- `SiLU`
- `Linear(C//4, C)`

### 11.4 初始化

降低早期扰动：

- gate MLP 最后一层 `weight = 0`
- gate MLP 最后一层 `bias = -2`

使初始 gate 约 `0.12`。

---

## 12. 模块接口定义

### 12.1 `forward` 输入

```python
forward(
    front_feat,      # [B, Cf, H, W]
    sat_tokens,      # [B, Ns, Cs]
    sat_xy,          # [B, Ns, 2]
    front_bev_xy,    # [B, Nf, 2]
)
```

### 12.2 `forward` 输出

```python
return {
    "front_feat_out": front_feat_out,   # [B, Cf, H, W]
    "read_feat": read_feat,             # [B, Cf, H, W]
    "attn_map": attn_map,               # optional, [B, heads, Nf, Ns]
}
```

---

## 13. 模块内部 forward 流程

1. 对 `front_bev_xy` 做 position MLP，得到 query embedding。
2. 对 query embedding 做 `q_proj`。
3. 对 `sat_tokens` 做 adapter，再做 `k_proj / v_proj`。
4. reshape 成 multi-head tensor。
5. 对 `Q/K` 应用 continuous xy GeoRoPE。
6. 计算 pairwise geometry bias。
7. 计算 attention logits、softmax、读取结果。
8. 将读取结果投影回 `[B, Cf, H, W]`。
9. 根据当前 `front_feat` 生成 channel gate。
10. 执行 gated residual injection，输出 `front_feat_out`。

---

## 14. 建议代码模块拆分

### 14.1 `ContinuousXYGeoRoPE`

职责：对 multi-head `Q/K` 应用 xy-separable continuous rotary。

接口：

```python
q_tilde, k_tilde = rope(q, k, q_xy, k_xy)
```

### 14.2 `SatelliteReadingAttention`

职责：构造 `Q/K/V`、做 GeoRoPE、做 geometry bias、输出 `read_tokens`。

### 14.3 `GatedResidualInject`

职责：`read_feat -> out_proj`，`front_feat -> gate`，`front_out = front + gate * read`。

### 14.4 `SatelliteReadingBlock`

职责：封装 reading attention + gated injection，对外暴露统一 `forward`。

---

## 15. U-Net 集成方式

### 15.1 插入位置

V1 建议：`middle block` + `decoder` 前两层（共 2~3 个 block）。

### 15.2 输入准备

每个 block 需要对应尺度的：

- `front_feat`
- `front_bev_xy`
- `sat_tokens`
- `sat_xy`

其中 `front_bev_xy` 需按当前 feature resolution 对齐到 `Nf = H*W`。

### 15.3 与 U-Net 主干关系

- 主干照常执行去噪。
- reading block 仅提供条件增强，不替代主干 attention。

---

## 16. 关键配置项

- `num_heads`
- `head_dim`
- `geo_ratio = 0.5`
- `rope_base = 10000.0`
- `lambda_geo = 1.0`
- `gate_hidden_ratio = 0.25`
- `use_geom_bias = True`
- `use_gated_residual = True`

---

## 17. 非目标声明

V1 不负责：

- 前视锚点从 ray/depth 分布迭代 refinement。
- 卫星 token 是否需要 LoRA / finetune。
- polar 坐标是否优于 xy。
- uncertainty-aware attention。
- 多尺度联合卫星读取。
- 前视内容参与 query 是否更优。

以上留作 V2 / ablation。

---

## 18. V1 成功标准

- 模块可在 U-Net 中正常前向。
- attention 权重 shape 正确。
- GeoRoPE 对输入坐标敏感。
- 读取特征可稳定注回 U-Net。
- 训练初期不会因读取模块导致 loss 爆炸。
- 可导出 attention map 做可视化。

---

## 19. 一句话总结

V1 实现的是：以归一化 ego-centric `xy` 坐标为几何锚点的卫星条件读取模块。模块使用 continuous `xy` GeoRoPE 在 `query/key` 上建立前视位置与卫星位置之间的几何寻址关系，并将读取到的地图特征通过门控残差注入 Stable Diffusion U-Net。

---

## 20. 待澄清项（按优先级）

### 20.1 P0（不明确会阻塞实现）

| 编号 | 不明确点 | 影响 | 建议默认决策（如暂不回复） |
|---|---|---|---|
| P0-1 | `front_bev_xy` 由谁提供、如何生成（射线投影/查表/外部模块） | 无法确定 query 几何语义，模块输入不可构造 | 由上游数据管线直接提供 `front_bev_xy`，本模块不内生生成 |
| P0-2 | `front_bev_xy` 与 `sat_xy` 的坐标约定细节（轴向、单位、是否 x 前 y 左） | 坐标系统不一致会导致注意力对齐失败 | 强制同一约定：ego-centric，单位米，归一化后 `[-1,1]` |
| P0-3 | `sat_xy` 与 `sat_tokens` 的映射规则（patch 中心点？插值点？） | K 的几何位置不确定，GeoRoPE/Bias 失真 | 使用 token 对应 patch center 作为 `sat_xy` |
| P0-4 | 通道维关系 `Cmodel/Cread/Cf` 最终约束 | 影响 `out_proj`、残差注入与 shape 正确性 | 统一为 `Cread = Cmodel`，注入前 `out_proj: Cread -> Cf` |
| P0-5 | `head_dim` 的合法约束（GeoRoPE 切分后的偶数要求） | rotary 可能无法成对旋转，运行报错 | 限制 `d_x`、`d_y` 均为偶数；不满足则 raise config error |
| P0-6 | 插入 2~3 个 U-Net 层时 `sat_tokens/sat_xy` 是否共享 | 影响显存、算力和实现复杂度 | V1 默认共享同一组卫星 token/坐标 |

### 20.2 P1（不阻塞首版，但影响训练质量/复现）

| 编号 | 不明确点 | 影响 | 建议默认决策（如暂不回复） |
|---|---|---|---|
| P1-1 | `lambda_geo` 是固定常数、按层配置还是可训练 | 不同层几何先验强度不可控 | V1 设为按层可配常数（默认 `1.0`） |
| P1-2 | `attn_map` 导出策略（训练/推理、开关控制） | 显存占用和日志开销不可控 | 默认 `return_attn=False`，仅调试时导出 |
| P1-3 | 卫星 encoder“冻结”的严格定义 | BN/Dropout 行为不一致，影响稳定性 | 参数 `requires_grad=False` + encoder `eval()` |
| P1-4 | `Nf x Ns` 复杂度上限与降采样策略 | 大分辨率可能 OOM | V1 默认全局 attention；超阈值时下采样 `Ns` |
| P1-5 | 初始化策略是否对所有 block 一致 | 不同层训练动态不一致 | 所有 reading block 统一 gate 初始化（`w=0,b=-2`） |

### 20.3 P2（评估与验收口径）

| 编号 | 不明确点 | 影响 | 建议默认决策（如暂不回复） |
|---|---|---|---|
| P2-1 | “改善前视生成”的评价指标未定义 | 无法判断 V1 是否达标 | 先用 FID/LPIPS + 定性可视化作为阶段性验证 |
| P2-2 | 训练稳定性判定阈值未定义 | “不爆炸”缺少量化标准 | 以 loss 无 NaN/Inf，且 5k step 内曲线平稳为准 |
| P2-3 | attention 可解释性验收标准未定义 | 难统一可视化结论 | 检查坐标扰动前后 `attn_map` 热区可迁移 |

## 21. 建议你确认的最小决策集（6项）

1. `front_bev_xy` 由上游输入，本模块不负责生成。  
2. 坐标统一采用 ego-centric、单位米、同原点同朝向、归一化到 `[-1,1]`。  
3. `sat_xy` 使用卫星 token 对应 patch 中心点。  
4. `Cread = Cmodel`，并通过 `out_proj` 映射到 `Cf` 后注入。  
5. 多层 reading block 共享同一组 `sat_tokens/sat_xy`。  
6. `lambda_geo` 采用“按层可配常数”（默认 `1.0`），非可训练参数。  

如果以上 6 项默认通过，就可以直接进入 V1 编码实现，剩余 P1/P2 作为训练与评估阶段继续细化。

---

## 22. 已确认决策（用户确认）

以下 6 项已确认，作为 V1 实现基线：

1. `front_bev_xy` 由上游输入，本模块不负责生成。  
2. 坐标统一采用 ego-centric、单位米、同原点同朝向、归一化到 `[-1,1]`。  
3. `sat_xy` 使用卫星 token 对应 patch 中心点。  
4. `Cread = Cmodel`，并通过 `out_proj` 映射到 `Cf` 后注入。  
5. 多层 reading block 共享同一组 `sat_tokens/sat_xy`。  
6. `lambda_geo` 采用“按层可配常数”（默认 `1.0`），非可训练参数。  

### 22.1 数据与坐标补充约束（已确认）

1. 卫星图输入尺寸固定为 `512 x 512`。  
2. 卫星图分辨率为 `0.2 m/pixel`。  
3. 透视图在 dataloader 中固定为 `640 x 256`。  
4. `sat_xy` 使用 token 对应 patch center（中心点坐标）构造。  
5. `front_bev_xy` 与 `sat_xy` 均采用同一 ego-centric 归一化约定：以卫星图像素中点为原点，scale 到 `[-1, 1]`。  

### 22.2 可直接落地的 `sat_xy` 归一化公式

设卫星图宽高为 `W=H=512`，patch 大小为 `P`，token 网格索引为 `(r, c)`：

- patch center 像素坐标：
  - `u = (c + 0.5) * P`
  - `v = (r + 0.5) * P`
- 归一化坐标（推荐，y 轴向上为正）：
  - `x_hat = (u - W/2) / (W/2)`
  - `y_hat = (H/2 - v) / (H/2)`

该定义与文档中的 ego-centric 约定一致，且可直接用于 GeoRoPE 与 geometry bias。

---

## 23. V1 文件级实现任务拆解（可直接开工）

### 23.1 新增模块文件

1. `models/unet/continuous_xy_georope.py`
   - 类：`ContinuousXYGeoRoPE`
   - 输入：`q, k, q_xy, k_xy`
   - 输出：`q_tilde, k_tilde`
   - 关键点：`x/y` 分支 rotary，`base` 分支直通。

2. `models/unet/satellite_reading_attention.py`
   - 类：`SatelliteReadingAttention`
   - 职责：`position_mlp + q_proj + sat_adapter + k/v_proj + GeoRoPE + geom_bias + softmax read`
   - 输出：`read_tokens`，可选 `attn_map`。

3. `models/unet/gated_residual_inject.py`
   - 类：`GatedResidualInject`
   - 职责：`read_tokens -> read_feat -> out_proj`，`front_feat -> gate`，做残差注入。
   - 初始化：gate 最后一层 `weight=0, bias=-2`。

4. `models/unet/satellite_reading_block.py`
   - 类：`SatelliteReadingBlock`
   - 职责：封装 reading attention + gated injection，对外统一 `forward`。

### 23.2 修改现有文件

1. `models/unet/__init__.py`
   - 导出新增类：`ContinuousXYGeoRoPE`、`SatelliteReadingAttention`、`GatedResidualInject`、`SatelliteReadingBlock`。

2. `models/sd_model.py`
   - 在 `SatelliteConditionedUNet` 中引入 reading block 注册与调用入口。
   - 保持兼容：若未提供 `front_bev_xy/sat_xy`，可跳过 reading block（回退原行为）。

3. `models/sd_trainer.py`
   - 训练前向增加 `sat_xy` 与多尺度 `front_bev_xy` 的输入传递。
   - 增加 `return_attn_map` 调试开关（默认关闭）。

4. `configs/train.yaml`
   - 新增 `reading_block` 配置：`num_heads/head_dim/geo_ratio/rope_base/lambda_geo/gate_hidden_ratio/use_geom_bias/use_gated_residual`。
   - 新增插入层配置：`injection_sites`（如 `mid, up0, up1`）。

5. `configs/inference.yaml`
   - 对齐推理阶段 `reading_block` 配置与可视化开关。

### 23.3 接口与张量约束（实现时强校验）

- `front_feat: [B, Cf, H, W]`
- `sat_tokens: [B, Ns, Cs]`
- `sat_xy: [B, Ns, 2]`
- `front_bev_xy: [B, Nf, 2]` 且 `Nf = H * W`
- `read_tokens: [B, Nf, Cmodel]`
- `read_feat/read_proj: [B, Cf, H, W]`
- `attn_map(optional): [B, heads, Nf, Ns]`

补充固定尺寸约束：

- 卫星图输入默认 `512 x 512`，对应物理覆盖约 `102.4m x 102.4m`。
- 前视图输入默认 `640 x 256`（dataloader 统一处理）。

建议在 `forward` 开始处做 `assert`/`shape check`，尽早报错。

### 23.4 建议实现顺序（最小可运行路径）

1. 先实现 `ContinuousXYGeoRoPE` 与单元 shape 验证。
2. 实现 `SatelliteReadingAttention`，跑通 `read_tokens` 输出。
3. 实现 `GatedResidualInject`，验证 gate 初始化与残差形状。
4. 组装 `SatelliteReadingBlock`，完成独立模块前向。
5. 接入 `SatelliteConditionedUNet` 指定层。
6. 最后对齐 trainer/config，并打开 `attn_map` 调试导出。

### 23.5 验收检查单（编码后立即可测）

- 前向可跑通且无 shape 错误。
- `GeoRoPE` 对 `q_xy/k_xy` 扰动敏感。
- `use_geom_bias` 开关可控且 logits 行为符合预期。
- gate 初值约 `sigmoid(-2)≈0.12`。
- reading block 关闭时回退到基线行为。
