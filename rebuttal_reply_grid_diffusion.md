# 返修回复：Grid Diffusion 相关意见

> **审稿意见原文**：相比已有 Grid Diffusion 方法，本文技术改进没有单独验证。稿件没有定义图块分辨率、最终网格分辨率、填充方式、帧排序、边界处理，以及网格滑动在反演和去噪阶段的具体执行方式。提供可执行伪代码，设置对应的 Grid Diffusion 基线；在完整测试集上量化网格滑动、网格尺寸带来的效果。

## 总体回复

感谢审稿人的意见。我们按四个方面完成修改：

1. **补充定义**：在修改稿第 III-D 节新增表 R1，明确定义图块分辨率、最终网格分辨率、帧排序、填充方式、边界处理；
2. **补充伪代码**：新增算法 1–3（可直接执行的 Python 风格伪代码），完整给出网格拼装、布局交替（滑动）算子、以及在反演与去噪循环中的执行方式；
3. **补充 Grid Diffusion 基线**：代表性 Grid Diffusion 方法 RAVE [16] 已在原稿表 II 中定量对比；为进一步隔离本文改进的各自贡献，我们新增"去网格滑动的 Vanilla Grid Diffusion"作为受控基线（即固定网格布局、仅保留网格内自注意力的网格扩散），并给出组件级消融；
4. **补充定量消融**：在完整测试集（UELVE-LD，与表 II 相同的视频与帧数设置 8/16/48/200 帧）上，新增表 R2（网格滑动开/关）与表 R3（网格尺寸 2×1/2×2/2×3/2×4/3×4）的定量结果，指标与表 II 一致（Masked-PSNR、SSIM、Text Alignment、Tem-Con），并保留原稿图 9 的耗时分析、补充各网格尺寸下的质量指标。

---

## 逐点回复

### 1. 网格表示的完整定义（新增表 R1，写入修改稿 III-D 节）

**表 R1：网格表示实现细节**

| 项目 | 定义 |
|---|---|
| 图块（tile）分辨率 | 每帧 512×512 像素：先短边等比缩放至 512（bilinear + antialias），再中心裁剪；对应 latent 分辨率 64×64 |
| 网格形状 n×l | n 为每列帧数、l 为每行帧数；正文实验取 2×2（n=l=2），消融覆盖 2×1、2×3、2×4、3×4 |
| 最终网格分辨率 | (512n)×(512l) 像素，latent (64n)×(64l)；2×2 网格即 1024×1024 像素 / 128×128 latent（16,384 个空间 token）。拼格为无损像素块拷贝，无任何插值，单帧有效分辨率恒为 512×512 |
| 帧排序 | 帧按时间顺序以行主序（光栅序）填充网格：第 t 帧置于第 ⌊t/l⌋ 行、第 t mod l 列；即 f0→(行0,列0)，f1→(行0,列1)，f2→(行1,列0)…… |
| 帧数约束 | 视频帧数 K 需为网格容量 n·l 的整数倍；不满足时以末帧重复补齐（padding）、解码后丢弃补齐帧 |
| 填充方式 | 布局交替产生的边界网格：以该网格自身最后一列帧重复补齐；恢复 GridLayout1 时删除填充列（多卡场景改为相邻 GPU 交换边界 n 帧，见算法 3） |
| 边界处理 | 空间边界：中心裁剪，图块间为硬像素边界，自注意力在整张网格上运算、不做 tile 级掩码；网格间边界：奇偶步布局交替实现跨网格交互；视频首尾：按上述填充/交换规则处理 |
| 滑动的阶段适用 | 注意力控制路线（UELVE-Attention）：反演与去噪两阶段均执行布局交替；ControlNet 路线（UELVE-ControlNet）：仅在去噪阶段执行（原稿脚注 1） |

### 2. 可执行伪代码（新增算法 1–3，写入修改稿 III-D 节）

**算法 1：网格拼装与还原（无损，无插值）**

```python
def combine_frames_in_grid(V, n, l):
    """V: [B, C, K, h, w] 帧序列latent; 返回 [B, C, K/(n*l), n*h, l*w] 网格序列"""
    assert K % (n * l) == 0            # 帧数须为网格容量整数倍, 否则先按表R1补齐
    G = zeros(B, C, K // (n * l), n * h, l * w)
    t = 0
    for g in range(K // (n * l)):      # 网格序号
        for i in range(n):             # 网格第 i 行
            for j in range(l):         # 网格第 j 列（行主序填充）
                G[:, :, g, i*h:(i+1)*h, j*w:(j+1)*w] = V[:, :, t]
                t += 1
    return G                            # restore_frames_from_grid 为其精确逆运算
```

**算法 2：网格布局交替（Grid Layout Alternation，即"网格滑动"）**

```python
def alternate(latent_grids, step, n, l):
    """每个扩散时间步调用一次; step 为当前时间步索引"""
    frames = restore_frames_from_grid(latent_grids, n, l)   # 还原为帧序列 [B,C,K,h,w]
    if K == n * l:                       # 单网格: 无跨网格交互需求, 恒等返回
        return latent_grids, identity(K)
    rot = list(range(n, K)) + list(range(n))   # 循环旋转一个网格列(n 帧)
    if step % 2 == 1:                    # 奇数步 -> GridLayout2 (滑动一格)
        perm = rot
    else:                                # 偶数步 -> 恢复 GridLayout1 (旋回原布局)
        perm = argsort(rot)
    return combine_frames_in_grid(frames[:, :, perm], n, l), perm
```

**算法 3：反演与去噪中的执行方式（含多卡边界交换）**

```python
# ---- 反演阶段 (UELVE-Attention 路线): x -> z_hat_T ----
z = VAE_encode(x)                                  # [B,C,K,64,64]
for i, t in enumerate(timesteps):                  # t: 0 -> T
    z, perm = alternate(z, i, n, l)                # 每个时间步先做布局交替
    eps = UNet(z, t, C_src)                        # 网格沿批次维串行过 UNet
    z = ddim_inversion_step(eps, t, z)             # 式(6)

# ---- 去噪阶段: 两条路线均在此执行滑动 ----
z_T = z_hat_T (注意力控制路线) 或 N(0,I) (ControlNet 路线)
for i, t in enumerate(timesteps):                  # t: T -> 0
    z, perm = alternate(z, i, n, l)
    eps = UNet(z, t, C_tgt)                        # 可插入注意力控制 / ControlNet 条件
    z = ddim_denoise_step(eps, t, z)               # 式(5)
x_edited = VAE_decode(restore_frames_from_grid(z, n, l))

# ---- 多卡变体: 仅将 rot 的回绕部分替换为相邻 GPU 边界交换 ----
# 每个交替步, 各 rank 仅与本视频分段相邻的 rank 交换 n 帧边界 latent
# (单次通信量 n*h*w*C, 与 GPU 数量无关), 其余计算完全独立、无同步。
```

### 3. Grid Diffusion 基线设置

- 已有 Grid Diffusion 方法的代表 RAVE [16] 已在原稿表 II 中与本文在全部四种帧数设置下定量对比（Masked-PSNR / SSIM / Text Alignment / Tem-Con 及运行时间）。
- 为满足"单独验证本文技术改进"的要求，我们以**去滑动的 Vanilla Grid Diffusion**（固定 GridLayout1、仅网格内自注意力，无跨网格交互，即经典网格扩散设定）为受控基线，逐项叠加本文组件得到组件级消融（表 R2）：Vanilla Grid → + 布局交替（本文）→ + 网格化微调与时序注意力 → + G2F/F2G 掩码修正（完整 UELVE）。原稿图 7、10、12、13 已给出各组件的定性消融，修改稿补充定量结果。

### 4. 完整测试集上的定量消融（新增表 R2、R3）

**表 R2：网格布局交替（滑动）消融，完整测试集（UELVE-LD），与表 II 同指标**

| 设置 | Masked-PSNR↑ | SSIM↑ | Text Alignment↑ | Tem-Con↑ |
|---|---|---|---|---|
| | 8/16/48/200f | 8/16/48/200f | 8/16/48/200f | 8/16/48/200f |
| Vanilla Grid（无滑动） | 【待填】 | 【待填】 | 【待填】 | 【待填】 |
| + 布局交替（本文） | 【待填】 | 【待填】 | 【待填】 | 【待填】 |
| + 网格化微调+时序注意力 | 【待填】 | 【待填】 | 【待填】 | 【待填】 |
| 完整 UELVE | 表 II 数值 | 表 II 数值 | 表 II 数值 | 表 II 数值 |

**表 R3：网格尺寸消融，完整测试集 + 48 帧耗时（与图 9 同设置）**

| 网格形状（每格帧数） | Masked-PSNR↑ | SSIM↑ | Text Alignment↑ | Tem-Con↑ | 采样耗时(s)↓ |
|---|---|---|---|---|---|
| 2×1（2） | 【待填】 | 【待填】 | 【待填】 | 【待填】 | 图 9 已有 |
| 2×2（4） | 【待填】 | 【待填】 | 【待填】 | 【待填】 | 图 9 已有 |
| 2×3（6） | 【待填】 | 【待填】 | 【待填】 | 【待填】 | 图 9 已有 |
| 2×4（8） | 【待填】 | 【待填】 | 【待填】 | 【待填】 | 图 9 已有 |
| 3×4（12） | 【待填】 | 【待填】 | 【待填】 | 【待填】 | 图 9 已有 |

预期结论（与原稿定性分析一致，需以补跑数据核实）：布局交替主要提升 Tem-Con 与跨网格结构一致性，对 Text Alignment 影响较小；网格尺寸增大提升全局一致性但粗化细节（原稿图 8），且 2×3 为质量-效率的局部最优（原稿图 9）。

---

## 给作者本人的执行清单（不放入回复信）

1. **先解决 padding 表述与实现不一致**（二选一）：
   - 方案 A（推荐，改动最小）：把论文 Eq.(8) 附近"以重复帧补齐"改为"循环旋转一列（单卡）/ 相邻 GPU 交换边界 n 帧（多卡）"，并相应改算法 2 的文字说明；表 R1 的"填充方式"行改为循环旋转。
   - 方案 B：改代码实现 padding（`grid.py` 的 `combine_frames_in_grid` 支持补齐、`restore` 支持裁剪），论文描述不动。方案 B 同时解锁第 3 条的 200 帧网格尺寸消融。
2. **加消融开关**：`p2p_ddim_spatial_temp.py` 四处调用点（反演 199–206 行、去噪 477–484 行）包一层 `if self.use_grid_sliding:`，flag 从 `editing_config` 读入；关闭时 `indices=org` 保持 GridLayout1。单网格情形（K=n·l）代码已自动跳过滑动，无需处理。
3. **帧数整除问题**：200 帧视频在 2×3（容量 6）、2×4（容量 8）、3×4（容量 12）下不满足 `K % (n·l) == 0` 的断言（grid.py:5）。表 R3 的 200f 列需要方案 B 的 padding，或将该列裁到 198/196 帧（需在回复中注明）。
4. **实验矩阵**：滑动 {开, 关} × 帧数 {8,16,48,200} × 完整测试集；网格形状 {2×1,2×2,2×3,2×4,3×4}（同帧数设置）。指标计算与表 II 管线完全一致。
5. **ControlNet 路线核实**：论文脚注 1 称 ControlNet 路线仅去噪阶段滑动，但 `ddim_clean2noisy_loop` 里 method=="ControlNet" 分支也会对 control_image 做 alternate（201–202 行）——确认该反演循环在 ControlNet 路线是否真的被执行，回复信第 1 点的口径要与事实一致。
