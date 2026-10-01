# 返修回复：Eqs. (7)–(8) 清晰性（索引 / 符号 P / 非整除处理）

> **审稿意见原文（意见 3）**：Eqs. (7)-(8) do not make the indexing, the symbol P, or the treatment of a video whose length is not divisible by the grid size sufficiently clear. State what happens to padding and boundary frames, and show one concrete example. A short pseudocode block would help more than another conceptual diagram. Quantitative results for 1x1, 2x2, and 3x3 grids would also support the choice of the default 2x2 layout.

## 总体回复

感谢审稿人指出公式 (7)–(8) 的表述问题。我们承认原稿记号确实不严谨：上标帧区间与下标网格编号混用、符号 $P$ 未定义、非整除情形未说明。我们将在修改稿第 3.4 节做四处修改：

1. **重写 Eqs. (7)–(8)**：删除未定义的 $P$，改用带显式帧下标的定义式与模运算形式（见下）；
2. **明确 padding 与边界帧处理**：末帧重复补齐、解码后丢弃；滑动时的边界帧在单卡下循环回绕、多卡下改为相邻 GPU 边界交换；
3. **新增一个具体数值示例**（13 帧视频、2×2 网格，同时覆盖补齐与滑动）与**一段可执行伪代码**（替代概念图）；
4. **新增 1×1 / 2×2 / 3×3 网格的定量对比表**（表 R4），并引用原稿 Figure 8 的定性结论支撑默认 2×2 的选择。

---

## 逐点回复

### 1. 重写后的 Eqs. (7)–(8)（写入修改稿 3.4 节）

**回复信中直接给出的英文修改文字：**

> We thank the reviewer for pointing out the ambiguous notation. In the revision, Eq. (7)–(8) are rewritten as follows. Let $N = n \cdot l$ denote the number of frames per grid. A video of $K$ frames is first padded to $\tilde{K} = \lceil K / N \rceil \cdot N$ frames by repeating the last frame $r = \tilde{K} - K$ times ($r = 0$ when $N \mid K$), and is organized into $L = \tilde{K}/N$ grids. The padded frames participate only as attention context during diffusion and are discarded after decoding, so they never appear in the output. We write a grid as an explicit tuple of frame indices (subscripts), replacing the previously undefined superscript ranges and the symbol $P$:
>
> $$\mathrm{Layout}_1 = \big\{ \mathcal{Z}_g \big\}_{g=1}^{L}, \quad \mathcal{Z}_g = \big( z_{(g-1)N+1},\, \dots,\, z_{gN} \big), \tag{7}$$
>
> where $z_t$ is the latent of the $t$-th frame. In the odd time steps, the frame sequence is cyclically rotated forward by one column ($n$ frames) and re-partitioned:
>
> $$\mathrm{Layout}_2 = \big\{ \mathcal{Z}'_g \big\}_{g=1}^{L}, \quad \mathcal{Z}'_g[j] = z_{\big( ((g-1)N + n + j - 1) \bmod \tilde{K} \big) + 1}, \; j = 1, \dots, N. \tag{8}$$
>
> The modular index makes the boundary treatment explicit: the leading $n$ frames wrap around into the last grid, so every pair of temporally adjacent grids shares one column of frames in attention across consecutive time steps, and the rotation is exactly inverted when returning to Layout$_1$ at the next even step. In the multi-GPU setting (Sec. 3.5), the wrap-around is replaced by exchanging the $n$ boundary latent frames between adjacent GPUs, so the video head and tail are never mixed.

**关于符号 $P$ 的说明**：原稿中 $P$ 仅表示 Layout$_1$ 最后一个网格的起始帧索引（即 $P = (L-1)N + 1$），但未给出定义，且与帧区间上标混排造成歧义，审稿人的批评成立。修改稿删除该符号，网格与帧的对应关系完全由式 (7) 的显式下标给出。

### 2. padding 与边界帧的处理（完整规则）

| 情形 | 处理方式 |
|---|---|
| $N \mid K$（整除） | 无需 padding，直接按式 (7) 组网格 |
| $N \nmid K$（不整除） | 末帧重复 $r = \lceil K/N\rceil \cdot N - K$ 次补齐；补齐帧参与扩散过程中的注意力计算（提供上下文），**解码后直接丢弃**，不进入输出视频 |
| 滑动时的网格边界帧 | 每帧在相邻两个时间步分别属于两个不同的网格（Layout$_1$ 与 Layout$_2$ 错位 $n$ 帧），从而间接实现跨网格交互 |
| 单卡滑动时的回绕边界 | 前 $n$ 帧循环进入最后一个网格（式 (8) 的模运算），首尾网格每两个时间步交互一次 |
| 多卡滑动时的 GPU 边界 | 回绕替换为：rank $g$ 的最后 $n$ 帧 latent 发送到 rank $g{+}1$，偶数步归还；单次通信量 $n \cdot h \cdot w \cdot C$，与 GPU 数量无关 |

实际实验中（8/16/48/200 帧，默认 $N=4$）帧数均可被 $N$ 整除，padding 仅在网格尺寸消融（如 3×3）中生效。

### 3. 具体示例（写入修改稿，替代一张概念图）

> **Example.** Consider a $K = 13$-frame video with the default 2×2 grid ($n = l = 2$, $N = 4$). Since $13 \not\equiv 0 \pmod 4$, the last frame is repeated 3 times, giving $\tilde{K} = 16$ and $L = 4$ grids (frames 14–16 are padding, shown in gray):
>
> - **Even steps (Layout$_1$):** $\mathcal{Z}_1 = (1,2,3,4)$, $\mathcal{Z}_2 = (5,6,7,8)$, $\mathcal{Z}_3 = (9,10,11,12)$, $\mathcal{Z}_4 = (13,\boxed{14,15,16})$
> - **Odd steps (Layout$_2$)**, rotating forward by $n = 2$: $\mathcal{Z}'_1 = (3,4,5,6)$, $\mathcal{Z}'_2 = (7,8,9,10)$, $\mathcal{Z}'_3 = (11,12,13,\boxed{14})$, $\mathcal{Z}'_4 = (\boxed{15,16},1,2)$ — note frames 1–2 wrap around into the last grid.
>
> After diffusion and decoding, frames 14–16 are dropped, and the output video has exactly 13 frames.

### 4. 伪代码（写入修改稿 3.4 节，Algorithm 1）

```python
def grid_alternation(z, step, n, l):          # z: [B, C, K, h, w] frame latents
    N = n * l
    if K % N: z = pad_last_frame(z, (-K) % N) # repeat last frame r times
    Kt = z.shape[2]                           # Kt = ceil(K/N) * N
    if Kt == N:                               # single grid: nothing to slide
        return combine(z, n, l)
    rot = list(range(n, Kt)) + list(range(n)) # cyclic rotation by one column
    perm = rot            if step % 2 == 1 else inverse(rot)  # Layout2 / Layout1
    return combine(z[:, :, perm], n, l)       # exact inverse on even steps

for step, t in enumerate(timesteps):          # inversion & denoising loops alike
    z_grids = grid_alternation(z, step, n, l)
    eps      = unet(z_grids, t, prompt_emb)   # grids processed serially / per GPU
    z        = ddim_step(eps, t, z_grids)
x_edited = vae_decode(restore_frames(z))[:K]  # drop padded frames, output K frames
```

（多卡变体：将 `rot` 的回绕段替换为相邻 rank 间交换 `n` 帧边界 latent，其余不变，对应修改稿 Sec. 3.5 的 Algorithm 2。）

### 5. 网格尺寸定量对比：1×1 / 2×2 / 3×3（新增表 R4）

原稿 Figure 8 与 Sec. 4.3.2 已给出 2×2 vs 3×3 的定性对比（3×3 全局美感与形状保真更佳，但出现细节伪影，如狮子左耳模糊、右耳异常折叠）。按审稿人要求，修改稿补充三档网格尺寸的定量结果（新增表 R4，完整测试集，指标与表 2 一致）：

**表 R4：网格尺寸消融（网格容量 N、网格分辨率、单网格空间 token 数一并列出）**

| 网格 | N | 网格分辨率 (px / latent) | 空间 token/网格 | Masked-PSNR↑ | SSIM↑ | Clip-T↑ | Tem-Con↑ | 采样耗时(s)↓ |
|---|---|---|---|---|---|---|---|---|
| 1×1 | 1 | 512 / 64² | 4,096 | 【待填】 | 【待填】 | 【待填】 | 【待填】 | 【待填】 |
| 2×2（默认） | 4 | 1024 / 128² | 16,384 | 【待填】 | 【待填】 | 【待填】 | 【待填】 | 图 9 已有 |
| 3×3 | 9 | 1536 / 192² | 36,864 | 【待填】 | 【待填】 | 【待填】 | 【待填】 | 【待填】 |

**默认选择 2×2 的论证（回复信口径）**：

- **1×1 退化为逐帧独立编辑**：网格自注意力在单帧内运算，等价于去掉帧间交互机制，短程一致性仅剩滑动带来的间接约束。预期 Tem-Con 显著下降（与原稿 Sec. 3.4 中"grid self-attention 确保 intra-grid 一致性"的论述对应，需以补跑数据核实）。
- **3×3 计算与分布代价**：token 数为 2×2 的 2.25 倍、自注意力开销约 5 倍；1536×1536 的拼格超出预训练 T2I 模型的常规运算分辨率，原稿 Figure 8 已观察到细节粗化伪影。此外 $N = 9$ 对帧数整除约束更强（8/16/200 帧均不整除，需依赖 padding）。
- **2×2 为质量-效率折中**：网格容量 4 已保证相邻 4 帧的直接注意力交互 + 滑动跨网格交互；1024×1024 拼格与社区 grid trick 的常用设定一致，属预训练模型的友好分辨率。

---

## 给作者本人的执行清单（不放入回复信）

1. **代码补 padding**（当前 `grid.py:5` 断言整除、不整除直接报错，与回复信口径不符）：`combine_frames_in_grid` 前置末帧重复补齐，`restore_frames_from_grid` 后裁剪回 K 帧；或在 pipeline 层做 pad/drop（改动更小）。这是表 R4 的 3×3 一列（8/16/200 帧不整除 9）的前提。
2. **补跑实验矩阵**：网格 {1×1, 2×2, 3×3} × 帧数 {8,16,48,200} × 完整测试集；1×1 直接用 `grid_sizeH=grid_sizeW=1`（K%1 恒整除，滑动自动退化为逐帧+循环移位）；3×3 注意 192×192 latent（36,864 token）的显存压力，可能需开 xformers / 分块注意力，若 OOM 需在回复信注明所用注意力后端与 2×2 一致性。
3. **耗时应补全**：回复信表 R4 的耗时列目前只有 2×2 有数据（原稿图 9），1×1、3×3 需补测（48 帧设置，与图 9 同口径）。
4. **论文正文修改**：按第 1 节重写 Eqs. (7)–(8)；第 3 节示例与第 4 节伪代码插入 3.4 节（可相应精简 Figure 4，审稿人明确表示伪代码比概念图更有用）；Sec. 4.3.2 扩充为定量表 R4 + 保留 Figure 8 定性图。
5. **与上一份 grid diffusion 意见的口径统一**：`rebuttal_reply_grid_diffusion.md` 表 R3 的网格形状是 2×1/2×2/2×3/2×4/3×4（含耗时），本意见是 1×1/2×2/3×3（含质量指标）。两份回复可共用一套补跑数据，但表格分开呈现，避免审稿人发现口径矛盾；其中 2×2 行数值必须完全一致。
6. **Eq. (8) 原式的 $Z_0^{1,\dots,N-n}$ 首网格是不完整网格**，与代码的循环旋转实现不符——本次重写采用模运算形式（即循环旋转口径，等价于上一份回复清单中的方案 A），确认两处论文修改一致，不要再保留 partial grid 的写法。
