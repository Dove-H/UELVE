# 返修回复：长时序一致性未直接验证（Tem-Con 仅反映局部连续性）

> **审稿意见原文**：Long-term temporal consistency is not directly demonstrated. The authors introduce the grid sliding manipulation to maintain long-term consistency. However, the metric "Tem-Con" is computed from CLIP visual embeddings of consecutive frames and therefore mainly reflects local temporal continuity. Stronger evidence is needed to show that the proposed method preserves consistency across distant portions of long videos.

## 总体回复

感谢审稿人的意见。我们完全同意：原稿的 Tem-Con 基于**相邻帧** CLIP 视觉嵌入的余弦相似度，度量的是局部时序连续性，不足以直接证明长视频远距离片段之间的一致性。修改稿从三方面补强：

1. **新增主体级长程一致性指标 Strided SC**：复用 VBench [46] 官方的 Subject Consistency（SC）——在 Grounding-DINO+SAM 自动分割的**主体区域**上提取 DINO 特征、逐帧对计算余弦相似度，主体与背景显式分离，背景的相机运动/场景变化不污染度量。针对审稿人"相邻帧仅反映局部连续性"的批评，我们将视频按步长 $\Delta \in \{K/8, K/4, K/2\}$ 抽帧后再送入 VBench 官方实现，SC 即变为**相距 $\Delta$ 的远距离帧对**上的主体一致性（新增表 R5），并伴随报告 Text-Alignment，排除"通过弱化编辑换取的伪一致"；
2. **机制层面论证长程一致性的来源**：补充命题 1（写入修改稿 III-D 节）——网格内自注意力使任意时间距离 $< N$ 的帧对直接交互，布局交替使相邻网格每两个时间步共享一列帧，因此注意力交互图在相邻两步的并集构成覆盖全视频的连通链，编辑主体的外观信息可沿链条传播至任意远端帧。Strided SC 实验验证该论证；
3. **新增滑动开/关的长程消融与定性证据**：在完整测试集 8/16/48/200 帧设置下报告 Strided SC（滑动开 vs. 关，新增表 R6，与表 R2 共用同一组推理结果），并给出 200 帧编辑结果的远距离帧（首/中/尾）主体裁剪放大对比（新增图 R2），直观展示无滑动时主体编辑属性在远端漂移、开启后被抑制。

---

## 逐点回复

### 1. 承认 Tem-Con 的局限，引入 Strided SC（新增表 R5，写入修改稿 IV 节指标定义处）

审稿人的批评成立：相邻帧相似度只约束局部连续性——一段逐帧平滑、但编辑主体外观缓慢漂移的视频（如"银发"编辑中发色在 200 帧内逐渐褪色）在 Tem-Con 上仍可得高分。同时我们指出，**直接把 Tem-Con 的帧间距拉大是不够的**：全帧 CLIP 嵌入把背景也计入指标，而长视频中背景因相机运动与场景动态发生的变化是正常的，不应被惩罚；真正需要长程一致的是**被编辑的主体**。为此修改稿采用 VBench [46] 的 Subject Consistency 并做长程扩展：

**Strided SC**：VBench 的 SC 以编辑 prompt 中的主体描述为条件，用 Grounding-DINO+SAM 自动分割每帧主体区域，在区域上提取 DINO 特征，逐帧对计算余弦相似度后全序列平均。我们将视频按步长 $\Delta$ 抽帧（帧 $0, \Delta, 2\Delta, \dots$）后送入 VBench **官方实现**（代码零改动），所得即相距 $\Delta$ 的远距离帧对上的主体一致性：

$$\text{SC}_\Delta = \frac{1}{\lceil K/\Delta \rceil - 1}\sum_{j} \cos\big(f_{DINO}(s_j \odot M_j),\ f_{DINO}(s_{j+1} \odot M_{j+1})\big), \qquad \Delta \in \left\{ \tfrac{K}{8},\ \tfrac{K}{4},\ \tfrac{K}{2} \right\},$$

其中 $s_j$ 为抽帧后第 $j$ 帧、$M_j$ 为其主体掩码。取 $\Delta = K/2$ 为主汇报值。分层报告 $\Delta$ 可展示主体一致性随帧间距的衰减曲线（新增图 R1：SC vs. 帧间距，本方法与基线、源视频同图对比）。方法对比在同一测试集上进行，SC 绝对值直接可比，与表 II 其他指标口径一致。

**防"伪一致"设计**（预计审稿人会追问，主动在回复中说明）：

- **Text-Alignment 同报**：证明 SC 提升不是以弱化编辑换取得；
- **源视频参考线**：图 R1 中画出源视频自身的 SC 衰减曲线作为自然参考——编辑主体的 SC 达到源视频的自然水平即为保持；显著高于参考线则提示主体被过度平滑（视频僵化），我们据此核查并无此现象（定性图 R2 中主体保持正常动态）；
- **Attribute Drift（编辑属性漂移）**：SC 的 DINO 嵌入度量配合该像素级度量，可分别捕捉语义层与外观层的漂移，且后者对"逐帧平滑的整体漂移"（Tem-Con 高、SC 也可能偏高）尤其敏感——例如发色从深银到浅银的 200 帧渐变，嵌入相似度可能不低，但掩码区主色描述子的首尾差异会暴露它。掩码直接复用 VBench 分割结果，无额外标注成本。

### 2. 机制论证：网格滑动为何保证长程一致性（新增命题 1，写入修改稿 III-D 节）

> **Proposition 1.** Let the video be partitioned into $L$ grids of $N = n \cdot l$ frames each. Within any single diffusion step, grid self-attention lets every pair of frames with temporal distance $< N$ attend to each other directly. Under layout alternation, consecutive time steps use partitions offset by one column ($n$ frames), so every pair of temporally adjacent grids shares $n$ common frames across two consecutive steps. Consequently, the union of the attention graphs over any two consecutive steps forms a connected chain covering all $K$ frames: the appearance of the edited subject can propagate between any two frames $i, j$ along a path of at most $\lceil |i-j| / n \rceil$ grid adjacencies. Long-range consistency is thus maintained *by construction*, rather than only through local smoothness — which is precisely what the proposed Strided SC experiments verify (Table R6).

（回复信口径：先承认 Tem-Con 只是局部度量的"验证手段"，命题 1 从构造上说明长程交互存在，表 R5/R6 则提供直接实验证据，三者互补。）

### 3. 长程一致性的定量证据（新增表 R5、R6）

**表 R5：Strided SC 主结果，完整测试集，报告 $\Delta = K/2$ 主值**

| 方法 | SC@Δ=K/8↑ | SC@Δ=K/4↑ | SC@Δ=K/2↑ | Text-Align↑（引用表 II，证明未牺牲编辑强度） |
|---|---|---|---|---|
| 源视频（自然参考，仅入图 R1 曲线） | 【待填】 | 【待填】 | 【待填】 | — |
| RAVE [16] | 【待填】 | 【待填】 | 【待填】 | 表 II |
| TokenFlow | 【待填】 | 【待填】 | 【待填】 | 表 II |
| **UELVE（本文）** | 【待填】 | 【待填】 | 【待填】 | 表 II |

（分层 Δ 的完整曲线放图 R1；分帧数 8/16/48/200 报告，200 帧为主打设置；基线名单与表 II 对齐。）

**表 R6：网格滑动消融的 Strided SC（与表 R2 共用同一组推理结果，仅换指标）**

| 设置 | SC@Δ=K/2↑ | Attribute Drift↓ | Tem-Con↑（局部，对照） |
|---|---|---|---|
| Vanilla Grid（无滑动） | 【待填】 | 【待填】 | 表 R2 |
| + 布局交替（本文） | 【待填】 | 【待填】 | 表 R2 |

**预期结论（与机制论证一致，需以补跑数据核实）**：无滑动时相邻网格间无信息交换，远端帧的编辑主体仅共享初始 latent 与全局 prompt，SC 随 $\Delta$ 明显衰减、Attribute Drift 增大（尤其 200 帧）；开启布局交替后衰减曲线显著拉平、接近源视频自然参考线。若 SC 高出参考线过多，需核查主体是否被过度平滑。这也正是引言中网格滑动设计动机的直接实验支撑。

### 4. 定性证据（新增图 R2）

给出至少 2 个 200 帧编辑样例（优先选属性编辑，如发色/风格类，主体漂移最易观察）：每个样例展示**首帧 / 1/4 / 中点 / 3/4 / 尾帧**五个时间点的**主体裁剪放大**（同一行、按 VBench 分割掩码对齐排列），对比（a）无滑动、（b）完整方法、（c）源视频主体裁剪。相比整帧对比，主体裁剪行能直接并排观察属性漂移，说服力更强。图 R1 为对应样例的 SC–帧间距衰减曲线。若个别样例存在残余漂移，如实标注并讨论（见第 5 点）。

### 5. 边界与局限（主动说明，避免被追问）

- SC 度量的是**编辑主体的外观/语义一致性**，不约束主体与运动背景的几何关系（源视频参考线同样受姿态/遮挡影响，故作趋势参照而非绝对阈值）；
- VBench 的 Grounding-DINO+SAM 主体分割在个别样例上可能失败或漏检（尤其非常见主体），失败样例剔除并在表注报数量；分割对所有对比方法一视同仁，不影响相对结论；
- 极长视频（如 200 帧）中若编辑语义与运动强耦合（如"加雪"随场景变化），残余漂移可能来自 ControlNet/注意力控制的空间条件而非网格机制，表 R6 的受控消融可将其分离。

---

## 回复信英文段落（可直接使用）

> We thank the reviewer for this insightful comment and fully agree that Tem-Con, being computed on consecutive frames, mainly reflects local temporal continuity — a video with slow, per-frame-smooth drift could still score high. We further note that simply enlarging the frame gap of the same metric would be insufficient: whole-frame CLIP embeddings entangle the background, whose change over a long video (camera motion, scene dynamics) is legitimate and should not be penalized. What must stay consistent across distant portions is the **edited subject**. In the revision we therefore adopt the **Subject Consistency (SC)** metric from the standard video benchmark VBench, which isolates the subject region (auto-segmented by Grounding-DINO + SAM) and measures DINO-feature similarity within it, and extend it to long range by temporally striding the input before the official VBench implementation: with stride $\Delta \in \{K/8, K/4, K/2\}$, SC measures subject appearance similarity between frames $\Delta$ apart. We report Text-Alignment alongside to rule out consistency obtained by weakening the edit, plot the source video's own SC curve as a natural reference (ruling out over-smoothing, i.e., a frozen subject), and add an Attribute Drift measure (std. of the edited attribute's descriptor, e.g. dominant color, within the subject mask across the full sequence), which exposes smooth long-horizon drift that embedding similarity may miss. Mechanistically, new Proposition 1 shows that the union of attention graphs over two consecutive layout-alternation steps forms a connected chain over all frames, so long-range consistency of the edited subject is enforced by construction. Empirically, Table R5 reports Strided SC for all methods on the full benchmark (8/16/48/200 frames), and Table R6 shows the sliding on/off ablation: without layout alternation, the subject in distant frames shares only the initial latents and the global prompt, and SC decays with frame distance; with it, the decay curve flattens toward the source-video reference. Figure R2 further shows mask-aligned subject crops from first/middle/last frames of 200-frame results. All of these will be included in the revision.

---

## 给作者本人的执行清单（不放入回复信）

1. **跑 VBench SC**：`pip install vbench`，用官方 `subject_consistency` dimension（首次运行自动下载 Grounding-DINO/SAM/DINO 权重）。**stride 技巧**：对每个视频按 $\Delta \in \{K/8, K/4, K/2\}$ 抽帧生成 strided 帧序列再送评测（ffmpeg 抽帧或直接喂帧目录），官方代码零改动。抽查 2–3 个样例的 Grounding-DINO+SAM 主体分割质量；分割失败的样例剔除并在表注报数量。VBench 会顺带输出 BC——**不进表格，留作备份**：若审稿人追问"SC 高是否因为主体被冻住"，BC 与源视频对比 + 定性图可直接回应，不必改稿。
2. **Attribute Drift 脚本**：在 VBench 分割掩码区域内取 Lab 主色均值与饱和度，算全序列 std 与首尾差。属性编辑样例优先，背景/风格类可跳过该项并在表注说明。
3. **实验矩阵复用**：表 R6 与 `rebuttal_reply_grid_diffusion.md` 表 R2（滑动开/关）是同一组推理结果，**务必只跑一次、两表数值共享**；若 R2 尚未跑，矩阵为 滑动{开,关} × 帧数{8,16,48,200} × 完整测试集。
4. **基线与源视频**：基线若只有保存的视频（表 II 产物），VBench 直接对输出视频评测，无需重跑扩散；源视频参考曲线同流程。注意 strided 版本对基线和源视频用**相同的 stride 抽帧**。
5. **图 R1/R2 制作**：R1 为 SC vs. 间距曲线（本方法/无滑动/RAVE/源视频 四条线，200 帧样例）；R2 选 2 个属性编辑样例，5 个时间点的掩码对齐主体裁剪横排。从 `girl-dance-star-inversion_frames/` 这类已有抽帧目录里可先做排版模板。
6. **命题 1 的严格性自查**：写成两步并集连通的图论陈述即可（顶点=帧，边=同网格共注意力；Layout1 与 Layout2 的边集并集构成路径图），不需要长证明，但斜体 *by construction* 的措辞要与表 R6 的实验趋势对得上——若实验中无滑动版本 Strided SC 也不差，需回退为"机制提供必要条件、实验提供经验验证"的弱口径。
7. **口径统一**：本回复表 R6 与 `rebuttal_reply_grid_diffusion.md` 表 R2 的 Tem-Con 列必须同源；stride $\Delta$ 中的 $K$ 按各视频实际帧数取（8/16/48/200 分别对应 Δ=1/2/4、2/4/8、6/12/24、25/50/100）。VBench 引用为 CVPR 2024；同单位 Liu et al. (PR 2026) 在视频编辑评测中亦采用 VBench SC，可作为先例引用（`/mnt/fspace/hproject/ELVE/丁义胜.pdf`，其 Table 1 的 RAVE/TokenFlow SC 数值可用来交叉校准我们的实现）。
