# Rebuttal Replies — PR-D-26-10575（全部审稿人意见与回复汇总）

> 用途：汇总各审稿人的意见（Comment）与我们的回复（Response），供后续统一修改与提交。
> 来源：/mnt/fspace/hproject/ELVE/Response_to_PR_reviews.pdf（25 页，已全文核对）。
> 状态标记：✅ 已完成 / ✍️ 草稿（待补数据）/ ❌ 未写
> 目前已录入全部内容：Meta Review + Reviewer 1–6。
>
> ⚠️ 注意：原 PDF 中 "Response to Reviewer 4/5/6" 段的引言均写着 "Reviewer 3: ..."，应为回复信模板未改干净；意见编号（4.x/5.x/6.x）与审稿人的对应关系需在提交前核对。

---

## Meta Review（Response to the Meta Review）

### Summary Comment 1

**意见原文：**
The submission has been substantially by a range of reviews across multiple respects. The reviewers recognise the proposed framework promising, but its central claims are insufficiently validated. Major revision is required to quantitatively evaluate the LLM router, grid/sliding ablations, long-term temporal consistency, and end-to-end efficiency, while clarifying metric definitions, implementation details, and the scope of long-video claims.

**Response S1：** ✍️（仅总述，未逐项）

We highly value the meta-review and the comprehensive assessments from all reviewers. We have thoroughly examined the entire manuscript and made targeted revisions to resolve every raised concern. In what follows, we list the meta-review and reviewer comments, alongside our itemized responses to address all issues.

> 💡 元审稿点名的五大主题 = 后面各审稿人意见的并集：① LLM 路由定量评估；② grid/sliding 消融；③ 长时序一致性；④ 端到端效率；⑤ 指标定义/实现细节/长视频声明边界。修订优先级应以此为纲。

---

## Reviewer 1

### Comment 1.1 — 三大机制缺乏定量消融

**意见原文（要点）：**
三个声称的核心机制均无定量消融。Grid sliding 仅在一个视频上靠肉眼验证（Fig. 7）；grid size 仅在一个狮子视频上验证，结果被描述为 "overall beauty is improved but the details are coarse"；LLM 路由仅在三段视频上验证（Fig. 9）——全部是定性的。对于一篇贡献正是这三个机制的论文，这不足以构成充分证据。
**建议：** 在完整评测集上增加一张消融表，报告全部四个指标。

**Response 1.1：** ✍️（表已有，正文措辞可再打磨）

We appreciate this critical comment and fully agree that quantitative ablations are essential to verify our three core mechanisms. Ablation experiments across the evaluation dataset have been supplemented in the revised manuscript. A complete ablation table is provided to report all four quantitative metrics for each variant, covering grid sliding, grid size, and the LLM router module.

**Table 1: Quantitative ablations covering grid sliding, grid size, and the LLM router module.**
（采用 attention control 方案作为图像编辑方法。由于 grid size 消融需要固定帧设置，其指标值与其他两组消融相比有波动。）

| Setting | Masked PSNR ↑ | SSIM ↑ | CLIP-T ↑ | Tem-Con ↑ |
|---|---|---|---|---|
| w (grid sliding) | 22.71 | 0.6424 | 0.2936 | 0.9619 |
| w/o (grid sliding) | 22.67 | 0.6412 | 0.2935 | 0.9311 |
| 1 × 1 (grid size) | 21.91 | 0.6220 | 0.2942 | 0.9205 |
| 2 × 2 (grid size) | 21.86 | 0.6221 | 0.2937 | 0.9638 |
| 3 × 3 (grid size) | 21.83 | 0.6214 | 0.2926 | 0.9651 |

---

### Comment 1.2 — LLM router 未被隔离，Fig. 9(a) 自相矛盾

**意见原文（要点）：**
LLM router 从未被单独隔离：Fig. 9(a) 中 Setting 1–3 只是配置不同，而非"有无 router"的差别，实验只能说明固定配置不足。LLM 自身的准确率、相对关键词规则的优势、延迟与 token 开销（Table 2 缺失）、跨 LLM 的敏感性均未测量。另外 Fig. 9(a) 中一个被 router 判为 motion editing（需 fine-tuning）的案例，source 和 target prompt 完全相同（"a girl is dancing" -> "a girl is dancing"），纯文本 prompt diff 不可能区分出该任务。
**建议：** 在 >100 对 prompt 上对照人工标注报告路由准确率，给出 oracle 上界与规则下界，另报 router 延迟；并修正或解释 Fig. 9(a)。

**Response 1.2：** ✍️

We greatly appreciate your valuable comments. We explain the experimental setup in Figure 9. Setting 3 utilizes an LLM for adaptive model adjustment according to text prompts. Setting 1 removes the masking mechanism without fine-tuning, and Setting 2 adopts a fixed masking mechanism and fine-tunes temporal attention. Setting 1 and Setting 2 adopt fixed model configurations, where the LLM router is removed, which corresponds to the case of isolating the LLM router. By comparing the performance of Setting 1, Setting 2 and Setting 3, we can obtain the performance gain brought by our LLM module. To intuitively demonstrate the benefits of the LLM module, we have supplemented quantitative ablation results of the LLM module, as shown in Table 1. Due to the lack of editing masks in Setting 1, editing leakage occurs. Its Masked PSNR, SSIM and Tem-Con metrics are poor, while the CLIP-T metric achieves a relatively high value because there is no mask constraint. Setting 2 fine-tunes the temporal module and achieves decent performance across all metrics, but it consumes substantial unnecessary time and computational resources for fine-tuning.

We have supplemented the confusion matrix to evaluate the accuracy of the LLM router, as shown in Table 2. The Average precision of the LLM router exceed 93 percent, which verifies its reliability. We use DeepSeek-v4 as our default LLM. Its average single inference latency is 0.51 seconds, with an average token cost of 1934 tokens. In addition, we test the robustness of our LLM router across different LLMs, and the evaluation results are shown in Table 3. There are slight performance gaps among different LLMs, which stem from the inherent capabilities of each LLM. We select DeepSeek-v4 after comprehensively considering performance, cost and usability. Readers can freely switch to more powerful LLMs to build the LLM router for higher routing accuracy.

Figure 9(a) shows the motion editing results under Setting 3. Our motion editing module is adapted from MotionEditor, which requires an additional motion guidance map to guide the generation of new motions. To enable the LLM to correctly recognize this as a motion editing task, our code appends the filename of the motion guidance map to the target prompt as a hint for the LLM. Therefore, the actual prompt pair is 'a girl is dancing' → 'a girl is dancing (/motion guidance filename)'. This is merely an engineering trick in our implementation. We omitted this detail in the figure, which caused misunderstanding. Thanks for your careful observation; we will correct Figure 9(a).

**Table 2: Confusion matrix of editing type prediction.**

| Prediction \ Ground Truth | Local | Global | Simple | Complex | Motion | Total | Precision |
|---|---|---|---|---|---|---|---|
| Local Editing (need mask) | 98 | 4 | — | — | — | 102 | 0.9608 |
| Global Editing (no mask) | 2 | 96 | — | — | — | 98 | 0.9796 |
| Simple Editing (no fine-tune) | — | — | 91 | 6 | 5 | 102 | 0.8922 |
| Complex Editing (need fine-tune) | — | — | 7 | 89 | 0 | 96 | 0.9271 |
| Motion Editing (need motion tune) | — | — | 2 | 5 | 95 | 102 | 0.9314 |
| Total/average | 100 | 100 | 100 | 100 | 100 | 500 | 0.9382 |

**Table 3: Robustness experiments on different LLMs.**

| LLM | Local-F1 | Global-F1 | Simple-F1 | Complex-F1 | Motion-F1 | Macro-F1 |
|---|---|---|---|---|---|---|
| DeepSeek v4 | 0.9703 | 0.9697 | 0.9010 | 0.9082 | 0.9406 | 0.9380 |
| GLM 5.2 | 0.9490 | 0.9770 | 0.9090 | 0.9160 | 0.9480 | 0.9400 |
| Qwen-2 | 0.9430 | 0.9785 | 0.8880 | 0.8960 | 0.9420 | 0.9295 |
| Llama3 | 0.9310 | 0.9540 | 0.8760 | 0.8980 | 0.9435 | 0.9205 |
| doubao-seed-2.0-lite | 0.9160 | 0.9555 | 0.8600 | 0.8995 | 0.9070 | 0.9076 |
| kimi-k2.6 | 0.9175 | 0.9380 | 0.8490 | 0.9010 | 0.9085 | 0.9028 |
| oracle (human) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

> ⚠️ 待办：审稿人要求的 rule-based lower bound（规则下界）尚未在 Table 3 中给出，目前只有 oracle 上界。可考虑补一行 "keyword rule" 基线。

---

### Comment 1.3（Question）— UELVE-Attention 禁用 FastSAM mask 后的四指标

**意见原文（要点）：**
UELVE-Attention 在禁用 FastSAM mask 时的四项指标是多少？由于 Masked PSNR 奖励背景复制，且 UELVE 是 Table 1 中唯一使用 mask 的方法，需要该实验来把 Masked PSNR 的优势归因于 grid 机制。

**Response 1.3：** ✍️

Thanks for this valuable comment, which helps us clarify the source of performance gain on Masked PSNR.

UELVE is not the only mask enabled method in Table 1. For instance, Tune-A-Video, Video-P2P and FateZero all employ masking mechanisms. Our Masked PSNR is comparable to these mask based approaches. The marginal performance advantage originates from the higher mask precision of FastSAM and inherent limitations of those baselines, rather than our grid mechanism. Tune-A-Video relies solely on the structural information preserved during inversion to guide editing, which leads to severe corruption of the background. The fine-tuning and null-text optimization in Video-P2P partially distort the original video tone in terms of brightness and contrast. As for FateZero, it reshapes the original self-attention into spatial-temporal self-attention, which induces background distortion and degrades Masked PSNR.

The core emphasis of UELVE is to flexibly decide whether to enable masking via prompt semantic analysis. This design reduces unnecessary mask generation overhead and avoids the inconvenience brought by manually controlled mask parameters, instead of pursuing higher mask quality. The specific choice of masking scheme is not critical to our method. Replacing FastSAM with alternative mask generation approaches of equivalent performance will not yield obvious metric variations. Naturally, adopting a lower quality masking mechanism will inevitably hurt Masked PSNR. As shown in Table 4, we disable FastSAM and adopt the same masking mechanism used in other baselines. After this modification, UELVE achieves reduced Masked PSNR and SSIM, a slight improvement in CLIP-T, and nearly unchanged Tem-Con.

We acknowledge that our original use of a different masking mechanism compared with baselines indeed creates an unfair comparison. Thank you for pointing this out, and we will revise accordingly.

**Table 4: 禁用 FastSAM、改用与基线相同的 masking 机制后的结果。**

| Method | Masked PSNR↑ 8-f | 16-f | 48-f | 200-f | SSIM↑ 8-f | 16-f | 48-f | 200-f | CLIP-T↑ 8-f | 16-f | 48-f | 200-f | Tem-Con↑ 8-f | 16-f | 48-f | 200-f |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Tune-A-Video | 13.91 | 15.87 | 15.13 | — | 0.4350 | 0.4749 | 0.4418 | — | 0.2960 | 0.2727 | 0.2894 | — | 0.9488 | 0.9481 | 0.9441 | — |
| Video-P2P | 23.66 | 22.23 | — | — | 0.6563 | 0.6217 | — | — | 0.2937 | 0.2955 | — | — | 0.9695 | 0.9640 | — | — |
| FateZero | 23.83 | — | — | — | 0.6572 | — | — | — | 0.2872 | — | — | — | 0.9576 | — | — | — |
| UELVE (Cross-Attn) | 23.86 | 22.66 | 21.64 | 21.57 | 0.6840 | 0.6627 | 0.6168 | 0.5827 | 0.2847 | 0.2962 | 0.2926 | 0.2898 | 0.9655 | 0.9654 | 0.9594 | 0.9725 |
| UELVE (FastSAM) | 24.05 | 22.83 | 21.78 | 21.65 | 0.6849 | 0.6635 | 0.6176 | 0.5833 | 0.2841 | 0.2955 | 0.2922 | 0.2893 | 0.9653 | 0.9655 | 0.9592 | 0.9725 |

---

### Comment 1.4（Question）— 200 帧、G=2 时每步每边界的通信量与占比

**意见原文（要点）：**
200 帧、G=2 时每个 timestep 每个边界交换多少 MB？通信占单步时间的比例多少？G=4、8 时交换量如何扩展？另外，随着 G 增大（同样 50 步内边界更多），编辑视频质量（不只是速度）是否退化？

**Response 1.4：** ✍️

Data exchange only occurs at the grid boundaries between adjacent GPUs. For an original video resolution of 512×512 and a 2×2 grid layout, when G=2 (GPUs) with a total of 200 frames, latent data of 2 frames is transmitted per inference timestep, which is approximately 0.0625 MB. The communication latency is around 30 ms, accounting for less than 1 percent of the total per-timestep runtime. The communication overhead rises to 90 ms for G=4 and 210 ms for G=8. In general, for G=n, the communication latency equals (n − 1) × 30 ms. Multi-GPU execution only distributes computational workload across devices and does not alter the inference pipeline. The model operates identically under single-GPU and multi-GPU settings, so the edited video quality does not degrade as the number of GPUs increases.

---

### Comment 1.5（Question）— 与 1.4 相同（重复编号）

**意见原文：** 与 Comment 1.4 完全相同（可能是审稿系统重复提交）。

**Response 1.5：** ✍️（与 1.4 回复相同）

同 Response 1.4：Data exchange only occurs at the grid boundaries between adjacent GPUs. For an original video resolution of 512×512 and a 2×2 grid layout, when G=2 (GPUs) with a total of 200 frames, latent data of 2 frames is transmitted per inference timestep, which is approximately 0.0625 MB. The communication latency is around 30 ms, accounting for less than 1 percent of the total per-timestep runtime. The communication overhead rises to 90 ms for G=4 and 210 ms for G=8. In general, for G=n, the communication latency equals (n − 1) × 30 ms. Multi-GPU execution only distributes computational workload across devices and does not alter the inference pipeline. The model operates identically under single-GPU and multi-GPU settings, so the edited video quality does not degrade as the number of GPUs increases.

> 💡 建议：可在回复中注明 "Response 1.5 is identical to Response 1.4, as Comments 1.4 and 1.5 are duplicates."

---

### Comment 1.6（Question）— FastSAM 是类无关的，谁把改动的名词短语映射到具体分割区域？如何跨帧/跨网格/跨两种布局传播？

**Response 1.6：** ✍️

FastSAM is indeed a class-agnostic segmentation model. We leverage the CLIP model to map the modified noun phrase to the target segmented region. We generate masks for each individual frame, and these masks are bound to frame indices. Therefore, mask propagation across frames is unnecessary. Similarly, no propagation is required across grids or between the two alternating layouts.

---

## Reviewer 2

> 总体评价：2×2 网格与交替布局是合理的工程选择，定性结果表明方法有用。未解决的问题在于论文的宽泛声明与实际测量之间的距离：LLM router 的可靠性、处理真正长片段的成本、以及计时比较的公平性尚不清楚。

### Comment 2.1 — CLIP-T 定义自相矛盾

**意见原文（要点）：**
Section 4.1.4 称 CLIP-T 是 average CLIP distance，但 Table 1 标注为向上箭头。若是距离，越低应越好；若是相似度，公式与描述需写明。请给出确切定义，并在用该指标下结论前修正表格与讨论。

**Response 2.1：** ✍️

Thank you very much for your careful review and valuable comments, which greatly help us improve our paper. The CLIP-T evaluation metric adopted in our work is referenced from the RAVE method. The original description in the RAVE paper states that "Additionally, we evaluate the textual alignment of the editing, given a prompt, by calculating the average distance between the CLIP embedding of the text prompt and all frames (CLIP-T)", where it is defined as cosine distance. However, after verifying the official code of RAVE, we found that the metric is actually implemented as cosine similarity, which is a writing error in the original RAVE paper. We will correct the description in our manuscript and uniformly define CLIP-T as cosine similarity in the revised version.

The precise definition is:

CLIP-T = (1/T) · Σ_{t=1}^{T} ⟨f_text, f_img^(t)⟩ / (‖f_text‖₂ · ‖f_img^(t)‖₂)

where f_text is the CLIP text embedding of the prompt, f_img^(t) is the CLIP image embedding of the t-th frame, and T is the number of frames. CLIP-T is a similarity, so higher is better.

---

### Comment 2.2 — LLM router 的可靠性未量化

**意见原文（要点）：**
当前系统中 DeepSeek 主要是在既有选项（masking、temporal fine-tuning、Attention Control、ControlNet）中做选择。论文未给出模型版本、prompt、解码设置与错误处理。Fig. 9 只给三个例子，没有测量 router 做出正确选择的可靠性。建议与固定设置、规则或无 LLM 的 router 做小规模定量对比；并把 LLM 带来的增益与 grid sliding 及既有编辑模块的增益分开。

**Response 2.2：** ✍️

**Model version, prompt, decoding settings, and error handling.** The LLM used in our router is DeepSeek-V4. The complete prompt is provided in the supplementary material. The decoding settings are temperature = 0.2 and max_tokens = 2048 with streaming output enabled (stream=True). For error handling, we adopt a retry mechanism: if the model output cannot be parsed after the maximum number of retries, the system falls back to a conservative default configuration.

**Router reliability and ablation.** We conducted a quantitative evaluation of the routing accuracy of the LLM module. The confusion matrix and the corresponding evaluation metrics are reported in Table 2; the average precision metric exceeds 93%, indicating that the LLM router is reliable. Furthermore, as shown in Table 1, we performed a quantitative ablation study to separate the gain brought by the LLM from the gains due to grid sliding and the existing editing modules.

---

### Comment 2.3 — Eqs. (7)-(8) 索引、符号 P、非整除处理不清晰；要求伪代码与 1×1/2×2/3×3 定量结果

**意见原文（要点）：**
Eqs. (7)-(8) 没有把索引方式、符号 P、以及视频长度不能被 grid size 整除时的处理写清楚。请说明 padding 与边界帧的处理，并给一个具体例子。一小段伪代码比再加概念图更有用。1×1、2×2、3×3 网格的定量结果也能支撑默认 2×2 布局的选择。

**Response 2.3：** ✍️

Let N = n · l be the grid capacity. For a video of K frames, we append M − K copies of the last frame, where M = ⌈K/N⌉ · N, so that M is a multiple of N.

Layout1 = { Z^{1,…,N}_1 , …, Z^{P,…,K,…,M}_L }　　(1)

In Eq. (7), P = (L − 1) · N + 1 (equivalently P = M − N + 1) denotes the first frame index of the last grid, which may contain fewer than N real frames and completed by the appended copies; the superscript of each Z denotes the frame-index range it covers.

Layout2 = { Z^{1,…,n}_0 , Z^{n+1,…,N+n}_1 , …, Z^{P+n,…,M}_L }　　(2)

In Eq. (8), sliding shifts every grid boundary forward by n frames, which introduces one extra grid Z^{1,…,n}_0. The appended copies frames participate only as attention context and are truncated before decoding, so the edited video has exactly K frames.

**Concrete example (K = 13, 2×2 grid: n = 2, N = 4, M = 16, P = 13; frames 14–16 are copies of frame 13):**

- Even steps (Layout1): Z^{1,2,3,4}_1, Z^{5,6,7,8}_2, Z^{9,10,11,12}_3, Z^{13,14,15,16}_4 — four full 1024×1024 grids;
- Odd steps (Layout2): Z^{1,2}_0 (processed as a 512×1024 grid), Z^{3,4,5,6}_1, Z^{7,8,9,10}_2, Z^{11,12,13,14}_3, Z^{15,16}_4. Frames 14–16 are dropped before decoding; the output has exactly 13 frames.

As shown in Table 1, we conduct quantitative ablation experiments on the grid size. When the grid size is 1 × 1, frame consistency cannot be maintained via grid sliding, resulting in poor editing quality with a low Tem-Con metric. For a grid size of 3 × 3, attention calculations need to be performed on 9 frames within each grid simultaneously. Although the overall performance improves, the model loses fine-grained details and tends to produce blurriness and artifacts, which is reflected in the Masked PSNR, SSIM and CLIP-T metrics. Furthermore, the primary computational bottleneck of the transformer block lies in self-attention, whose computational cost scales quadratically with the image area. For the original video with 36 frames, using a 2 × 2 grid size means each grid contains 4 frames and the whole sequence is processed in 9 iterations. The total computational cost is approximately 4 × 4 × 9 = 144 units. When adopting a 3 × 3 grid size, each grid covers 9 frames and the sequence is processed in 4 iterations, yielding a total computational cost of roughly 9 × 9 × 4 = 324 units, which imposes substantially higher computation requirements. Therefore, we select the 2 × 2 grid size as the default setting after comprehensive trade-offs.

**Algorithm 1: Grid diffusion with layout alternation**

```text
Require: frame latents z_{1:K}, grid shape n × l, timesteps
 1: N ← n·l;  L ← ⌈K/N⌉;  M ← L·N;  P ← M − N + 1
 2: z ← z_{1:K} ∥ (z_K × (M−K))          ▷ tail padding with copies of the last frame
 3: for step, t in timesteps do
 4:   if step is odd then
 5:     grids ← { Z^{1,…,n}_0, Z^{n+1,…,N+n}_1, …, Z^{P+n,…,M}_L }   ▷ Layout2, Eq. (2)
 6:   else
 7:     grids ← { Z^{1,…,N}_1, …, Z^{P,…,M}_L }                       ▷ Layout1, Eq. (1)
 8:   end if
 9:   for each Z^{s,…,e} ∈ grids do
10:     ε ← UNet(grid(z_{s:e}), t, prompt)
11:     z_{s:e} ← DDIMStep(ε, t, z_{s:e})
12:   end for
13: end for
14: return Decode(Restore(z_{1:K}))        ▷ drop appended copies; output has K frames
```

> ⚠️ 待办：回复末尾有一句未写完的残句 "as shown in Table 1, we performed a quantitative ablation study to"——提交前需删除或补全。

---

### Comment 2.4 — Table 2 计时比较的公平性（GPU 数、峰值显存、端到端时间）

**意见原文（要点）：**
Table 2 混合了普通单 GPU 基线与 RAVE×2GPU、UELVE×2GPU；且只报 inversion + denoising，而 UELVE 还可能额外花费时间在 LLM 路由、分割、磁盘传输、通信、以及 300 步 one-shot fine-tuning 上。请为每一行报告 GPU 数与峰值显存，并给出含/不含可选 fine-tuning 的端到端时间。否则宣称相对 one-shot 方法的优势难以解释。

**Response 2.4：** ❌ 未写回复（已准备了 Table 5 骨架：GPU 数 + GMem/Time，8f/16f/48f/200f，但 UELVE 各行数据空缺）

**Table 5（骨架，待补 UELVE 行数据）：**

| Method | GPU | 8f GMem | 8f Time | 16f GMem | 16f Time | 48f GMem | 48f Time | 200f GMem | 200f Time |
|---|---|---|---|---|---|---|---|---|---|
| Tune-A-Video | 1 | 7.24GB | 5m03s | 10.27GB | 8m56s | 20.1GB | 27m17s | — | — |
| Video-P2P | 1 | 11.7GB | 5m25s | 15.8GB | 9m55s | — | — | — | — |
| FateZero | 1 | 13.9GB | 2m44s | — | — | — | — | — | — |
| RAVE | 1 | 8.7GB | 1m48s | 10.2GB | 3m08s | 11.7GB | 8m37s | 12.4GB | 35m05s |
| RAVE | 2 | （待补） | | | | | | | |
| ControlVideo | 1 | 21.7GB | 6m28s | 23.5GB | 9m11s | — | — | — | — |
| Control-A-Video | 1 | 15.8GB | 42s | 16.6GB | 59s | 17.3GB | 2m22s | 18.5GB | 9m13s |
| TokenFlow | 1 | 10.9GB | 1m52s | 12.9GB | 3m17s | 15.5GB | 9m56s | 21.1GB | 47m33s |
| AdaFlow | 1 | 12.6GB | 1m07s | 15GB | 1m41s | 18.8GB | 5m14s | 23.82GB | 18m29s |
| GLFM | 1 | 10.0GB | 1m43s | 11.7GB | 3m14s | 22.1GB | 8m43s | 23.42GB | 41m20s |
| UELVE-Att (zero-shot) | 2 | （待补） | | | | | | | |
| UELVE-Att (one-shot) | 2 | （待补） | | | | | | | |
| UELVE-Att (LLM) | 2 | （待补） | | | | | | | |
| UELVE-CNet (zero-shot) | 2 | （待补） | | | | | | | |

---

### Comment 2.5 — "任意长度视频"声明 vs 定量结果止于 200 帧

**意见原文（要点）：**
论文声称可处理任意长度视频，但定量结果止于 200 帧。所描述的 1–3 分钟片段在给定帧率下远长于此。请说明用的是完整片段还是选取的窗口。若不补充全长度的时间与质量结果，请把宽泛声明替换为关于已测长度的更有限表述，并讨论磁盘 I/O 与累积质量漂移。

**Response 2.5：** ❌ 未写回复

---

## Reviewer 3

> 总体评价：稿件提出 UELVE——LLM 驱动的长视频编辑统一框架，结合 LLM prompt 比较与模型适配、带 grid sliding 的 grid diffusion、多 GPU 数据并行。主题相关、grid manipulation 有意思。但当前版本若干核心声明支撑不足，建议 major revision（在 major revision 与 reject 之间偏 borderline）。

### Comment 3.1 — LLM 的贡献验证不足

**意见原文（要点）：**
LLM 的贡献验证不足。LLM 是框架的核心组件，用于分析 source/target prompt 并决定相应的模型适配（local/global 编辑、zero-shot/one-shot 设置），但 Fig. 9 的消融只有三个定性示例，且只是把 LLM 自适应设置与两个固定配置比较。需要更广泛的定量评估，证明基于 LLM 的 prompt 分析在所考虑的各类编辑任务上都有收益。

**Response 3.1：** ✍️

Thank you for your valuable comments. We appreciate your feedback. Admittedly, the three qualitative ablation cases in Figure 9 alone cannot sufficiently verify the editing-quality benefits from LLM-based prompt analysis. We have therefore added quantitative ablation experiments for the LLM routing module, as reported in Table 1.

Setting1 (w/o LLM) disables masking and temporal fine-tuning. While it performs well for global editing, it fails local editing and yields low Masked PSNR, SSIM and Tem-Con scores. Setting2 (w/o LLM) enables masking and temporal fine-tuning and suits local editing, yet suffers interference from redundant masks and fine-tuning for global editing. Despite metrics comparable to Setting3 (w/ LLM), it introduces extra overhead for mask generation and fine-tuning.

It is worth noting that improved editing quality is a side effect of the LLM module. Its primary role is to automatically adjust model configurations by analyzing prompts, so as to construct a unified video-editing framework that adapts to diverse editing tasks. The three examples in Figure 9 aim to highlight that the LLM can accomplish all three distinct types of editing tasks via flexible model-configuration adjustments, whereas fixed model configurations are only capable of partial editing tasks, as illustrated in Table 1.

**Table 6: Quantitative ablation study of the LLM routing module.**（任务可完成性，✓/×）

| Editing Task | Setting 1 (w/o LLM) | Setting 2 (w/o LLM) | Setting 3 (w/ LLM) |
|---|---|---|---|
| Local attribute editing | × | ✓ | ✓ |
| Object replacement | × | ✓ | ✓ |
| Global stylization | ✓ | × | ✓ |
| Motion editing | × | × | ✓ |
| Motion Preservation | × | ✓ | ✓ |

---

### Comment 3.2 — 消融大多只是定性的

**意见原文（要点）：**
大部分消融只有定性结果，缺乏说服力。grid sliding（Fig. 7）、grid size（Fig. 8）、LLM 组件（Fig. 9）的分析都依赖少量视觉示例，没有定量测量；且图中差异因尺寸和清晰度有限并不容易评估。这些组件是方法的核心，作者应在足够大的视频集上给出定量消融，并提供更清晰的视觉对比。

**Response 3.2：** ❌ 未写回复（可复用 Response 1.1 的 Table 1 + 承诺改进图片清晰度）

---

### Comment 3.3 — 长时序一致性未被直接证明

**意见原文（要点）：**
长时序一致性未被直接证明。作者引入 grid sliding 以保持长程一致性，但 Tem-Con 由连续帧的 CLIP 视觉嵌入计算，主要反映局部时间连续性。需要更强的证据表明方法能保持长视频中远隔片段之间的一致性。

**Response 3.3：** ❌ 未写回复
（相关材料：仓库中已有 rebuttal_reply_longterm_consistency.md，或可复用；另可考虑远隔帧对上的 CLIP/PSNR 指标，如 Tem-Con@stride-k。）

---

### Comment 3.4 — 预备知识部分过长

**意见原文（要点）：**
Section 3.2 用了大量篇幅介绍 latent diffusion、DDPM、DDIM 的标准背景（含前向/反向扩散公式和训练目标），这些概念对本期刊读者已是常识。建议缩短该节，只保留描述所提方法直接需要的记号与公式，节省的篇幅用于澄清方法与实现细节。

**Response 3.4：** ❌ 未写回复

---

### Comment 3.5 — 统一框架的实验验证有限（需要分任务定量分析）

**意见原文（要点）：**
统一框架的实验验证有限。论文主要声称之一是 UELVE 能通过 LLM prompt 分析与相应模型适配来适应不同编辑任务，但该能力主要靠 Fig. 9 定性展示。Table 1 只报了 UELVE-Attention 和 UELVE-ControlNet 的结果，没有清楚显示基于 LLM 的适配在各编辑类别上的贡献。分任务的定量分析能更好支撑"同一框架可有效处理多种编辑场景"的声明。

**Response 3.5：** ✍️（正文未写，已有 Table 7 骨架，多数格子空缺）

**Table 7: Per-task quantitative ablation study for LLM-driven adaptation（骨架，待补）**
（所有 Setting 共享相同的 grid-diffusion backbone。Setting 1：固定 zero-shot，无 mask、无 fine-tuning；Setting 2：固定 one-shot，mask 与 temporal 模块 fine-tuning 恒开；Setting 3：完整 UELVE，LLM 指导的自适应配置选择；Setting 4：固定 one-shot，mask 与 Motion 模块 fine-tuning 恒开。）

| Editing Task | Setting | Masked-PSNR ↑ | SSIM ↑ | Clip-T ↑ | Tem-Con ↑ |
|---|---|---|---|---|---|
| Global stylization | Setting1 | （待补） | | | |
| | Setting2 | （待补） | | | |
| | Setting3 | — | 0.668 | 0.286 | 0.970 |
| | Setting4 | — | — | — | — |
| Local Attribute Modification | Setting1 | （待补） | | | |
| | Setting2 | 28.4？ | 0.71？ | 0.287 | 0.966 |
| | Setting3 | 24.01 | 0.676 | 0.284 | 0.968 |
| | Setting4 | — | — | — | — |
| Complex Structure Change | Setting1 | （待补） | | | |
| | Setting2 | （待补） | | | |
| | Setting3 | （待补） | | | |
| | Setting4 | — | — | — | — |
| Motion Preservation | Setting1 | （待补） | | | |
| | Setting2 | （待补） | | | |
| | Setting3 | （待补） | | | |
| | Setting4 | — | — | — | — |
| Motion Editing | Setting1 | — | — | — | — |
| | Setting2 | — | — | — | — |
| | Setting3 | （待补） | | | |
| | Setting4 | — | 0.668 | 0.286 | 0.970 |

> ⚠️ 待核对：原稿 "Setting2 284 071 0.287 0.966" 数字粘连，疑为 28.4 / 0.71（或其他），需查原始数据；"Motion Editing Setting4 0.668 0.286 0.970" 与 "Global stylization Setting3" 数值完全相同，疑为复制错误或排版串行。

---

### Comment 3.6 — 未提供代码/实现仓库

**意见原文（要点）：**
作者未提供代码或实现仓库。鉴于方法复杂度较高，这限制了可复现性。应发布代码并提供足以复现报告结果的实现细节。

**Response 3.6：** ❌ 未写回复

---

### Comment 3.7 — 长视频编辑声明需要更强的实验支撑

**意见原文（要点）：**
长视频编辑的声明需要更强的实验支持。论文描述的数据集包含约 1 分钟的视频和最长数分钟的自录视频，而主要定量评测最多只考虑 200 帧。鉴于长视频编辑是本文的核心动机与贡献，需要在显著更长的序列上做实验来支撑该声明。

**Response 3.7：** ❌ 未写回复（与 Comment 2.5 同主题，可合并准备实验/声明收敛）

---

### Comment 3.8 — 数学记号需修订；语法/排版错误；Fig. 5 "GUP1/GUP2" 拼写错误

**意见原文（要点）：**
Section 3.2 的数学记号应仔细修订。稿件中还有大量语法与排版错误。若干图中文字过小；Fig. 5 中 "GUP1/GUP2" 等标签应更正（应为 GPU）。

**Response 3.8：** ✍️

Thank you for pointing out the writing and presentation issues. In response to your suggestions, we have thoroughly proofread the entire manuscript. We have corrected all typos, grammatical and punctuation errors, as well as non-standard usages. Moreover, we have standardized the formatting of technical terms and polished the academic writing to improve the overall readability of the paper.

> ⚠️ 待办：确认已在稿件中实际更正 "GUP1/GUP2" → "GPU1/GPU2"，并放大过小文字的图表。

---

## Reviewer 4

> 总体评价（原文标注 "Reviewer 3"，疑为笔误）：论文研究文本引导的长视频编辑，是 diffusion 与视频生成领域活跃且重要的课题。grid sliding 机制与多 GPU 方案很可能被其他研究者复用；LLM 路由组件则更专属于本 pipeline，且随着视觉语言模型（VLM）日益普及用于直接视觉引导，其竞争力可能已经下降。

### Comment 4.1 — Tables 1–3 缺少统计报告（跨运行方差、置信区间、显著性检验）

**Response 4.1：** ✍️（表格已备好，正文仅有两句话，可再充实；Table 2/3 的方差与置信区间段落只剩标题句）

The variance across runs, confidence intervals and significance tests for Table 1.

Since our UELVE has two implementations, we present the results of significance tests separately.

The variance across runs and confidence intervals for Table 2. Runtime metrics shown in Table 2 and Table 3 represent wall-clock inference time. Significance tests (p-values) are not required for runtime measurements.

The variance across runs and confidence intervals for Table 3.

**Table 8: Table 1 对应的跨运行方差（Mean ± std）。**

| Method | PSNR 8-f | PSNR 16-f | PSNR 48-f | PSNR 200-f | SSIM 8-f | SSIM 16-f | SSIM 48-f | SSIM 200-f | CLIP-T 8-f | CLIP-T 16-f | CLIP-T 48-f | CLIP-T 200-f | Tem-Con 8-f | Tem-Con 16-f | Tem-Con 48-f | Tem-Con 200-f |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Tune-A-Video | 13.91±0.12 | 15.87±0.14 | 15.13±0.12 | — | 0.4350±0.0239 | 0.4749±0.0170 | 0.4418±0.0110 | — | 0.2960±0.0048 | 0.2727±0.0080 | 0.2894±0.0077 | — | 0.9488±0.0039 | 0.9481±0.0075 | 0.9441±0.0109 | — |
| Video-P2P | 23.66±0.23 | 22.23±0.21 | — | — | 0.6563±0.0226 | 0.6217±0.0202 | — | — | 0.2937±0.0123 | 0.2955±0.0079 | — | — | 0.9695±0.0061 | 0.9640±0.0109 | — | — |
| FateZero | 23.83±0.19 | — | — | — | 0.6572±0.0166 | — | — | — | 0.2872±0.0119 | — | — | — | 0.9576±0.0124 | — | — | — |
| RAVE | 22.98±0.16 | 20.18±0.20 | 20.24±0.12 | 20.16±0.16 | 0.5680±0.0230 | 0.5349±0.0073 | 0.5894±0.0103 | 0.5419±0.0120 | 0.3145±0.0085 | 0.3128±0.0082 | 0.3044±0.0127 | 0.2928±0.0141 | 0.9795±0.0171 | 0.9753±0.0074 | 0.9696±0.0147 | 0.9619±0.0088 |
| ControlVideo | 20.60±0.15 | 19.97±0.17 | — | — | 0.5923±0.0210 | 0.5517±0.0180 | — | — | 0.3070±0.0074 | 0.3012±0.0091 | — | — | 0.9746±0.0082 | 0.9732±0.0067 | — | — |
| Control-A-Video | 10.72±0.14 | 9.715±0.18 | 11.52±0.13 | — | 0.4802±0.0221 | 0.3944±0.0242 | 0.4245±0.0131 | — | 0.2942±0.0062 | 0.3065±0.0078 | 0.2979±0.0083 | — | 0.9673±0.0071 | 0.9638±0.0093 | 0.9506±0.0112 | — |
| TokenFlow | 23.55±0.08 | 20.71±0.27 | 20.03±0.29 | 20.38±0.27 | 0.6502±0.0198 | 0.6431±0.0127 | 0.6313±0.0096 | 0.6130±0.0149 | 0.2917±0.0098 | 0.2980±0.0090 | 0.2973±0.0072 | 0.2898±0.0099 | 0.9776±0.0160 | 0.9712±0.0104 | 0.9724±0.0132 | 0.9718±0.0055 |
| AdaFlow | 23.10±0.19 | 21.15±0.27 | 20.15±0.18 | 19.32±0.08 | 0.6500±0.0261 | 0.6499±0.0201 | 0.6357±0.0129 | 0.5819±0.0209 | 0.2945±0.0065 | 0.2961±0.0071 | 0.3040±0.0072 | 0.2708±0.0121 | 0.9789±0.0119 | 0.9762±0.0070 | 0.9720±0.0109 | 0.9517±0.0094 |
| GLFM | 22.91±0.16 | 20.49±0.18 | 20.05±0.23 | 20.19±0.19 | 0.6031±0.0064 | 0.6190±0.0149 | 0.6217±0.0204 | 0.6017±0.0133 | 0.3149±0.0152 | 0.3117±0.0070 | 0.3037±0.0091 | 0.2991±0.0073 | 0.9791±0.0139 | 0.9677±0.0104 | 0.9626±0.0163 | 0.9563±0.0055 |
| UELVE-Attention | 24.05±0.13 | 22.83±0.18 | 21.78±0.09 | 21.65±0.18 | 0.6849±0.0210 | 0.6635±0.0200 | 0.6176±0.0187 | 0.5833±0.0074 | 0.2841±0.0047 | 0.2955±0.0051 | 0.2922±0.0056 | 0.2893±0.0076 | 0.9653±0.0058 | 0.9655±0.0123 | 0.9594±0.0124 | 0.9725±0.0139 |
| UELVE-ControlNet | 21.18±0.15 | 20.00±0.27 | 20.29±0.25 | 20.47±0.09 | 0.5711±0.0290 | 0.5853±0.0274 | 0.5412±0.0185 | 0.5570±0.0135 | 0.3174±0.0145 | 0.3210±0.0095 | 0.3056±0.0088 | 0.2957±0.0050 | 0.9801±0.0028 | 0.9765±0.0095 | 0.9707±0.0083 | 0.9643±0.0106 |

**Table 9: 各方法的 95% t 分布置信区间。**

| Method | PSNR 8-f | PSNR 16-f | PSNR 48-f | PSNR 200-f | SSIM 8-f | SSIM 16-f | SSIM 48-f | SSIM 200-f | CLIP-T 8-f | CLIP-T 16-f | CLIP-T 48-f | CLIP-T 200-f | Tem-Con 8-f | Tem-Con 16-f | Tem-Con 48-f | Tem-Con 200-f |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Tune-A-Video | [13.76,14.06] | [15.69,16.05] | [14.98,15.28] | — | [0.405,0.465] | [0.454,0.496] | [0.428,0.456] | — | [0.290,0.302] | [0.263,0.282] | [0.280,0.299] | — | [0.943,0.955] | [0.939,0.957] | [0.931,0.957] | — |
| Video-P2P | [23.37,23.95] | [21.96,22.50] | — | — | [0.628,0.684] | [0.590,0.653] | — | — | [0.278,0.309] | [0.286,0.305] | — | — | [0.962,0.977] | [0.951,0.977] | — | — |
| FateZero | [23.59,24.07] | — | — | — | [0.637,0.677] | — | — | — | [0.273,0.301] | — | — | — | [0.942,0.973] | — | — | — |
| RAVE | [22.78,23.18] | [19.93,20.43] | [20.09,20.39] | [19.96,20.36] | [0.536,0.600] | [0.525,0.545] | [0.575,0.604] | [0.527,0.557] | [0.304,0.325] | [0.302,0.323] | [0.289,0.320] | [0.275,0.311] | [0.958,1.000] | [0.965,0.986] | [0.949,0.990] | [0.950,0.974] |
| ControlVideo | [20.41,20.79] | [19.75,20.19] | — | — | [0.563,0.622] | [0.527,0.576] | — | — | [0.298,0.316] | [0.290,0.312] | — | — | [0.964,0.985] | [0.963,0.983] | — | — |
| Control-A-Video | [10.54,10.90] | [9.48,9.95] | [11.35,11.69] | — | [0.449,0.511] | [0.360,0.429] | [0.405,0.444] | — | [0.287,0.301] | [0.296,0.317] | [0.288,0.308] | — | [0.957,0.978] | [0.951,0.977] | [0.936,0.965] | — |
| TokenFlow | [23.45,23.65] | [20.37,21.05] | [19.67,20.39] | [20.04,20.72] | [0.626,0.674] | [0.628,0.658] | [0.619,0.644] | [0.593,0.633] | [0.279,0.304] | [0.286,0.310] | [0.288,0.307] | [0.278,0.302] | [0.957,0.998] | [0.958,0.984] | [0.956,0.989] | [0.964,0.980] |
| AdaFlow | [22.87,23.33] | [20.81,21.49] | [19.92,20.38] | [19.22,19.42] | [0.618,0.682] | [0.625,0.675] | [0.619,0.653] | [0.557,0.607] | [0.287,0.302] | [0.287,0.305] | [0.296,0.312] | [0.259,0.283] | [0.965,0.993] | [0.967,0.985] | [0.960,0.984] | [0.941,0.962] |
| GLFM | [22.72,23.10] | [20.27,20.71] | [19.78,20.32] | [19.95,20.43] | [0.594,0.612] | [0.601,0.637] | [0.589,0.654] | [0.583,0.621] | [0.294,0.336] | [0.303,0.321] | [0.293,0.314] | [0.289,0.309] | [0.960,0.998] | [0.954,0.981] | [0.943,0.982] | [0.948,0.964] |
| UELVE-Attention | [23.89,24.21] | [22.55,23.11] | [21.67,21.89] | [21.38,21.92] | [0.659,0.711] | [0.637,0.690] | [0.594,0.641] | [0.573,0.593] | [0.278,0.290] | [0.289,0.302] | [0.285,0.299] | [0.281,0.298] | [0.958,0.973] | [0.950,0.981] | [0.944,0.975] | [0.955,0.990] |
| UELVE-ControlNet | [21.00,21.36] | [19.58,20.42] | [19.84,20.74] | [20.33,20.61] | [0.533,0.609] | [0.545,0.626] | [0.516,0.566] | [0.539,0.575] | [0.297,0.338] | [0.307,0.335] | [0.297,0.314] | [0.288,0.303] | [0.976,0.984] | [0.963,0.990] | [0.960,0.981] | [0.951,0.977] |

**Table 10: 配对双边 t 检验 p 值 — UELVE-Attention vs 对应行基线。**（p < 0.05 为显著差异）

| Method | PSNR 8-f | PSNR 16-f | PSNR 48-f | PSNR 200-f | SSIM 8-f | SSIM 16-f | SSIM 48-f | SSIM 200-f | CLIP-T 8-f | CLIP-T 16-f | CLIP-T 48-f | CLIP-T 200-f | Tem-Con 8-f | Tem-Con 16-f | Tem-Con 48-f | Tem-Con 200-f |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Tune-A-Video | <0.001 | <0.001 | <0.001 | — | <0.001 | <0.001 | <0.001 | — | 0.002 | 0.002 | 0.074 | — | 0.005 | 0.185 | 0.002 | — |
| Video-P2P | 0.115 | 0.028 | — | — | 0.764 | 0.127 | — | — | 0.029 | 0.031 | — | — | 0.072 | 0.800 | — | — |
| FateZero | 0.525 | — | — | — | 0.452 | — | — | — | 0.474 | — | — | — | 0.652 | — | — | — |
| RAVE | <0.001 | <0.001 | <0.001 | <0.001 | 0.002 | <0.001 | 0.042 | 0.044 | 0.003 | 0.093 | 0.009 | 0.523 | 0.142 | 0.016 | 0.801 | 0.608 |
| ControlVideo | <0.001 | <0.001 | — | — | 0.003 | 0.008 | — | — | 0.011 | 0.022 | — | — | 0.033 | 0.041 | — | — |
| Control-A-Video | <0.001 | <0.001 | <0.001 | — | <0.001 | <0.001 | <0.001 | — | 0.004 | 0.007 | 0.013 | — | 0.008 | 0.014 | 0.021 | — |
| TokenFlow | 0.006 | <0.001 | <0.001 | <0.001 | 0.191 | 0.154 | 0.716 | 0.004 | 0.086 | 0.132 | 0.002 | 0.867 | 0.070 | 0.384 | 0.232 | 0.139 |
| AdaFlow | 0.005 | <0.001 | <0.001 | <0.001 | 0.111 | 0.457 | 0.783 | 0.317 | <0.001 | 0.003 | 0.009 | 0.040 | 0.063 | 0.392 | 0.402 | 0.404 |
| GLFM | <0.001 | <0.001 | <0.001 | <0.001 | 0.001 | 0.047 | 0.687 | 0.025 | 0.022 | 0.006 | 0.002 | 0.012 | 0.076 | 0.642 | 0.893 | 0.119 |
| UELVE-Attention | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| UELVE-ControlNet | <0.001 | <0.001 | <0.001 | <0.001 | 0.005 | 0.011 | 0.002 | 0.583 | 0.012 | 0.009 | 0.003 | 0.060 | 0.002 | 0.491 | 0.872 | 0.535 |

**Table 11: 配对双边 t 检验 p 值 — UELVE-ControlNet vs 对应行基线。**

| Method | PSNR 8-f | PSNR 16-f | PSNR 48-f | PSNR 200-f | SSIM 8-f | SSIM 16-f | SSIM 48-f | SSIM 200-f | CLIP-T 8-f | CLIP-T 16-f | CLIP-T 48-f | CLIP-T 200-f | Tem-Con 8-f | Tem-Con 16-f | Tem-Con 48-f | Tem-Con 200-f |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Tune-A-Video | <0.001 | <0.001 | <0.001 | — | <0.001 | 0.002 | <0.001 | — | 0.101 | <0.001 | 0.054 | — | <0.001 | 0.001 | 0.026 | — |
| Video-P2P | <0.001 | <0.001 | — | — | 0.006 | 0.097 | — | — | 0.156 | 0.005 | — | — | 0.286 | 0.314 | — | — |
| FateZero | <0.001 | — | — | — | 0.003 | — | — | — | 0.015 | — | — | — | 0.035 | — | — | — |
| RAVE | <0.001 | 0.384 | 0.668 | 0.010 | 0.210 | 0.007 | 0.008 | 0.046 | 0.545 | 0.095 | 0.874 | 0.058 | 0.776 | 0.063 | 0.695 | 0.761 |
| ControlVideo | 0.042 | 0.812 | — | — | 0.088 | 0.124 | — | — | 0.331 | 0.066 | — | — | 0.442 | 0.537 | — | — |
| Control-A-Video | <0.001 | <0.001 | <0.001 | — | <0.001 | <0.001 | <0.001 | — | 0.032 | <0.001 | 0.022 | — | 0.001 | 0.003 | 0.007 | — |
| TokenFlow | <0.001 | 0.027 | 0.037 | 0.173 | 0.001 | 0.009 | 0.001 | 0.001 | 0.097 | <0.001 | 0.296 | 0.348 | 0.857 | 0.920 | 0.123 | 0.058 |
| AdaFlow | <0.001 | 0.001 | 0.532 | <0.001 | 0.011 | 0.007 | 0.004 | 0.226 | 0.130 | 0.001 | 0.790 | 0.012 | 0.850 | 0.899 | 0.392 | 0.428 |
| GLFM | <0.001 | 0.056 | 0.086 | 0.023 | 0.064 | 0.128 | 0.013 | 0.027 | 0.950 | 0.352 | 0.260 | 0.044 | 0.290 | 0.860 | 0.958 | 0.199 |
| UELVE-Attention | <0.001 | <0.001 | <0.001 | <0.001 | 0.005 | 0.011 | 0.002 | 0.583 | 0.012 | 0.009 | 0.003 | 0.060 | 0.002 | 0.491 | 0.872 | 0.535 |
| UELVE-ControlNet | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — |

**Table 12: 主文 Table 2 对应的 Mean ± std（秒）。**

| Method | 8-f (sec) | 16-f (sec) | 48-f (sec) | 200-f (sec) |
|---|---|---|---|---|
| Tune-A-Video | 19.0±0.7 | 35.0±1.1 | 102.0±2.4 | — |
| Video-P2P | 61.0±1.6 | 124.0±3.0 | — | — |
| FateZero | 145.0±3.3 | — | — | — |
| RAVE ×2GPU | 19.0±0.6 | 37.0±1.0 | 104.0±2.2 | 429.0±7.1 |
| ControlVideo | 62.0±1.7 | 250.0±4.4 | — | — |
| Control-A-Video | 25.0±0.8 | 46.0±1.2 | 123.0±2.6 | 492.0±7.8 |
| TokenFlow | 51.0±1.3 | 108.0±2.3 | 334.0±5.5 | 1301.0±14.2 |
| AdaFlow | 34.0±0.9 | 72.0±1.6 | 215.0±3.8 | 901.0±11.3 |
| GLFM | 51.0±1.2 | 115.0±2.4 | 350.0±5.8 | 1266.0±13.8 |
| UELVE-Attention ×2GPU | 33.0±0.8 | 63.0±1.4 | 189.0±3.2 | 825.0±10.4 |
| UELVE-ControlNet ×2GPU | 18.0±0.5 | 34.0±0.9 | 94.0±2.0 | 372.0±6.3 |

**Table 13: 主文 Table 2 对应的 95% t 分布置信区间（秒）。**

| Method | 8-f (sec) | 16-f (sec) | 48-f (sec) | 200-f (sec) |
|---|---|---|---|---|
| Tune-A-Video | [18.1, 19.9] | [33.6, 36.4] | [99.2, 104.8] | — |
| Video-P2P | [59.0, 63.0] | [120.3, 127.7] | — | — |
| FateZero | [140.3, 149.7] | — | — | — |
| RAVE ×2GPU | [18.2, 19.8] | [35.7, 38.3] | [101.4, 106.6] | [420.6, 437.4] |
| ControlVideo | [59.9, 64.1] | [244.6, 255.4] | — | — |
| Control-A-Video | [24.1, 25.9] | [44.5, 47.5] | [120.0, 126.0] | [482.8, 501.2] |
| TokenFlow | [49.3, 52.7] | [105.0, 111.0] | [328.0, 340.0] | [1284.4, 1317.6] |
| AdaFlow | [32.8, 35.2] | [69.8, 74.2] | [210.8, 219.2] | [887.8, 914.2] |
| GLFM | [49.4, 52.6] | [111.9, 118.1] | [343.7, 356.3] | [1249.9, 1282.1] |
| UELVE-Attention ×2GPU | [31.9, 34.1] | [60.8, 65.2] | [185.3, 192.7] | [812.8, 837.2] |
| UELVE-ControlNet ×2GPU | [17.3, 18.7] | [32.7, 35.3] | [91.5, 96.5] | [364.6, 379.4] |

**Table 14: 主文 Table 3 对应的 Mean ± std（秒）— Inversion / Denoising 分解。**

| Method | Inv-8f | Den-8f | Inv-16f | Den-16f | Inv-48f | Den-48f | Inv-200f | Den-200f |
|---|---|---|---|---|---|---|---|---|
| Video-Infinity | — | 12.0±0.4 | — | 18.0±0.5 | — | 48.0±1.1 | — | — |
| Video-Infinity ×2GPU | — | 22.0±0.6 | — | 36.0±0.9 | — | 40.0±1.0 | — | 103.0±2.1 |
| RAVE | 17.0±0.5 | 20.0±0.6 | 30.0±0.8 | 39.0±0.9 | 83.0±1.8 | 107.0±2.2 | 322.0±5.3 | 448.0±6.6 |
| RAVE ×2GPU | 9.0±0.3 | 10.0±0.4 | 17.0±0.5 | 20.0±0.6 | 47.0±1.2 | 57.0±1.4 | 188.0±3.7 | 241.0±4.5 |
| UELVE-Attention | 24.0±0.7 | 34.0±0.9 | 57.0±1.3 | 61.0±1.4 | 164.0±3.1 | 196.0±3.6 | 688.0±9.2 | 865.0±10.7 |
| UELVE-Attention ×2GPU | 15.0±0.5 | 18.0±0.6 | 29.0±0.7 | 34.0±0.8 | 88.0±1.8 | 101.0±2.1 | 366.0±5.8 | 459.0±6.9 |
| UELVE-ControlNet | 16.0±0.5 | 18.0±0.6 | 29.0±0.7 | 36.0±0.8 | 72.0±1.6 | 98.0±2.0 | 316.0±5.1 | 367.0±5.7 |
| UELVE-ControlNet ×2GPU | 9.0±0.3 | 9.0±0.3 | 16.0±0.4 | 18.0±0.5 | 40.0±1.1 | 54.0±1.3 | 170.0±3.4 | 202.0±4.0 |

**Table 15: 主文 Table 3 对应的 95% t 分布置信区间（秒）。**

| Method | Inv-8f | Den-8f | Inv-16f | Den-16f | Inv-48f | Den-48f | Inv-200f | Den-200f |
|---|---|---|---|---|---|---|---|---|
| Video-Infinity | — | [11.5,12.5] | — | [17.4,18.6] | — | [46.6,49.4] | — | — |
| Video-Infinity ×2GPU | — | [21.2,22.8] | — | [34.8,37.2] | — | [38.8,41.2] | — | [100.4,105.6] |
| RAVE | [16.4,17.6] | [19.3,20.7] | [29.1,30.9] | [38.0,40.0] | [80.8,85.2] | [104.3,109.7] | [315.4,328.6] | [439.8,456.2] |
| RAVE ×2GPU | [8.6,9.4] | [9.5,10.5] | [16.4,17.6] | [19.3,20.7] | [45.5,48.5] | [55.3,58.7] | [183.4,192.6] | [235.4,246.6] |
| UELVE-Attention | [23.1,24.9] | [32.9,35.1] | [55.4,58.6] | [59.3,62.7] | [160.1,167.9] | [191.4,200.6] | [676.5,699.5] | [851.6,878.4] |
| UELVE-Attention ×2GPU | [14.4,15.6] | [17.3,18.7] | [28.1,29.9] | [33.0,35.0] | [85.5,90.5] | [98.4,103.6] | [358.7,373.3] | [450.3,467.7] |
| UELVE-ControlNet | [15.4,16.6] | [17.3,18.7] | [28.2,29.8] | [35.1,36.9] | [70.2,73.8] | [95.7,100.3] | [309.7,322.3] | [360.0,374.0] |
| UELVE-ControlNet ×2GPU | [8.6,9.4] | [8.6,9.4] | [15.5,16.5] | [17.4,18.6] | [38.7,41.3] | [52.4,55.6] | [165.6,174.4] | [196.8,207.2] |

---

### Comment 4.2 — 技术深度不均：grid diffusion / 多 GPU 部分形式化详细，消融却基本是定性的

**Response 4.2：** ✍️

Thank you for your positive feedback on our work. We also recognize that our ablation section lacks support from quantitative experimental results. As shown in Table 1 above, we present quantitative ablation experiments for grid-sliding and grid-size.

（显著性检验结果见上方 Table 10 / Table 11。）

---

### Comment 4.3 — 未引用 TCSVT 2026 的同类 grid diffusion 工作

**意见原文（要点）：**
文献引用有一处不足："Toward General-Purpose Video Reconstruction through Synergy of Grid-Splicing Diffusion and Large Language Models"（IEEE TCSVT, Feb 2026）采用了相同的 grid diffusion 原理与相同的动机和类比，并结合视觉语言模型做引导，但未被引用与讨论。应补充引用并澄清相对其的贡献，特别是选择纯文本 LLM 而非视觉语言模型的原因。

**Response 4.3：** ✅

Thank you for pointing out this relevant TCSVT reference. We have added the citation "Toward General-Purpose Video Reconstruction through Synergy of Grid-Splicing Diffusion and Large Language Models" (IEEE TCSVT, Feb 2026) into our manuscript, and added a dedicated discussion in Related Work to clarify the differences between our work and this paper.

Although both works adopt grid based diffusion schemes to process long videos, there exist fundamental differences in task objective, grid mechanism design, and the role of large models:

1) **Task difference:** The TCSVT work targets video reconstruction / enhancement (restoration task). In contrast, our UELVE focuses on text-guided long video editing.

2) **Grid mechanism:** The grid-splicing scheme in the TCSVT work is designed primarily to reduce GPU memory consumption for long video reconstruction. It packs patches from multiple frames into a static grid and relies on self-attention within this concatenated grid image for implicit cross-patch information flow, without explicit grid-sliding design to enforce long-range consistency across separated grids. In our UELVE, besides memory-friendly grid partitioning, we further propose grid sliding manipulation, which alternates grid layouts across diffusion timesteps to explicitly model cross-grid interactions for long-term temporal coherence.

3) **Rationale for text-only LLM instead of vision-language-model (VLM):** The TCSVT paper employs a VLM, since its reconstruction task requires visual perception over input video frames to identify visual artifacts and guide restoration. However, our text-driven video editing task derives all editing intent purely from the semantic difference between source text prompt and target text prompt. In our pipeline, the LLM acts as a decision making agent; it only performs text-level comparison and outputs editing strategy decisions. Our LLM never observes video frames, so visual understanding capability from VLM is unnecessary for our core workflow. Adopting VLM would introduce extra visual encoding overhead and potential visual noise that may perturb strategy decisions derived from prompt semantics. Therefore we choose a text-only LLM for our framework. We also note that integrating VLM for more complex multi-modal editing can be a promising direction for future extensions, which is out-of-scope for the current paper.

---

### Comment 4.4 — 摘要纯定性、引言可更紧凑、Fig. 2 配色（teal/orange）影响可读性

**意见原文（要点）：**
标题恰当、关键词合理。摘要纯定性，应加入关键定量结果。引言解释了动机与贡献但可以更紧凑。论文组织与篇幅总体合适，但消融部分相对其他部分的定量严谨性偏薄。英语总体满意、有小瑕疵。呈现质量高于平均、图表清晰、无冗余，但 Figure 2 的配色（teal 与 orange 文字）降低可读性，应修改。

**Response 4.4：** ❌ 未写回复（文末附了一张 "Figure 1: layout"，疑为修改 Fig. 2 / Fig. 5 布局的示意，需确认）

---

## Reviewer 5

> 总体评价（原文标注 "Reviewer 3"，疑为模板笔误）：稿件提出 UELVE——一个 LLM 驱动的统一框架，用 grid diffusion、grid sliding 与多 GPU 并行实现高效长视频编辑。主题相关、结果有前景，但发表前需进一步解决若干重要问题。

### Comment 5.1 — LLM 模块贡献验证不足

**意见原文（要点）：**
LLM 模块的贡献验证不足。LLM 主要作为 router，按预定义准则决定 local/global 编辑与 zero-/one-shot 配置，当前评估基本是定性的。请给出路由准确率的定量评估，并与规则、人工或 oracle 策略选择对比；同时考察对不同 prompt 与不同 LLM 的鲁棒性。

**Response 5.1：** ❌ 未写回复（可复用 Table 2 混淆矩阵 + Table 3 LLM 鲁棒性；另需补 rule-based 对比——见 R1 Comment 1.2 待办）

---

### Comment 5.2 — 长视频声明需要更强实验支撑

**意见原文（要点）：**
长视频声明需要更强的实验支持。数据集包含分钟级视频，但主要定量评估最多 200 帧。请在显著更长的（最好完整分钟级）视频上评估，并随视频长度增加报告编辑质量、时间一致性、显存消耗、运行时间与性能退化。

**Response 5.2：** ❌ 未写回复（与 R2 Comment 2.5、R3 Comment 3.7、R6 Comment 6.3 同主题——可统一准备一组长视频实验）

---

### Comment 5.3 — grid sliding / grid size 消融偏定性；需不同 sliding 配置对比

**意见原文（要点）：**
grid sliding 与 grid size 的消融主要是定性的。grid sliding 是核心技术贡献，应使用时间一致性与编辑质量指标，在整个评测集上定量对比：有无 sliding、不同 grid size、不同 sliding 配置。

**Response 5.3：** ✍️（正文已写，Table 16 数据疑为占位符）

Thank you for your comments. In addition to the grid sliding configuration (Layout 1) presented in the main text, we designed two other grid sliding strategies, as illustrated in 1 and 2.

Layout 2 connects the start and end segments during editing. It not only eliminates empty grid regions at the two ends that occur in Layout 1 during grid sliding, but also further enhances consistency between the beginning and end to mitigate consistency drift. For short videos, the content at the start and end generally does not differ much from the main body, making Layout 2 suitable for such cases. In contrast, long videos usually exhibit substantial differences between their opening and closing segments, where such start-end connections are unnecessary. Therefore, Layout 2 is tailored for short-video editing.

Built upon Layout 1, Layout 3 strengthens cross-grid interactions. This grants Layout 3 superior capability in preserving long-range consistency, rendering it more suitable for long videos. Nevertheless, when applied to short videos, the excessive interactions in Layout 3 introduce undesirable mutual interference among content regions and degrade editing quality. By comparison, Layout 1 achieves a balanced performance across both short and long video editing tasks. For this reason, we adopt Layout 1 as the default configuration.

**Table 16: grid sliding 机制消融（数据待补/待核对）。**

| Method | PSNR 8-f | PSNR 16-f | PSNR 48-f | PSNR 200-f | SSIM 8-f | SSIM 16-f | SSIM 48-f | SSIM 200-f | CLIP-T 8-f | CLIP-T 16-f | CLIP-T 48-f | CLIP-T 200-f | Tem-Con 8-f | Tem-Con 16-f | Tem-Con 48-f | Tem-Con 200-f |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Layout1(Attention) | 13.91 | 15.87 | 15.13 | — | 0.4350 | 0.4749 | 0.4418 | — | 0.2960 | 0.2727 | 0.2894 | — | 0.9488 | 0.9481 | 0.9441 | — |
| Layout1(ControlNet) | 13.91 | 15.87 | 15.13 | — | 0.4350 | 0.4749 | 0.4418 | — | 0.2960 | 0.2727 | 0.2894 | — | 0.9488 | 0.9481 | 0.9441 | — |
| Layout2(Attention) | 23.66 | 22.23 | — | — | 0.6563 | 0.6217 | — | — | 0.2937 | 0.2955 | — | — | 0.9695 | 0.9640 | — | — |
| Layout2(ControlNet) | 23.66 | 22.23 | — | — | 0.6563 | 0.6217 | — | — | 0.2937 | 0.2955 | — | — | 0.9695 | 0.9640 | — | — |
| Layout3(Attention) | 23.83 | — | — | — | 0.6572 | — | — | — | 0.2872 | — | — | — | 0.9576 | — | — | — |
| Layout3(ControlNet) | 23.83 | — | — | — | 0.6572 | — | — | — | 0.2872 | — | — | — | 0.9576 | — | — | — |

> ⚠️ 严重疑点：Table 16 当前数值与主文 Table 1 中 Tune-A-Video / Video-P2P / FateZero 的数值完全相同（Layout1=TAV、Layout2=Video-P2P、Layout3=FateZero），且 Attention 与 ControlNet 两行完全一致——几乎可以确定是**占位数据未替换**，必须用真实 Layout1/2/3 实验结果重填。另外正文说 "as illustrated in 1 and 2" 指向 Fig. 1（layout）与 Fig. 2（layout3）示意图。

---

### Comment 5.4 — LLM 只看文本 prompt，无法感知视觉/运动复杂度

**意见原文（要点）：**
LLM 只观察 source 与 target 文本 prompt，而某些决策（尤其 motion 相关编辑复杂度）可能强烈依赖源视频的视觉与时序内容。请讨论该局限，并评估相似文本 prompt 对应运动复杂度差异很大的视频案例。

**Response 5.4：** ✍️（PDF 中已含案例分析 Fig. 3 与 future work；注意 Fig. 3 图注当前是占位符 "Enter Caption"）

Thank you for your insightful feedback. We fully acknowledge this critical limitation. Our LLM makes decisions by comparing source and target prompts. It works well when prompts faithfully describe the video, yet may yield inaccurate judgements if prompts are vague or cannot capture real-world video characteristics.

As shown in Fig 3, consider two source videos with identical prompt pairs — source prompt: "A rooster standing on a tree branch", target prompt: "A metal rooster standing on a tree branch". For video 1 in (a) the rooster is nearly static; for video 2 in (b) the rooster continuously turns its head left and right. Despite substantially different motion complexity, the LLM outputs the same decision: need mask and zero-shot.

For static video 1 (a), this decision matches human preference and produces good results. For video 2 (b) with frequent head motion, human oracle would select need mask and one-shot. Without temporal fine-tuning under LLM's zero-shot setting, temporal consistency will degrade, especially on the rooster's head across frames.

In future work, we will introduce lightweight video motion statistics (e.g. optical-flow magnitude, frame-wise CLIP feature distance) as auxiliary input for the LLM agent. Fast-inference VLMs may also be adopted to support visual-grounded decision-making.

> ⚠️ 待办：Fig. 3 图注在 PDF 中为 "Figure 3: Enter Caption" 占位符，需替换。

---

### Comment 5.5 — grid 表示的实现细节需澄清

**意见原文（要点）：**
grid 表示的实现需要澄清。所有帧被 resize 到 512×512，随后多帧排入 2×2 或 3×3 网格。请澄清最终网格分辨率、每帧有效分辨率、resize 策略、不完整网格的处理，以及所有基线是否在相同有效分辨率下评估。

**Response 5.5：** ✅

Thank you for pointing out these missing implementation details about our grid representation. We acknowledge that these details were not sufficiently clarified in the original manuscript. Each raw video frame is pre-processed to 512×512 by first resizing the short side to 512 with bilinear interpolation and anti-aliasing, followed by center-cropping. When assembling frames into an n × l grid, each individual frame retains its original 512×512 resolution; the stitched grid image therefore has a pixel resolution of (512n) × (512l), corresponding to a VAE latent resolution of (64n) × (64l). For example, a 2×2 grid yields a 1024×1024 pixel image with 128×128 latent representation. Frame stitching is performed via lossless pixel block copying without additional resampling, so the effective resolution per frame remains 512×512. After latent denoising, we reverse the stitching operation to recover individual 512×512 output frames. For video sequences whose total frame count cannot be perfectly divided by the grid capacity n·l, we pad the final incomplete grid by replicating the last valid frame of the video to form a full grid input; these padded duplicate frames are discarded after decoding and are not present in the final edited video. For fair comparison, all baseline methods take exactly the same pre-processed 512×512 frames as input and run under their official default 512×512 resolution setting.

---

### Comment 5.6 — 效率比较应报告完整端到端成本

**意见原文（要点）：**
效率比较应报告完整的端到端成本。当前 runtime 主要只含 inversion 与 denoising，而 UELVE 还可能涉及 LLM 推理、mask 生成和逐视频 fine-tuning。在支撑 "高效且高性价比的统一编辑" 声明时应计入这些成本。

**Response 5.6：** ❌ 未写回复（与 R2 Comment 2.4 同主题——可共用 Table 5 + 端到端分解）

---

### Comment 5.7 — 声明措辞需更谨慎；需分任务结果与统计信息

**意见原文（要点）：**
一些实验声明的措辞应更谨慎。Table 1 显示 UELVE 并非在每个指标、每个配置上都最优。建议给出 local/global/motion 编辑的分任务结果；并更清楚地报告评估样本数、指标定义与统计波动。

**Response 5.7：** ❌ 未写回复（分任务结果可复用 Table 7；统计信息可复用 Table 8–11；措辞收敛需改文中 "consistently superior" 类表述）

---

## Reviewer 6

> 总体评价（原文标注 "Reviewer 3"，疑为模板笔误）：UELVE 用 LLM 选择 local/global 与 zero-/one-shot 配置，将视频帧排入网格用图像扩散处理，并在扩散步之间滑动网格以促进时间一致性；框架还将网格批次分布到多 GPU，用四个自动质量指标和最多 200 帧的运行时间测量对比九个基线。作者声称持续更优的编辑质量、时间连贯性与效率——**这些声明超出了证据**，因为实验既未证实长视频运行、也未隔离 LLM 贡献、评估规格与统计支撑也不足。

### Comment 6.1 — 相对 Grid Diffusion 的技术进步未隔离；缺伪代码与定义

**意见原文（要点）：**
相对所引 Grid Diffusion 方法的技术进步未被隔离；稿件未定义 tile 分辨率、最终网格分辨率、padding、帧排序、边界处理，以及 sliding 在 inversion 与 denoising 中的确切应用方式。请提供可执行的伪代码与匹配的 Grid Diffusion 基线，并在完整评测集上量化 sliding 与 grid size 的影响。

**Response 6.1：** ✍️（已写 Frame ordering 与 Boundary handling 两段；tile/grid 分辨率部分见 Response 5.5 残段；缺 matched Grid Diffusion 基线与 sliding/grid-size 全集量化——可指向 Table 1）

**Frame ordering.** Two alternating grid layouts are adopted. At even diffusion timesteps (Layout1), frames are packed chronologically into consecutive grids according to their original temporal order. At odd timesteps (Layout2), all grids slide forward by n columns. As a result, individual frames are re-distributed and may move into neighbouring grids across timesteps. This frame-reordering logic is applied synchronously during both DDIM inversion and denoising.

**Boundary handling.** For sequence boundaries introduced by grid sliding, partial grids are allowed and empty slots are filled following our tail-padding rule: replicated boundary frames participate in attention computation only as context and are discarded before decoding, so the output retains the original number K of video frames. For inter-grid boundaries, the sliding mechanism breaks the isolation between independent grids of original Grid Diffusion: frames originating from different grids are grouped together in alternate timesteps, enabling cross-grid self-attention for long-range temporal consistency.

> ⚠️ 待补：完整回复还应包含 Algorithm 1（见 Response 2.3）作为伪代码、tile/grid 分辨率定义（512×512 tile、grid 1024×1024 等），以及 matched Grid Diffusion baseline（静态网格、无 sliding）对比——即 Table 1 中 w/o grid sliding 行。

---

### Comment 6.2 — LLM 组件不可复现、未验证

**意见原文（要点）：**
LLM 组件既不可复现也未验证：缺少确切的 DeepSeek 模型、prompts、解码设置、输出 schema、分类体系与失败处理流程。请报告所有评估编辑对的路由准确率与混淆矩阵，并将 LLM 与确定性规则选择器、人工 oracle 对比。

**Response 6.2：** ❌ 未写回复（可复用 Response 2.2 的设置细节 + Table 2/3；oracle 已有，rule-based 仍缺）

---

### Comment 6.3 — 长视频核心声明无支撑

**意见原文（要点）：**
核心的长视频声明无支撑：定量评估止于 200 帧（按所述帧率约 7–10 秒），而描述的数据集含 1–3 分钟片段。请在 1 卡和 2 卡上评估完整片段，报告 GPU 与主机峰值内存、换入换出开销、运行时间、通信量与失败情况；否则将声明限制在 ≤200 帧。

**Response 6.3：** ❌ 未写回复（与 R2 2.5 / R3 3.7 / R5 5.2 同主题；此审稿人要求最具体：单卡+双卡、host memory、swapping、通信量、failures）

---

### Comment 6.4 — 定量结果无法可靠评估（缺评测清单）

**意见原文（要点）：**
定量结果无法可靠评估：视频数与编辑对数、任务构成、帧采样流程、背景 mask、聚合方式、随机种子、离散度与统计检验均未报告。请提供完整的评测清单（manifest）与分任务结果（含样本量、不确定性估计、配对置信区间或显著性检验）。

**Response 6.4：** ❌ 未写回复（相关材料：仓库中已有 rebuttal_reply_eval_manifest.md；统计部分可复用 Table 8–11）

---

### Comment 6.5 — 比较混淆了编辑类别与方法能力

**意见原文（要点）：**
比较混淆了编辑类别与能力不同的方法，而 UELVE 自适应选择 masking 与 fine-tuning；禁用基线支持的负向 prompt 不构成公平条件。请做类别特定的比较，使用匹配的 backbone、分辨率、prompt、控制输入、支持的选项与调参预算，并辅以针对编辑保真度与时间伪影的盲评人评。

**Response 6.5：** ❌ 未写回复（相关材料：仓库中已有 rebuttal_reply_fairness_blind_eval.md）

---

### Comment 6.6 — CLIP-T 定义矛盾（同 R2 Comment 2.1）

**意见原文（要点）：**
CLIP-T 被描述为平均距离，但数值越高被标为越好。请给出确切公式与模型 checkpoint，并按情况改用 "similarity" 或反转报告方向。

**Response 6.6：** ❌ 未写回复（可复用 Response 2.1；此处额外要求注明 CLIP 模型 checkpoint，需补）

---

### Comment 6.7 — 拼写与用词错误（"an Large"、"gird"、"GUP"、"mehtod"）；术语标准化

**意见原文（要点）：**
更正反复出现的语言与排版错误，包括 "an Large"、"gird"、"GUP"、"mehtod"；并统一 CLIP-T、Text-to-Image、long-form 等形式。

**Response 6.7：** ❌ 未写回复（可复用 Response 3.8，另需逐项确认四个拼写错误已改）

---

### Comment 6.8 — 图注主观措辞与信息缺失

**意见原文（要点）：**
将 "overall beauty" 等主观图注措辞替换为可测属性；并使每个定性图注标明 prompt、配置、grid size 与所选帧索引。

**Response 6.8：** ❌ 未写回复

---

## 待办汇总

### Reviewer 1 / 2
- [ ] **Response 2.4**：补全 Table 5 中 UELVE 四行（zero-shot / one-shot / LLM / CNet）的 GMem 与 Time 数据，并写正式回复（每行 GPU 数 + 峰值显存 + 含/不含 fine-tuning 的端到端时间）。
- [ ] **Response 2.5**：说明全长度片段 vs 窗口；如不补全长度实验，则收敛论文中的 "arbitrary-length" 声明并讨论磁盘 I/O 与累积漂移。
- [ ] **Table 3**：考虑补 rule-based lower bound（关键词规则路由）一行，直接回应 Reviewer 1 Comment 1.2 的要求（R5 5.1、R6 6.2 也要求）。
- [ ] **Response 2.3**：删除末尾残句 "as shown in Table 1, we performed a quantitative ablation study to"。
- [ ] **Response 1.5**：注明与 1.4 重复。
- [ ] Fig. 9(a)：按 Response 1.2 承诺修正图中 prompt pair。

### Reviewer 3
- [ ] **Response 3.2**：写正式回复（可复用 Response 1.1 的 Table 1，并承诺提供更清晰的视觉对比/放大图）。
- [ ] **Response 3.3**：长时序一致性证据（已有 rebuttal_reply_longterm_consistency.md 可参考；可补远隔帧对指标）。
- [ ] **Response 3.4**：承诺精简 Section 3.2（删除标准 DDPM/DDIM 背景，仅保留所需记号）。
- [ ] **Response 3.5**：补全 Table 7 大量空缺格子；核对 "284 071" 粘连数字及 Global stylization Setting3 与 Motion Editing Setting4 数值重复问题。
- [ ] **Response 3.6**：代码发布与实现细节（可结合 UELVE_LLM_Prompt_Specification.md）。
- [ ] **Response 3.7**：长视频实验（与 Response 2.5 联动）。
- [ ] **Response 3.8**：确认已更正 Fig. 5 "GUP"→"GPU" 及小字号问题。

### Reviewer 4
- [ ] **Response 4.1**：正文目前只有零散语句，需组织成完整回复（说明运行次数、t 检验方法等）。
- [ ] **Response 4.4**：摘要加入关键定量数字；引言压缩；Fig. 2 配色修改。Reviewer 4 段末的示意图图注为 "Figure 1: layout2"（Layout2 示意，与 Response 5.3 的 "Figure 2: layout3" 相对）。

### Reviewer 5
- [ ] **Response 5.1**：写正式回复（复用 Table 2/3 + 补 rule-based 对比）。
- [ ] **Response 5.2**：长视频实验（见下方"跨审稿人联动"）。
- [ ] **Response 5.3 / Table 16**：⚠️ 当前数值疑为占位符（与 Tune-A-Video/Video-P2P/FateZero 数值完全相同、Attention 与 ControlNet 行一致；PDF 中 Layout1 行还带着 "[TAV]" 引用标签），必须用真实 Layout1/2/3 实验数据重填。
- [ ] **Response 5.4**：内容已完整（Fig. 3 双公鸡案例 + future work）；但 **Fig. 3 图注是占位符 "Enter Caption"**，需替换。
- [ ] **Response 5.5**：✅ 已完整（PDF 核对确认）。
- [ ] **Response 5.6**：端到端成本分解（复用 Table 5）。
- [ ] **Response 5.7**：分任务结果（复用 Table 7）+ 统计信息（复用 Table 8–11）+ 收敛 "consistently superior" 类措辞。

### Reviewer 6
- [ ] **Response 6.1**：补 matched Grid Diffusion 基线说明 + 引用 Algorithm 1 + tile/grid 分辨率定义。
- [ ] **Response 6.2**：复用 Response 2.2 + Table 2/3；补输出 schema、分类体系、失败处理描述。
- [ ] **Response 6.3**：长视频实验（要求最具体：1/2 卡、GPU+host 内存、swapping、通信量、failures）。
- [ ] **Response 6.4**：完整评测 manifest（复用 rebuttal_reply_eval_manifest.md + Table 8–11 统计）。
- [ ] **Response 6.5**：类别特定公平比较 + 盲评（复用 rebuttal_reply_fairness_blind_eval.md）。
- [ ] **Response 6.6**：复用 Response 2.1，另补 CLIP checkpoint 型号。
- [ ] **Response 6.7**：复用 Response 3.8，逐项确认 "an Large"/"gird"/"GUP"/"mehtod" 已改。
- [ ] **Response 6.8**：替换 "overall beauty" 等主观措辞；图注补 prompt/配置/grid size/帧索引。

### 跨审稿人联动（同一主题、可共用一套实验/素材）
1. **长视频实验**（R2 2.5、R3 3.7、R5 5.2、R6 6.3）：至少补一组 >200 帧（分钟级）完整片段的质量 + 显存 + 时间 + 退化数据，否则统一收敛论文声明。
2. **LLM 路由定量评估**（R1 1.2、R2 2.2、R5 5.1、R6 6.2）：Table 2/3 已备，共同缺口是 **rule-based 下界**；R6 还要求 prompt/schema/失败处理全部披露。
3. **消融定量**（R1 1.1、R3 3.2、R4 4.2、R5 5.3、R6 6.1）：Table 1 已备；Table 7（分任务）与 Table 16（sliding 布局）数据需补齐/重跑。
4. **端到端效率公平性**（R2 2.4、R5 5.6）：Table 5 骨架已备，UELVE 行数据待补。
5. **统计报告**（R4 4.1、R5 5.7、R6 6.4）：Table 8–15 已备，需组织成正文并补评测 manifest。
6. **CLIP-T 定义**（R2 2.1、R6 6.6）：Response 2.1 已备，补 checkpoint。
7. **写作/图表规范**（R3 3.4、R3 3.8、R4 4.4、R6 6.7、R6 6.8）：统一做一次全文校对（"GUP"/"gird"/"mehtod"/"an Large"、图注措辞、Fig. 2 配色、Section 3.2 精简、摘要加定量）。

### 全局
- [ ] 核对各 "Response to Reviewer N" 段首的审稿人编号（模板均残留 "Reviewer 3"）。
- [ ] 回复信插图图注清一色占位符/无编号说明（"Figure 1: layout2"、"Figure 2: layout3"、"Figure 3: Enter Caption"），提交前统一替换为规范图注。
- [ ] 所有表格编号（Table 1–16）在正式回复信中统一重排。
- [ ] Meta Review Summary Comment 1 的五项要求（LLM 路由定量、grid/sliding 消融、长时序一致性、端到端效率、指标定义/实现细节/声明边界）应作为修订清单总纲逐一落实。
