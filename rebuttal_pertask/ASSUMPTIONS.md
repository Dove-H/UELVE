# 返修 per-task 表 + paired CI 表: 全部假设与生成逻辑(务必先读)

> ⚠️ 本目录所有数值为**分析基线(合成)**: 以论文 Table 1 实测总均值为锚点、
> 按各方法已知特性拆分到 7 个子任务, 用于核对真实重跑结果的方向/量级/显著性。
> 不可直接粘贴进返修稿。真实数据到位后, 替换本目录脚本的 ANCHOR/NPAIRS/STD/RHO
> 或直接用逐视频 CSV 重算。

## 1. 生成管线(gen_rebuttal_tables.py, 可复现, 两次运行逐字节一致)

1. **锚点**: Table 1 每个 (method × metric × frame) 总均值为硬约束
   (脚本内 ANCHOR, 逐字来自论文; CAV 16f PSNR 论文印 9.715, 取 9.72)。
2. **拆分**: per-task 值 = anchor + task_offset + 常量平移(平移使按 n 加权
   的均值**精确**回到 anchor)。verify_anchors() 打印每个锚点的还原误差,
   当前最大 drift ≤ 0.005 (PSNR 2 位小数取整所致), 全部通过。
   => 核对方法: 对任一 method×metric×frame, 按 NPAIRS 权重平均 7 个
   子任务值, 应等于 Table 1 原值。
3. **task_offset 的方法学依据**(见脚本 OFF 注释):
   - 掩码/注意力类 (Video-P2P/FateZero/TokenFlow/AdaFlow/UELVE-Attn):
     LAM/BR/OR 的 Masked-PSNR 高; CSC 弱(编辑泄漏)。
   - ControlNet 无掩码类 (RAVE/ControlVideo/CAV/GLFM/UELVE-CNet):
     BR/OR 背景易损 → PSNR 低; RAVE 16f+ 因混洗额外 −0.3~−0.6。
   - Tune-A-Video: 微调覆写 → PSNR 13~16 全面崩; GS 渲染尚可(Clip-T 偏高)。
   - TokenFlow/AdaFlow: 特征传播 → SSIM/Tem-Con 强、Clip-T 弱;
     AdaFlow 200f 剪枝 → 各项额外掉点(对应正文 "四项全差")。
   - GS: 强渲染(RAVE/GLFM/U-CNet) → PSNR 低 Clip-T 高; 欠渲染(VP2P/FZ/U-Attn)
     → PSNR 高(虚高) Clip-T 低。MP: 平凡保持, 各方法 PSNR 都高。
4. **ME 口径(2026-09-28 用户指定, 取消 N/A)**:
   - 所有 baseline 与 UELVE-Attention **无法执行** Motion Editing(无时序编辑
     模块) => 输出≈源视频: ME-PSNR 高(~24.5, 掩码区域未变)、SSIM 虚高、
     Clip-T 低。
   - **只有 UELVE-ControlNet 能做 ME**: ME 行 = 成功编辑剖面, PSNR 低
     (−1.5 offset, 区域真的变了) + Clip-T 高(+0.010)。
   - ME-PSNR 的"高=没编辑动"病理与 GS-PSNR 同类, 已写入两张表的 Notes。
5. **std**: 方法级基准(STD_BASE) × 帧数乘子(1.0/1.05/1.12/1.22) × 任务
   确定性抖动 ±10% (det_hash/crc32, **不用内建 hash**——后者有
   PYTHONHASHSEED 随机盐, 曾导致每次重跑 std 漂移)。
6. **配对样本**: 每 (frame,task,metric) 生成 n 个逐视频值, 仿射校正到与表中
   mean/std 完全一致; 跨方法共享"视频难度"因子 z0, 配对相关
   rho = 0.70 (PSNR/SSIM) / 0.60 (Clip-T/Tem-Con)。
7. **统计**: Δ=UELVE−最强基线; 95% CI = 10k 百分位 bootstrap; 配对 t;
   Wilcoxon 符号秩; Holm 校正族 = 同一 (frame, sub-task) 内全部指标
   (PSNR 现已覆盖全部 7 任务 → 共 112 组比较)。

## 2. 关键口径决定(与用户 9-27 模板不同处)

- **Clip-T 方向 = ↑(高更好)**, 模板写 ↓ 是笔误: 论文 Table 1 表头即 Clip-T↑;
  正文 "AdaFlow 200f 四项全差"(其 0.2708 为最低)、"GLFM favorable on SSIM
  and CLIP-T"(其 Clip-T 偏高) 只有 ↑ 才自洽。
- **Tune-A-Video 48f 有值**(15.13/0.4418/0.2894/0.9441), 仅 200f 为 ---。
  FZ 只到 8f; VP2P/CV 到 16f; CAV 到 48f。
- **ROUTING(配对表中 UELVE 变体)**: LAM/BR/OR/MP→UELVE-Attention;
  GS/CSC/**ME**→UELVE-ControlNet(ME 改路由: 只有 C 变体能做 ME)。
- **排版(2026-09-28 二次返修后)**: caption 在正常字号环境下执行(修复 8pt
  小标签"Table N"), 短句一行; 详细说明移到表末 Notes 行; 表体 8pt 用逐单元
  格 \tblf(单元格是分组, 环境级字号无法穿透, 也不能在 caption 前设小字);
  \LTcapwidth=\textwidth + \LTleft/\LTright=0pt + @{\extracolsep{\fill}}
  撑满版心; 面板头 "(total n=96; GS 14, ...)" 不写 "per sub-task"(防超宽)。
  ⚠️ 本机无 texlive 宏包, 排版未经编译实测。
- 方法顺序 = Table 1 原顺序; 不可用方法在面板内以单行 `Method & ---` 表示。

## 3. 占位假设(真实数据必须替换的)

- **NPAIRS(每帧×子任务编辑对数)**: 8/16f: GS14 LAM21 BR12 OR15 MP10 ME10 CSC14
  (共 96); 48f: 11/17/10/12/8/8/10 (共 76, 剔除 VE-Bench 32 帧定长序列);
  200f: 6/8/5/6/6/6/5 (共 42, 仅长视频子集)。=> 必须用真实测试集统计(表 R10)。
- **最强基线的选择**: 逐 (frame, sub-task, metric) 取有数据基线中的最高均值
  (最保守口径, 对 UELVE 最不利)。改法见 paired_stats() 中 best 的选取。
- rho / STD_BASE 量级: 典型配对视频编辑评测的经验值。

## 4. 汇总(当前合成数据的显著性面貌, 112 组比较)

- UELVE 显著优 15 / 显著劣 35 / n.s. 62 (Holm 校正后)。
  8f 6/14/8, 16f 3/9/16, 48f 4/8/16, 200f 2/4/22。
- 劣势模式(Table 1 锚点的直接推论, 部分为指标病理而非方法劣势):
  1. **ME-PSNR / ME-SSIM 输给"没编辑"的 baseline**(FZ 等 ME-PSNR≈24.7 虚高)
     —— 病理: 掩码覆盖必须改变的区域, PSNR/SSIM 奖励不编辑; Notes 已声明;
  2. **GS-PSNR 同理**(U-CNet 强渲染 19.6 vs 欠编辑 ~24.9);
  3. Tem-Con 8~48f 输 RAVE/AdaFlow/TokenFlow(U-Attn 0.9653 vs RAVE 0.9795);
  4. GS SSIM 输欠编辑方法(输出≈源视频, SSIM 病理性偏高)。
- 优势模式: LAM/BR/OR 的 SSIM/PSNR(全帧长)、GS Clip-T(U-CNet)、
  **ME Clip-T 方向为正**(8f +0.017 / 16f +0.026, 但 n=10 且 std≈0.03 下
  Holm 后不显著 —— 叙事提示: 真实数据若 ME Clip-T 优势更大或 n 更大即显著;
  这是"只有 U-CNet 能做 ME"主张的量化证据点)、48/200f 长视频各项
  (TokenFlow 是 200f 最强对手)。

## 5. 文件

- `gen_rebuttal_tables.py` — 生成器(改 ANCHOR/NPAIRS 后重跑即可)
- `table_S_per_task.tex` — 表 S2: per-task longtable(4 面板 8/16/48/200f,
  11 方法 × 7 子任务, mean±std, 面板内每列最优加粗, 表末 Notes)
- `table_S_paired_ci.tex` — 表 S3: paired CI longtable(Δ/95%CI/p_t/p_W/
  Holm/星号, 112 组)
- `per_task_metrics.csv` / `paired_stats.csv` — 机读版
- 运行: `python3 gen_rebuttal_tables.py`(numpy+scipy)
