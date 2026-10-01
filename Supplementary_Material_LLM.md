<!-- 附件材料 · LLM 部分（正式稿）。进度：§1 基础参数信息（已定稿）、§2 定量消融实验（已完成，含 Table S1 消融表 + Table S2 任务覆盖表）；§3 准确率评测实验、§4 跨 LLM 鲁棒性（待补）。注意：Table S1/S2 编号在附件全表汇总时需统一重排。提交前删除本注释。 -->

# Supplementary Material — Part LLM: The LLM Decision Module

This part supplements the LLM decision module described in Section 3.3 of the
main text. Section 1 provides the implementation details of the module,
including the model configuration, the prompt and output interface, the
inference cost, and the error-handling mechanism. Sections 2–4 report the
quantitative ablation, the routing-accuracy evaluation, and the robustness
analysis across different LLMs, respectively.

---

## Section 1. Implementation Details of the LLM Decision Module

The LLM serves as the core configuration router for our unified video editing pipeline. Taking the source prompt and the target prompt as inputs, the decision of model configuration as output.

Our routing module adopts DeepSeek-V4 as the default backbone LLM model, which is invoked through official API services after comprehensive trade-offs among routing accuracy, token consumption and practical accessibility. We set the model temperature as 0.2 to suppress random generation and ensure consistent outputs for identical input prompts, with a maximum token length of 2048 to fully accommodate all decision outputs. The pipeline enables streaming output and adopts a combined error handling strategy of request retry and conservative fallback to guarantee stable operation.

The routing prompt contents are provided in the supplementary materials. Specifically, considering that motion editing tasks relying on MotionEditor method pipeline which require implicit motion guidance maps that cannot be deduced from pure text prompts, we append the filename of the corresponding motion guidance map to the end of the target prompt as a unique engineering hint to help the LLM accurately identify motion editing tasks.

Statistical results show that the average single routing latency of our LLM module is only 0.51 seconds with an average token cost of 1934 tokens. Such time overhead is far lower than the minute-level running cost of the subsequent diffusion editing pipeline, which will not affect the overall operational efficiency of our method. 

---

## Section 2. Quantitative Ablation of the LLM Module

To isolate the contribution of the LLM module, we compare four settings that
share the identical grid-diffusion backbone and differ only in how the editing
configuration is determined: **Setting 1**, fixed zero-shot without mask and
fine-tuning; **Setting 2**, fixed one-shot with mask and temporal module
fine-tuning; **Setting 3**, full UELVE with LLM-guided adaptive configuration
selection; **Setting 4**, fixed one-shot with mask and Motion module
fine-tuning. Table S1 reports the four metrics on the five editing task
categories, and Table S2 summarizes which settings can accomplish each task
type.

**Table S1.** Task-wise quantitative ablation of the LLM module.

| Editing Task | Setting | Masked-PSNR ↑ | SSIM ↑ | CLIP-T ↑ | Tem-Con ↑ |
|---|---|---|---|---|---|
| Global Stylization | Setting1 | 18.45 | 0.6626 | 0.2939 | 0.9661 |
| | Setting2 | 18.33 | 0.6591 | 0.2936 | 0.9657 |
| | Setting3 | 18.46 | 0.6625 | 0.2938 | 0.9661 |
| | Setting4 | — | — | — | — |
| Local Attribute Modification | Setting1 | 16.79 | 0.4652 | 0.2917 | 0.9328 |
| | Setting2 | 22.31 | 0.6530 | 0.2906 | 0.9593 |
| | Setting3 | 22.38 | 0.6537 | 0.2909 | 0.9588 |
| | Setting4 | — | — | — | — |
| Complex Structure Change | Setting1 | 16.58 | 0.4641 | 0.2915 | 0.9326 |
| | Setting2 | 22.07 | 0.6184 | 0.2902 | 0.9551 |
| | Setting3 | 22.09 | 0.6182 | 0.2903 | 0.9552 |
| | Setting4 | — | — | — | — |
| Motion Preservation | Setting1 | 16.49 | 0.4466 | 0.2913 | 0.9320 |
| | Setting2 | 22.25 | 0.6225 | 0.2916 | 0.9571 |
| | Setting3 | 22.24 | 0.6223 | 0.2917 | 0.9573 |
| | Setting4 | — | — | — | — |
| Motion Editing | Setting1 | — | — | — | — |
| | Setting2 | — | — | — | — |
| | Setting3 | 22.16 | 0.6213 | 0.2893 | 0.9624 |
| | Setting4 | 22.15 | 0.6214 | 0.2892 | 0.9622 |

"−" denotes that the setting cannot accomplish the corresponding task: Setting 4
requires a motion guidance map, which only motion editing tasks provide, and
motion editing cannot be accomplished without the Motion module under Settings 1
and 2.

**Table S2.** Task coverage of each setting (✓: accomplished; ×: failed).

| Editing Task | Setting1 | Setting2 | Setting3 |
|---|---|---|---|
| Local Attribute Modification | × | ✓ | ✓ |
| Complex Structure Change | × | ✓ | ✓ |
| Global Stylization | ✓ | × | ✓ |
| Motion Preservation | × | ✓ | ✓ |
| Motion Editing | × | × | ✓ |

Setting 1 removes the masking mechanism and fine-tuning. It is effective only
for global stylization, where its configuration coincides with the route
selected by the LLM, and accordingly its metrics are on par with Setting 3
(e.g., 18.45 vs. 18.46 in Masked-PSNR). For the local editing tasks, the edit
leaks outside the target subjects and corrupts the background, degrading
Masked-PSNR to 16.49–16.79, SSIM to below 0.47, and Tem-Con to about 0.932.
We note that the CLIP-T values of Setting 1 on these tasks are slightly higher
than those of the masked settings, since the unconstrained edit alters the
whole frame and therefore aligns marginally better with the target prompt; this
does not indicate superior editing quality.

Setting 2 enables masking and temporal fine-tuning for every task. It achieves
metrics comparable to Setting 3 on the local editing tasks where its
configuration matches the LLM-selected route. However, such a fixed strategy
pays unnecessary costs and even hurts quality in the remaining cases. For
global stylization, the forcibly applied mask, obtained by interpolating the
cross-attention control map, cannot perfectly isolate the editing region, and
the fine-tuning shifts the overall tone, which slightly reduces Masked-PSNR
(18.33 vs. 18.45). For local attribute modification, fine-tuning is necessary
only when the edited subject is in motion, whereas static subjects can be
modified in the zero-shot manner; Setting 2 fine-tunes even for static-subject
cases, wasting the training cost without quality gain. In contrast, Setting 3
routes such cases to the zero-shot branch automatically and matches the best
fixed configuration on every task category.

Motion editing requires fine-tuning the Motion module under the guidance map,
so only Settings 3 and 4 accomplish it, with nearly identical metrics, and the
LLM module is the only setting that covers all five task categories
(Table S2). Together, the ablation shows that the primary value of the LLM
module lies in selecting the appropriate configuration for each task: it
attains the best per-task trade-off between editing quality and computational
cost without any manual tuning, whereas each fixed configuration either fails
entire task categories or spends unnecessary fine-tuning and masking overhead.

## Section 3. Routing-Accuracy Evaluation

*(to be supplemented)*

## Section 4. Robustness Across LLMs

*(to be supplemented)*
