# Supplementary Material — Part LLM: The LLM Decision Module

> 附件材料 LLM 部分共四个小节：
> 1. **LLM 基础参数信息**（本节，已完成）
> 2. LLM 模块的定量消融实验（待写：Table 1 指标消融、Table 6 任务可完成性、Table 7 分任务消融）
> 3. LLM 准确率评测实验（待写：Table 2 混淆矩阵，500 对人工标注；含 rule-based 下界缺口）
> 4. LLM 模块对不同 LLM 的鲁棒性（待写：Table 3，六个 LLM + oracle）
>
> 各审稿意见的回复按需引用对应小节，避免重复展开。全部内容以回复信（Response to PR reviews）中已写明的事实为依据。

---

## Section 1. LLM 基础参数信息（Basic Information and Configuration）

Six reviewers raised questions about the LLM decision module. This section
consolidates the **basic parameters and configuration** of the LLM router — its
role, model identity, decoding settings, prompt, routing taxonomy, per-request
inference cost, and error handling — into a single authoritative description that
all itemized responses cite. Routing-accuracy evaluation, ablation experiments,
and cross-LLM robustness are reported in Sections 2–4 of this Part.

### 1.1 Role and Scope of the LLM in UELVE

The LLM is the **decision and configuration layer** of the unified editing
pipeline; it is not a generative component.

- **Input.** Exactly two strings: the source prompt (description of the original
  video) and the target prompt (description of the desired edit). All editing
  intent is derived purely from the semantic difference between the two prompts;
  the LLM never observes video frames or any visual signal.
- **Function.** The LLM acts as a decision-making agent: it performs text-level
  comparison of the prompt pair and selects among the existing pipeline options —
  the masking mechanism, temporal / motion fine-tuning (zero-shot vs. one-shot),
  the attention-control editing path, and auxiliary ControlNet conditioning.
  Parameters that users of prior systems must hand-write in configuration files
  are thereby determined automatically.
- **Output.** A structured routing decision consumed programmatically by the
  pipeline loader; no manual configuration file is required from the end user.

The routing decision is obtained from a single LLM inference issued before the
diffusion pipeline starts (average latency 0.51 s, see §1.6), and is then fixed
for the whole run.

Two framing points are worth restating:

1. **The primary role of the LLM is unification, not metric gain.** Improved
   editing quality is a side effect of the LLM module; its essential value is
   automatically adapting the model configuration to diverse editing tasks,
   which fixed configurations cannot cover (quantified in Section 2).
2. **Why a text-only LLM rather than a vision-language model.** The text-driven
   editing task derives all editing intent from prompt semantics, so visual
   understanding is unnecessary for the core workflow; a VLM would introduce
   extra visual-encoding overhead and potential visual noise that may perturb
   decisions. Integrating VLMs for more complex multi-modal editing is noted as
   a future extension.

### 1.2 Model Identity

| Item | Value |
|---|---|
| Default LLM | **DeepSeek-V4** |
| Role in pipeline | decision / configuration agent (single inference per editing task) |
| Switchability | the router can be rebuilt on any other chat LLM by replacing the model (quantified in Section 4) |

DeepSeek-V4 was selected as the default after comprehensively weighing routing
accuracy, token cost, and API usability. Readers can freely switch to more
powerful LLMs to build the router for higher routing accuracy.

### 1.3 Decoding Settings

| Parameter | Value |
|---|---|
| `temperature` | **0.2** — low temperature for a deterministic classification task; creativity is undesirable |
| `max_tokens` | **2048** |
| `stream` | **true** — streaming output enabled |
| Error handling | retry mechanism with conservative default fallback (§1.7) |

### 1.4 Prompt

The **complete prompt is provided in the supplementary material**; this section
records the one convention that reviewers specifically asked about.

**Motion-editing hint convention (explains Fig. 9(a)).** The motion editing
module is adapted from MotionEditor and requires an additional motion guidance
map to guide the generation of new motions — something that cannot be inferred
from text alone. To enable the LLM to correctly recognize a motion editing task,
the implementation appends the filename of the motion guidance map to the target
prompt as a hint. The actual prompt pair for the motion editing case of Fig. 9(a)
is therefore:

```text
source: a girl is dancing
target: a girl is dancing (/<motion guidance map filename>)
```

This is an engineering trick of the implementation; the figure will be corrected
accordingly in the revised manuscript.

### 1.5 Routing Taxonomy and Decision Classes

The router classifies each prompt pair along three routing dimensions, which
produce the five decision classes evaluated in Section 3:

| Dimension | Decision classes | Pipeline consequence |
|---|---|---|
| Masking | **Local Editing (need mask)** / **Global Editing (no mask)** | whether the mask mechanism restricts the edit to the foreground region |
| Fine-tuning | **Simple Editing (no fine-tune)** / **Complex Editing (need fine-tune)** | zero-shot attention control vs. one-shot fine-tuning of the temporal module |
| Motion | **Motion Editing (need motion tune)** | one-shot fine-tuning of the Motion module, driven by the motion guidance map |

Each prompt pair carries coupled labels — one from the masking pair and one from
the fine-tuning triple — so the evaluation protocol covers both axes over the
same prompt set (Section 3).

**Mask generation (when the router enables masking).** FastSAM is class-agnostic;
a CLIP model maps the modified noun phrase to the target segmented region. Masks
are generated for each individual frame and bound to frame indices, so no
propagation across frames, grids, or the two alternating grid layouts is
required. The specific masking scheme is not critical to the method — replacing
FastSAM with an alternative mask generator of equivalent performance does not
yield obvious metric variations.

### 1.6 Per-Request Inference Cost

| Metric | Value |
|---|---|
| Average single-inference latency | **0.51 s** |
| Average token cost per request | **1934 tokens** |
| Inferences per editing task | 1 |

Routing is thus a one-off, sub-second cost per video, negligible against the
minutes-scale diffusion runtime (cf. the efficiency tables of the response
letter).

### 1.7 Error Handling

- **Retry mechanism.** If the model output cannot be parsed, the request is
  retried up to the maximum number of retries.
- **Conservative fallback.** If the output is still unparseable after the maximum
  number of retries, the system falls back to a conservative default
  configuration rather than aborting the run.

### 1.8 本节覆盖的审稿问题（基础信息部分）

| Requested item | Asked in | Answered in |
|---|---|---|
| Exact model / version | C2.2, C6.2 | §1.2 |
| Decoding settings | C2.2, C6.2 | §1.3 |
| Complete prompt (+ motion-map hint) | C2.2, C6.2, C1.2 (Fig. 9a) | §1.4 |
| Routing taxonomy / decision classes | C6.2 | §1.5 |
| Failure-handling procedure | C2.2, C6.2 | §1.7 |
| Router latency and token cost | C1.2 | §1.6 |

准确率（含 oracle 上界与 rule-based 下界）、消融、跨 LLM 鲁棒性三类问题分别由第 3、2、4 节回答。
