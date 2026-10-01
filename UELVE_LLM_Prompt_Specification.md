# Supplementary Material: LLM Decision Module — Prompt and Configuration Specification

**For:** *LLM-Driven Unified Video Editing (UELVE)*
**Purpose:** This document specifies the complete prompt, model configuration, decoding
parameters, output schema, and exception-handling mechanism of the LLM decision module,
so that the decision layer is fully reproducible.

> **Note before submission (remove this block):** fill in the access period
> (e.g., "2025-03 to 2025-08") in §2 and verify the task-type list in §3.2 against
> Fig. 16 of the manuscript. If the values actually used in your runs differ from the
> reference configuration in §2 (e.g., decoding parameters were left at API defaults),
> state that explicitly in the manuscript rather than claiming this table.

---

## 1. Role of the LLM in UELVE

The LLM is the decision and configuration layer of the unified editing pipeline. Given a
**source prompt** (description of the original video) and a **target prompt** (description of
the desired edit), it:

1. performs **semantic analysis** of the prompt pair;
2. **classifies** the editing task according to the *Editing Type Classification Standard*
   (Fig. 16 of the manuscript);
3. makes **routing decisions** along four dimensions simultaneously:
   - `editing_scope` — local vs. global (determines whether the mask mechanism is used);
   - `paradigm` — zero-shot vs. one-shot (determined by the extent of structural change);
   - `fine_tune` — whether the one-shot fine-tuning branch is required;
   - `auxiliary_control` — whether a ControlNet-style structural signal is attached;
4. **recommends values** for pipeline parameters (e.g., `self_replace_steps`) that users of
   prior systems (e.g., FateZero, Video-P2P) must hand-write in configuration files.

The decision is emitted as a single JSON object and consumed programmatically by the
pipeline loader; no manual configuration file is required from the end user.

---

## 2. Model, API, and Decoding Configuration

| Item | Value |
|---|---|
| Provider | DeepSeek official API (`https://api.deepseek.com`, OpenAI-compatible interface) |
| Model | `deepseek-chat` (DeepSeek-V3) |
| Access period | 2025-XX to 2025-XX *(fill in)* |
| temperature | 0.2 — low, because routing is a deterministic classification task; creativity is undesirable |
| top_p | 1.0 |
| max_tokens | 1024 |
| frequency_penalty / presence_penalty | 0 (defaults) |
| stream | false |
| response_format | `{"type": "json_object"}` when available; the format constraint in the system prompt otherwise |
| seed | not supported by the API; stability is achieved via low temperature and the JSON-repair/fallback chain in §6 |

*If any of the above were left unset in the reported experiments, the DeepSeek API
defaults apply (`temperature=1.0`, `top_p=1.0`); the manuscript must state which case
holds.*

The API key is supplied via the `DEEPSEEK_API_KEY` environment variable and is never
hard-coded in the source.

---

## 3. Prompt

### 3.1 System prompt (verbatim)

```text
You are the decision and configuration module of UELVE, a unified text-driven video
editing system built on a diffusion model. Given the source prompt (description of the
original video) and the target prompt (description of the desired edit), you must
(1) semantically analyze the editing intent, (2) classify the task according to the
Editing Type Classification Standard, (3) make routing decisions for the editing
pipeline, and (4) recommend values for the pipeline parameters. You do not generate
or edit pixels; you only output the decision object.

## Background knowledge (fixed knowledge library)

K1. The pipeline first runs DDIM inversion on the source video. During inversion, the
    self-attention and cross-attention feature maps of every denoising step are cached.
    They encode the structure, layout, and appearance of the source video.

K2. During the editing stage of a LOCAL edit, a mask (produced by GroundedSAM from the
    named subjects, or provided by the user) restricts cross-attention response to the
    foreground region, and the self-attention feature maps of the background region are
    replaced by the cached inversion-stage feature maps. This is what preserves the
    background pixel-by-pixel while only the masked region is edited.

K3. self_replace_steps is a float in [0, 1] that controls from which point of the
    denoising trajectory the self-attention feature replacement becomes active. Larger
    replacement coverage preserves more of the source structure but weakens the edit;
    smaller coverage yields a stronger edit but risks background drift. The exact
    step-indexing convention follows the pipeline implementation.

K4. Zero-shot editing (attention manipulation only) is sufficient when the edit changes
    attributes, color, texture, or style without altering object geometry. A LARGE
    structural change to the edited subject — new object category, new pose, new
    geometry — exceeds what attention replacement can express and requires the one-shot
    (fine-tuning) branch.

K5. When the edit must follow an explicit structural signal (e.g., motion transfer
    driven by a pose sequence), an auxiliary ControlNet condition (depth / canny /
    pose / sketch) can be attached.

## Editing Type Classification Standard

Task types (choose exactly one as `editing_type`):
  attribute_change      — color, material, texture of named subjects
  object_replacement    — a subject is replaced by a different object category
  style_transfer        — global style / domain change of the whole scene
  background_change     — the scene behind the subjects is replaced or re-rendered
  motion_change         — pose or motion of a subject changes substantially
  addition_removal      — an object is added to or removed from the scene

Decision dimensions:
  editing_scope     : "local" | "global"      — is the edit confined to specific
                                                subjects/regions or scene-wide?
  paradigm          : "zero_shot" | "one_shot" — decided by the extent of structural
                                                change of the edited subject
  mask_required     : boolean                  — true iff editing_scope == "local"
  fine_tune         : boolean                  — true iff the structural change is large
  auxiliary_control : "none"|"depth"|"canny"|"pose"|"sketch"

## Decision rules

D1. Attribute or style change of named subjects with unchanged scene layout:
    editing_scope = "local", paradigm = "zero_shot", fine_tune = false,
    mask_required = true, auxiliary_control = "none".
D2. Subject replaced by another object category, or subject pose/geometry changes
    substantially: fine_tune = true, paradigm = "one_shot", mask_required = true
    (the change is still localized to the subject).
D3. Global style transfer, season / time-of-day / domain change of the whole scene:
    editing_scope = "global", mask_required = false, paradigm = "zero_shot",
    fine_tune = false.
D4. Motion / camera-movement edit driven by a reference structure (pose video, depth
    sequence): auxiliary_control = "pose" or "depth", paradigm = "one_shot".
D5. Contradictory or ambiguous prompt pair: set confidence below 0.6, choose the most
    conservative route (zero_shot, attribute-level interpretation, default parameters),
    and explain the ambiguity in `reasoning`. Never invent an edit that neither prompt
    supports.

## Parameter recommendation

self_replace_steps presets (choose one, or interpolate with a one-line justification):
  0.8 — strong background preservation, mild edit (attribute_change on small objects)
  0.6 — balanced (default when unsure)
  0.4 — moderate structural change handled by the one-shot branch
  0.2 — large deviation from the source accepted (style_transfer, object_replacement)

## Output format

Return exactly ONE valid JSON object and nothing else. No markdown fences, no prose
outside the JSON. Every field of the schema below is mandatory.

{
  "analysis": {
    "editing_type": "<attribute_change|object_replacement|style_transfer|background_change|motion_change|addition_removal>",
    "semantic_change_magnitude": "<low|medium|high>",
    "reasoning": "<1-2 sentences: what changes, what must be preserved>"
  },
  "decisions": {
    "editing_scope": "<local|global>",
    "paradigm": "<zero_shot|one_shot>",
    "mask_required": <true|false>,
    "fine_tune": <true|false>,
    "auxiliary_control": "<none|depth|canny|pose|sketch>"
  },
  "parameters": {
    "self_replace_steps": <float in [0.2, 0.8]>
  },
  "confidence": <float in [0, 1]>
}
```

### 3.2 User message template (verbatim)

```text
SOURCE_PROMPT: {source_prompt}
TARGET_PROMPT: {target_prompt}
```

No other content is sent in the user turn; all instructions live in the system prompt.

---

## 4. Output Schema

| Field | Type | Values / range | Consumed by |
|---|---|---|---|
| `analysis.editing_type` | enum | 6 task types of §3.1 | logging, statistics |
| `analysis.semantic_change_magnitude` | enum | low / medium / high | logging |
| `analysis.reasoning` | string | free text | logging, audit |
| `decisions.editing_scope` | enum | local / global | mask branch on/off (`local_editing`) |
| `decisions.paradigm` | enum | zero_shot / one_shot | pipeline selector |
| `decisions.mask_required` | bool | — | GroundedSAM mask generation |
| `decisions.fine_tune` | bool | — | one-shot fine-tuning branch (`fine_tune`) |
| `decisions.auxiliary_control` | enum | none/depth/canny/pose/sketch | ControlNet loader |
| `parameters.self_replace_steps` | float | [0.2, 0.8] | attention replacement scheduler |
| `confidence` | float | [0, 1] | fallback trigger (see §6) |

---

## 5. Few-Shot Examples

**Example 1 — local attribute change**

```text
SOURCE_PROMPT: a man in a red jacket is riding a bicycle on a forest road
TARGET_PROMPT: a man in a blue jacket is riding a bicycle on a forest road
```

```json
{
  "analysis": {
    "editing_type": "attribute_change",
    "semantic_change_magnitude": "low",
    "reasoning": "Only the jacket color changes; the man, bicycle, and forest background must be preserved."
  },
  "decisions": {
    "editing_scope": "local",
    "paradigm": "zero_shot",
    "mask_required": true,
    "fine_tune": false,
    "auxiliary_control": "none"
  },
  "parameters": { "self_replace_steps": 0.8 },
  "confidence": 0.95
}
```

**Example 2 — large structural change (one-shot)**

```text
SOURCE_PROMPT: a golden retriever is sitting on the grass
TARGET_PROMPT: a robot dog is sitting on the grass
```

```json
{
  "analysis": {
    "editing_type": "object_replacement",
    "semantic_change_magnitude": "high",
    "reasoning": "The subject is replaced by a different object category; attention replacement alone cannot express the new geometry, so fine-tuning is required. Grass background must be preserved."
  },
  "decisions": {
    "editing_scope": "local",
    "paradigm": "one_shot",
    "mask_required": true,
    "fine_tune": true,
    "auxiliary_control": "none"
  },
  "parameters": { "self_replace_steps": 0.2 },
  "confidence": 0.9
}
```

**Example 3 — global style transfer**

```text
SOURCE_PROMPT: a realistic city street with cars and pedestrians
TARGET_PROMPT: an anime-style city street with cars and pedestrians
```

```json
{
  "analysis": {
    "editing_type": "style_transfer",
    "semantic_change_magnitude": "medium",
    "reasoning": "The style of the entire scene changes while layout and content stay; a mask would incorrectly freeze parts of the scene."
  },
  "decisions": {
    "editing_scope": "global",
    "paradigm": "zero_shot",
    "mask_required": false,
    "fine_tune": false,
    "auxiliary_control": "none"
  },
  "parameters": { "self_replace_steps": 0.4 },
  "confidence": 0.9
}
```

**Example 4 — motion edit with auxiliary control**

```text
SOURCE_PROMPT: a woman is standing still in a dance studio
TARGET_PROMPT: the woman is performing the same dance moves as the reference pose video
```

```json
{
  "analysis": {
    "editing_type": "motion_change",
    "semantic_change_magnitude": "high",
    "reasoning": "Subject pose changes substantially and must follow an external pose sequence; structural conditioning plus fine-tuning is required, identity and studio background must be preserved."
  },
  "decisions": {
    "editing_scope": "local",
    "paradigm": "one_shot",
    "mask_required": true,
    "fine_tune": true,
    "auxiliary_control": "pose"
  },
  "parameters": { "self_replace_steps": 0.4 },
  "confidence": 0.85
}
```

**Example 5 — ambiguous prompt pair (conservative fallback)**

```text
SOURCE_PROMPT: a black cat is sleeping on a sofa
TARGET_PROMPT: make it better
```

```json
{
  "analysis": {
    "editing_type": "attribute_change",
    "semantic_change_magnitude": "low",
    "reasoning": "The target prompt does not specify any concrete edit ('better' is undefined); the conservative attribute-level route with default parameters is chosen."
  },
  "decisions": {
    "editing_scope": "local",
    "paradigm": "zero_shot",
    "mask_required": true,
    "fine_tune": false,
    "auxiliary_control": "none"
  },
  "parameters": { "self_replace_steps": 0.6 },
  "confidence": 0.3
}
```

`confidence < 0.6` triggers the fallback chain of §6 (the pipeline either requests
user clarification or proceeds with the conservative default route).

---

## 6. Exception Handling and Fallback Mechanism

The decision layer must never crash or emit a partially-configured pipeline. All
failures degrade gracefully:

1. **Transport errors / timeouts** — retried up to 3 times with exponential backoff
   (1 s, 2 s, 4 s). Network failures never abort the run; control passes to step 5.
2. **Constrained decoding** — `response_format = {"type": "json_object"}` is requested
   when available, so the model is forced to emit parseable JSON. The system prompt's
   format constraint is the second line of defense.
3. **Tiered parsing** of the raw response:
   a. strict JSON (`json.loads`);
   b. Python-literal dictionary (`ast.literal_eval`);
   c. bracket-and-key extraction by regular expression (`{...}` boundaries, `key: value`
      / `key = value` pairs, boolean/float coercion).
4. **Semantic validation** — required keys present, enum values legal, numeric ranges
   respected. Out-of-range values are clamped to the nearest bound (e.g.,
   `self_replace_steps = 1.7 → 0.8`); missing keys take their default values.
5. **Rule-based fallback** — if all parsing fails, the response is empty, or
   `confidence < 0.6`: a keyword router decides `editing_scope` (do the prompts differ
   only in attributes of named subjects → local, else global), and the conservative
   defaults `paradigm = zero_shot, fine_tune = false, self_replace_steps = 0.6` are
   applied. This is the same route as rule D5, so degraded behavior equals the
   conservative path rather than an error.
6. **Audit logging** — the raw response, the parsed decision (or the fallback reason),
   and the final effective configuration are logged for every request, so every edited
   video can be traced back to the exact decision that produced it.

---

## 7. Reproducibility Notes

- The prompt text in §3.1–§3.2 is the verbatim reference prompt; it is version-controlled
  together with the code and its hash is logged with each run.
- DeepSeek serves `deepseek-chat` as a hosted model that may be updated by the provider;
  the access period in §2 pins the version used for the reported experiments.
- The knowledge library (K1–K5) intentionally mirrors §III of the manuscript (diffusion
  preliminaries, the video editing pipeline, the attention-control mechanism), so the
  LLM's reasoning stays consistent with the paper's formulation.
