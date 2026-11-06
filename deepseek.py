# LLM 编辑类型决策模块（DeepSeek）
# 依据论文 "LLM-Driven Unified Video Editing (UELVE)" 的决策模块设计：
# 给定 source prompt 与 target prompt，由 LLM 完成语义分析、编辑类型分类，
# 并输出结构化的 "model_adaption"（模型适配方案摘要）与 "Thinking"（四步推理）。
# "model_adaption" 前置输出：流式场景下管线无需等 Thinking 生成完即可先拿到
# 该字段并提前决定配置；同时保留 local_editing / fine_tune / self_replace_steps 供管线读取。
#
# 用法:
#   export DEEPSEEK_API_KEY=sk-xxx
#   python deepseek.py

import os
import json
import ast
import re
import time

from openai import OpenAI

# ============================================================
# 1. 角色定位 (Role)
# ============================================================
PROMPT_ROLE = """\
You are the decision assistant of a unified text-driven video editing system built on a diffusion model.
Given the source prompt (description of the original video) and the target prompt (description of the desired edit).
You do not generate or edit pixels; you only output the decision object."""

# ============================================================
# 2. 分析推理原则 (背景知识 + 分类标准 + 决策规则)
# ============================================================
PROMPT_PRINCIPLES = """\


## Editing Type Classification Standard

Editing scope:
  Local Editing  — the edit is confined to specific subjects/regions of the frames
  Global Editing — the edit applies scene-wide 

Editing types:
  Simple editing -  the contour change is small AND the edited object is static.
  Complex editing - the contour change is large, OR the edited object is in motion.
  Motion editing -  only the filename of a motion guidance map is attached to the target prompt.

## Decision rules

D1. Mask necessity: a mask is needed if the edit is LOCAL, i.e. the
    edited (replaced / modified) object occupies only a part of the frames.
    If the edit is GLOBAL (scene-wide), no mask is needed.

D2. Fine-tuning necessity (temporal module): focus on TWO criteria:
    (a) LARGE structural change — the contour or structure of the edited object
        differs greatly between source and target (e.g., "a man is cutting a
        watermelon" -> "a man is cutting a pear": their shapes differ greatly);
    (b) Moving object — the edited/replaced object is in motion (e.g., "a dog
        is running on the grass" -> "a cat is running on the grass").
    If NEITHER holds — the contour barely changes AND the object is static —
    zero-shot attention manipulation is sufficient and no fine-tuning is needed
    (e.g., "a apple on the table" -> "a pear on the table": the contour difference is small and the edited
    objects are static).
    To judge whether the contour change is large or small, compare the overall
    silhouettes of the source and target objects: two similar-sized animals of
    the same kind, or two objects of the same shape family (round fruits, cups
    and mugs, boxes, balls), have similar contours; pairs with clearly different
    proportions (tall vs stout, thin vs bulky, flat vs voluminous) or of different
    shape families have a LARGE contour difference. An structure change that
    visibly alters the silhouette (e.g., adding a wing, a canopy, a cargo box, a
    large hairstyle change; opening a convertible top) also counts as a LARGE
    structural change, while attribute changes that keep the contour (color,
    material, pattern) do not.
   

D3. Motion editing recognition: judge not only whether the action/motion of the
    subject is changed in the prompts, but ALSO whether the target prompt ends
    with  motion signals: a MotionEditor case tag — a parenthesized "case-<name>" such as
      (e.g., "a girl is dancing" -> "a girl is dancing(case-1)"     .
    If any holds, it is motion editing: Local Editing, no mask, fine-tune the
    Motion module (one-shot). 
"""

# ============================================================
# 3. Few-shot 示例（输出必须遵循 model_adaption 前置 + Thinking 结构）
# ============================================================
PROMPT_EXAMPLES = """\
## few-shot Examples

Example 1 — global editing / simple editing
Input:
SOURCE_PROMPT: a street with cars and pedestrians
TARGET_PROMPT: a street with cars and pedestrians in watercolor painting style
Output:
{"model_adaption": 'no-mask, zero-shot',
 "Thinking":
 '1.Prompt comparison analysis:
      The difference between source and target prompts is that the rendering style of the whole scene 
      changes to watercolor painting style.
  2.Editing type:
      The style of the entire scene changes while layout and content stay, which
      belongs to the editing type: Global Editing (style transfer).
  3.Mask mechanism:
      A mask would incorrectly freeze part of the scene, so no mask mechanism is
      needed here.
  4.Fine-tuning:
      Style transfer(global editing) does not modify object geometry, so zero-shot is sufficient and no fine-tuning is needed.'}

Example 2 —  local editing / complex editing(obvious structural change)
Input:
SOURCE_PROMPT: a man is cutting a watermelon on the table
TARGET_PROMPT: a man is cutting a pear on the table
Output:
{"model_adaption": 'mask, one-shot(temporal module)',
 "Thinking":
 '1.Prompt comparison analysis:
      The difference between source and target prompts is: the subject "watermelon"
      is replaced with "pear".
  2.Editing type:
      The change of subject in prompts involves an object replacement in the spatial
      domain of video frames, which belongs to the editing type: Local Editing.
  3.Mask mechanism:
      Therefore, we need a mask mechanism to segment the areas that are originally
      occupied by "watermelon" in the source frames and then generate the new object
      "pear" there. 
  4.Fine-tuning:
      Since the shape change between "watermelon" and "pear" may be significant,
      which involves structure modification, we also like to add a temporal module
      to the base model and fine-tune it to smooth the complex changes across video
      frames.'}

Example 3 — local editing / complex editing(object in motion)
Input:
SOURCE_PROMPT: a man in a red jacket is riding a bicycle on a forest road
TARGET_PROMPT: a man in a blue jacket is riding a bicycle on a forest road
Output:
{"model_adaption": 'mask , one-shot(temporal module)',
 "Thinking":
 '1.Prompt comparison analysis:
      The difference between source and target prompts is: the attribute "red" of the
      subject "jacket" is changed to "blue".
  2.Editing type:
      Only the color of a named subject changes while the scene layout stays, which
      belongs to the editing type: Local Editing (attribute change).
  3.Mask mechanism:
      Therefore, we need a mask mechanism to segment the areas originally occupied by
      the "jacket" and re-generate it in blue there, keeping the man, bicycle and
      forest background unchanged. 
  4.Fine-tuning:
      Since the man is in motion, fine-tuning is needed.'}

Example 4 — local editing / simple editing(object is static and the contour change is small)
Input:
SOURCE_PROMPT: a peach sits on the kitchen scale
TARGET_PROMPT: a apple sits on the kitchen scale
Output:
{"model_adaption": 'mask , zero-shot',
 "Thinking":
 '1.Prompt comparison analysis:
      The difference between source and target prompts is: the object "peach" is replaced with "apple".
  2.Editing type:
      The object "peach" is replaced with "apple" while the scene layout stays, which
      belongs to the editing type: Local Editing (object replacement).
  3.Mask mechanism:
        Therefore, we need a mask mechanism to segment the areas originally occupied by
        the "peach" and re-generate it as "apple" there, keeping the scale and background
        unchanged. 
  4.Fine-tuning:
      Since the fruit sits still on the scale and the shape difference between "peach"
      and "apple" is small, zero-shot attention manipulation is sufficient and no
      fine-tuning is needed.'}

Example 5 — local editing / simple editing(static subject, similar contour)
Input:
SOURCE_PROMPT: a dog is standing beside the wooden fence
TARGET_PROMPT: a wolf is standing beside the wooden fence
Output:
{"model_adaption": 'mask , zero-shot',
 "Thinking":
 '1.Prompt comparison analysis:
      The difference between source and target prompts is: the subject "dog"
      is replaced with "wolf".
  2.Editing type:
      The subject is replaced while the scene layout stays, which belongs to the
      editing type: Local Editing (object replacement).
  3.Mask mechanism:
      Therefore, we need a mask mechanism to segment the areas originally occupied
      by the "dog" and generate the "wolf" there, keeping the fence and
      background unchanged.
  4.Fine-tuning:
      Unlike change giraffe to pig, whose body proportions differ greatly. a dog and a wolf are similar with almost identical
      overall contours. The contour change is small AND the edited animal is static, so BOTH conditions hold simultaneously and zero-shot attention manipulation is
      sufficient and no fine-tuning is needed.'}

Example 6 — motion editing via MotionEditor case tag (no mask, one-shot Motion module)
Input:
SOURCE_PROMPT: a girl is dancing
TARGET_PROMPT: a girl is dancing (case-1)
Output:
{"model_adaption": 'no-mask, one-shot(Motion module)',
 "Thinking":
 '1.Prompt comparison analysis:
      The source and target prompts are word-for-word identical, but the target
      prompt ends with a MotionEditor case tag "(case-1)".
  2.Editing type:
      The case tag indicates that the motion of the subject is re-driven by the
      reference motion of the MotionEditor case, which belongs to the editing
      type: Motion Editing.
  3.Mask mechanism:
      Motion editing does not need a mask mechanism; the motion of the subject is
      driven directly by the reference motion of the MotionEditor case.
  4.Fine-tuning:
      Since changing the motion involves large structural (pose) changes of the
      subject across video frames, we need to invoke the motion editing module
      and fine-tune it, i.e. one-shot (Motion module).'}
"""

PROMPT_OUTPUT_SPEC = """\
## Output format
Return exactly ONE Python-dict-style object and nothing else.
IMPORTANT: output the "model_adaption" key FIRST, before the "Thinking" key — the downstream pipeline reads "model_adaption" 
from the stream before "Thinking" is fully generated, so its value must appear at the very beginning of the object. Do NOT 
use any apostrophes (') inside the "Thinking" string — write full words
(e.g. "we would" instead of "we'd"), because an apostrophe breaks the dict syntax.
{
  "model_adaption": '<ALWAYS two comma-separated parts: the mask decision ("mask " or "no-mask") followed by the paradigm ("zero-shot" or "one-shot(temporal module)" or "one-shot(Motion module)"). e.g. mask , one-shot(temporal module)>',
  "Thinking":
  '1.Prompt comparison analysis: <Semantic analysis of what differs between the source and target prompts>
  2.Editing type: <classify the editing operation>
  3.Mask mechanism: <if local editing: needing mask; if global editing: no mask is needed>
  4.Fine-tuning: <if change with with the motion object or the object need LARGE structural change: one-shot(temporal module); if Motion editing: one-shot(Motion module); otherwise: zero-shot >',
}"""

# 四段拼接为最终 system prompt，可按需增删段落
SYSTEM_PROMPT = "\n\n".join(
    [PROMPT_ROLE, PROMPT_PRINCIPLES, PROMPT_EXAMPLES, PROMPT_OUTPUT_SPEC]
)

# user 消息模板：实际编辑任务描述通过这里传入
USER_PROMPT_TEMPLATE = "SOURCE_PROMPT: {source_prompt}\nTARGET_PROMPT: {target_prompt}"

# API key 只从环境变量 DEEPSEEK_API_KEY 读取（已配置在 ~/.bashrc；勿硬编码进代码/仓库）
def _get_client():
    """按需创建 client；缺 key 时立刻给出明确指引，避免调用阶段才爆晦涩的 401。"""
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "[DeepSeek] 未检测到 DEEPSEEK_API_KEY 环境变量（Use_DeepSeek=True 时必需）。\n"
            "  解决：在运行终端执行 export DEEPSEEK_API_KEY=sk-xxx，或 source ~/.bashrc 后重试；\n"
            "  VSCode/IDE 内启动的终端需要重开才会继承新配置。")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def LLM(source_prompt, target_prompt):
    print("source_prompt=", source_prompt)
    print("target_prompt=", target_prompt[0])
    content = USER_PROMPT_TEMPLATE.format(
        source_prompt=source_prompt, target_prompt=target_prompt
    )

    response = _get_client().chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        temperature=0.2,   # 分类任务，低温度保证稳定
        top_p=1.0,
        max_tokens=2048,
        stream=False,
    )

    result = parse_llm_response(response.choices[0].message.content)

    return result


# 流式决策用：model_adaption 被要求前置输出（见 PROMPT_OUTPUT_SPEC），
# 其值以闭引号完整出现时该正则即可命中，无需等 Thinking 生成完。
ADAPTION_STREAM_RE = re.compile(r'"model_adaption"\s*:\s*[\'"](.*?)[\'"]')


def LLM_stream(source_prompt, target_prompt):
    """流式版 LLM()：不等全部输出结束，解析出 model_adaption 即提前返回。

    输出规范要求 model_adaption 前置，因此流式读取时该字段最先完整出现，
    管线可立刻根据它决定模型配置（zero-shot 免 tune / one-shot(temporal
    module) 启动 tune / one-shot(Motion module) 启动 MotionEditor），
    无需等待较长的 Thinking 文本生成完毕。

    返回 (model_adaption, info)：
      model_adaption — str，如 'no-mask, one-shot(Motion module)'；
                       流结束仍未解析出时用 parse_llm_response 全量兜底，
                       再失败则为 None
      info           — dict：t_adaption_ms(提前拿到决策的时刻) /
                       t_total_ms(流到此结束的时刻) / partial(已生成的部分响应)
    """
    print("source_prompt=", source_prompt)
    print("target_prompt=", target_prompt[0])
    content = USER_PROMPT_TEMPLATE.format(
        source_prompt=source_prompt, target_prompt=target_prompt
    )

    t0 = time.perf_counter()
    stream = _get_client().chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        temperature=0.2,   # 分类任务，低温度保证稳定
        top_p=1.0,
        max_tokens=2048,
        stream=True,
    )

    buf, adaption, t_adaption = "", None, None
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content or ""
        if not delta:
            continue
        buf += delta
        if adaption is None:
            m = ADAPTION_STREAM_RE.search(buf)
            if m:
                adaption = m.group(1).strip()
                t_adaption = (time.perf_counter() - t0) * 1000
                print(f"[DeepSeek][stream] model_adaption 提前解析成功 @ {t_adaption:.0f} ms: {adaption!r}")
                break  # 决策字段已到手，不等 Thinking 生成完，立刻返回供管线路由
    t_end = (time.perf_counter() - t0) * 1000

    if adaption is None:
        # 兜底：模型未按规范前置输出，流已读完，全量解析
        result = parse_llm_response(buf)
        adaption = result.get("model_adaption") if isinstance(result, dict) else None
        t_adaption = t_end
        print(f"[DeepSeek][stream] 未能提前解析，流结束后全量解析 @ {t_end:.0f} ms: {adaption!r}")

    info = {"t_adaption_ms": t_adaption, "t_total_ms": t_end, "partial": buf}
    return adaption, info


def parse_llm_response(response):
    # 剥掉可能的 markdown 代码围栏与首尾空白，避免严格解析失败
    response = response.strip()
    if response.startswith("```"):
        response = response.split("\n", 1)[1] if "\n" in response else response
        if response.rstrip().endswith("```"):
            response = response.rstrip()[:-3]
    try:
        # 首先尝试直接解析JSON
        result_dict = json.loads(response)
        return result_dict
    except json.JSONDecodeError:
        try:
            # 如果JSON解析失败，尝试使用ast.literal_eval解析Python字典
            result_dict = ast.literal_eval(response)
            if isinstance(result_dict, dict):
                return result_dict
            else:
                raise ValueError("解析结果不是字典格式")
        except (ValueError, SyntaxError):
            # 常见失败原因：模型在单引号 Thinking 字符串里输出了真实换行，
            # 使其成为非法 Python 字面量。按已知结构（model_adaption + Thinking）
            # 用 DOTALL 正则直接提取两个键的值；两个值的引号可能混用单双引号，
            # 因此分别独立匹配。两种键顺序都兼容：model_adaption 前置（当前
            # 输出规范要求的顺序）与 Thinking 前置（模型不遵守顺序时的兜底）。
            m = None
            for q1 in ("'", '"'):
                for q2 in ("'", '"'):
                    # 键顺序 A：model_adaption 在前
                    m = re.search(
                        r'"model_adaption"\s*:\s*' + q1 + r'(.*?)' + q1 +
                        r'\s*,\s*"Thinking"\s*:\s*' + q2 + r'(.*?)' + q2,
                        response, re.DOTALL)
                    if m:
                        return {"model_adaption": m.group(1).strip(),
                                "Thinking": m.group(2).strip()}
                    # 键顺序 B：Thinking 在前
                    m = re.search(
                        r'"Thinking"\s*:\s*' + q1 + r'(.*?)' + q1 +
                        r'\s*,\s*"model_adaption"\s*:\s*' + q2 + r'(.*?)' + q2,
                        response, re.DOTALL)
                    if m:
                        return {"Thinking": m.group(1).strip(),
                                "model_adaption": m.group(2).strip()}
            try:
                # 如果以上方法都失败，使用正则表达式提取键值对
                result_dict = {}
                # 清理响应字符串
                response = response.strip().replace('\n', '').replace(' ', '')

                # 提取大括号内的内容
                match = re.search(r'\{([^}]*)\}', response)
                if match:
                    content = match.group(1)
                    pairs = content.split(',')

                    for pair in pairs:
                        if ':' in pair:
                            key, value = pair.split(':', 1)
                        elif '=' in pair:
                            key, value = pair.split('=', 1)
                        else:
                            continue

                        key = key.strip().strip('"\'')
                        value = value.strip().strip('"\'')

                        # 处理布尔值
                        if value.lower() == 'true':
                            value = True
                        elif value.lower() == 'false':
                            value = False
                        else:
                            # 尝试数值转换
                            try:
                                value = float(value) if '.' in value else int(value)
                            except ValueError:
                                pass

                        result_dict[key] = value

                    return result_dict
                else:
                    return {}
            except Exception as e:
                print(f"解析响应时出错: {e}")
                return {}


if __name__ == "__main__":
    # 测试用例：西瓜 -> 披萨（局部替换 + 大结构变化，one-shot）
    result = LLM(
        source_prompt="a man is cutting a watermelon on the table",
        target_prompt=["a man is cutting a pizza on the table"],
    )
    print(result)
