# 流式 + model_adaption 前置 的用时对比实验
# A: 现状 —— stream=False, Thinking 在前（需等全部生成完才有结果）
# B: 实验 —— stream=True,  few-shot/输出规范重排为 model_adaption 在前,
#           记录 拿到 model_adaption 的时刻(提前决策点) 与 全部完成时刻
# 用法: python bench_stream.py

import re
import time
import deepseek

# 代表性用例：覆盖四种决策路由
CASES = [
    ("颜色修改", "a man in a red jacket is riding a bicycle on a forest road",
     "a man in a blue jacket is riding a bicycle on a forest road"),
    ("大结构替换", "a man is cutting a watermelon on the table",
     "a man is cutting a pizza on the table"),
    ("全局风格", "a realistic city street with cars and pedestrians",
     "an anime-style city street with cars and pedestrians"),
    ("动作编辑(带map)", "a girl is dancing", "a girl is dancing (dance_pose_map_03.gif)"),
    ("运动物体替换", "a dog is running on the grass", "a cat is running on the grass"),
    ("静止小结构替换", "a dog is eating pizza on the floor", "a cat is eating pizza on the floor"),
]

# ---------- 构造 model_adaption 前置版 SYSTEM_PROMPT ----------
def build_reordered_prompt():
    p = deepseek.SYSTEM_PROMPT
    # 1) 重排每个 few-shot 示例: {"Thinking": '...', "model_adaption": '...'}
    #    -> {"model_adaption": '...', "Thinking": '...'}
    def reorder(m):
        thinking, adaption = m.group(1), m.group(2)
        return '{"model_adaption": \'' + adaption + '\',\n "Thinking": \'' + thinking + "'}"
    p = re.sub(
        r'\{"Thinking":\s*\'(.*?)\',\s*"model_adaption":\s*\'(.*?)\'\s*\}',
        reorder, p, flags=re.DOTALL)
    # 2) 输出规范里加一条硬性要求
    p = p.replace(
        "## Output format",
        '## Output format\nIMPORTANT: output the "model_adaption" key FIRST, '
        'before the "Thinking" key.')
    return p

REORDERED_PROMPT = build_reordered_prompt()

ADAPTION_DONE = re.compile(r'"model_adaption"\s*:\s*[\'"]([^\'"]+)[\'"]\s*,')


def run_A(src, tgt):
    """现状: 非流式, 返回总用时(ms)"""
    t0 = time.perf_counter()
    resp = deepseek.client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": deepseek.SYSTEM_PROMPT},
            {"role": "user", "content": deepseek.USER_PROMPT_TEMPLATE.format(
                source_prompt=src, target_prompt=[tgt])},
        ],
        temperature=0.2, top_p=1.0, max_tokens=2048, stream=False,
    )
    total = (time.perf_counter() - t0) * 1000
    result = deepseek.parse_llm_response(resp.choices[0].message.content)
    adaption = result.get("model_adaption", "<解析失败>") if isinstance(result, dict) else "<解析失败>"
    return total, total, adaption  # 拿到决策的时刻 == 总用时


def run_B(src, tgt):
    """实验: 流式 + adaption 前置, 返回 (TTFT, 决策时刻, 总用时, adaption)"""
    t0 = time.perf_counter()
    stream = deepseek.client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": REORDERED_PROMPT},
            {"role": "user", "content": deepseek.USER_PROMPT_TEMPLATE.format(
                source_prompt=src, target_prompt=[tgt])},
        ],
        temperature=0.2, top_p=1.0, max_tokens=2048, stream=True,
    )
    buf, ttft, t_decision, adaption = "", None, None, "<未捕获>"
    for chunk in stream:
        now = (time.perf_counter() - t0) * 1000
        delta = chunk.choices[0].delta.content or ""
        if not delta:
            continue
        if ttft is None:
            ttft = now
        if t_decision is None:
            buf += delta
            m = ADAPTION_DONE.search(buf)
            if m:
                t_decision = now
                adaption = m.group(1)
    total = (time.perf_counter() - t0) * 1000
    if t_decision is None:  # adaption 在末尾或流中未匹配到, 兜底解析全量
        adaption = deepseek.parse_llm_response(buf).get("model_adaption", "<解析失败>")
        t_decision = total
    return ttft, t_decision, total, adaption


def main():
    print(f"{'用例':14s} | {'A:现状总用时':>10s} | {'B:TTFT':>7s} | {'B:决策时刻':>8s} | {'B:总用时':>7s} | A结果 / B结果")
    a_total, b_ttft, b_dec, b_total = [], [], [], []
    ok = 0
    for name, src, tgt in CASES:
        ta, _, a_ad = run_A(src, tgt)
        ttft, tdec, tb, b_ad = run_B(src, tgt)
        a_total.append(ta); b_ttft.append(ttft); b_dec.append(tdec); b_total.append(tb)
        match = str(a_ad).lower().replace(" ", "") == str(b_ad).lower().replace(" ", "")
        ok += match
        print(f"{name:14s} | {ta:8.0f}ms | {ttft:5.0f}ms | {tdec:6.0f}ms | {tb:5.0f}ms | "
              f"{'一致' if match else '不一致!'} {a_ad!r} / {b_ad!r}")
    n = len(CASES)
    print("=" * 100)
    print(f"平均    A 现状(非流式)总用时          : {sum(a_total)/n:7.0f} ms   <- 现在拿到决策要等这么久")
    print(f"平均    B 流式 TTFT                  : {sum(b_ttft)/n:7.0f} ms")
    print(f"平均    B 流式 拿到model_adaption时刻: {sum(b_dec)/n:7.0f} ms   <- 提前决策点")
    print(f"平均    B 流式 全部生成完            : {sum(b_total)/n:7.0f} ms")
    print(f"平均    提前量(A总用时 - B决策时刻)   : {sum(a_total)/n - sum(b_dec)/n:7.0f} ms "
          f"({(1 - sum(b_dec)/n/(sum(a_total)/n))*100:.0f}% 提速)")
    print(f"两组决策一致率: {ok}/{n}")


if __name__ == "__main__":
    main()
