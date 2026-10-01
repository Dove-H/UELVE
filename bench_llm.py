# LLM 决策模块基准测试：推理延迟 / token 开销 / 模型版本 / 准确率
# 合并两轮共 27 个测试用例，每例真实调用一次
# 用法: python bench_llm.py

import time
import deepseek
from test_prompt_classification import CASES as CASES_A
from test_prompt_classification_20 import CASES as CASES_B

CASES = CASES_A + CASES_B  # 8 + 20 = 27（含 4 个观察用例）


def check(model_adaption, expected):
    if expected is None:
        return "OBSERVE"
    text = model_adaption.lower()
    return "PASS" if all(kw.lower() in text for kw in expected) else "FAIL"


def main():
    latencies, prompt_toks, completion_toks = [], [], []
    n_pass = n_fail = model_version = 0

    for name, src, tgt, expected in CASES:
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
        dt = (time.perf_counter() - t0) * 1000
        latencies.append(dt)
        u = resp.usage
        prompt_toks.append(u.prompt_tokens)
        completion_toks.append(u.completion_tokens)
        model_version = resp.model

        result = deepseek.parse_llm_response(resp.choices[0].message.content)
        adaption = result.get("model_adaption", "<解析失败>") if isinstance(result, dict) else "<解析失败>"
        verdict = check(adaption if adaption != "<解析失败>" else "", expected)
        if verdict == "PASS":
            n_pass += 1
        elif verdict == "FAIL":
            n_fail += 1
        print(f"{name:34s} {dt:7.0f}ms  in={u.prompt_tokens:5d} out={u.completion_tokens:4d}  "
              f"{verdict:7s} {adaption!r}")

    lat = sorted(latencies)
    n = len(lat)
    print("=" * 90)
    print(f"模型版本(model)        : {model_version}")
    print(f"用例数                 : {n}（计分 {n_pass + n_fail}，观察 {n - n_pass - n_fail}）")
    print(f"准确率                 : {n_pass}/{n_pass + n_fail} PASS")
    print(f"推理延迟(ms)  平均     : {sum(lat)/n:.0f}")
    print(f"              中位数   : {lat[n//2]:.0f}")
    print(f"              P95      : {lat[int(n*0.95)-1]:.0f}")
    print(f"              最小/最大 : {lat[0]:.0f} / {lat[-1]:.0f}")
    print(f"token 开销    输入平均 : {sum(prompt_toks)/n:.0f}（system 提示词为固定大头）")
    print(f"              输出平均 : {sum(completion_toks)/n:.0f}")
    print(f"              单次合计 : {(sum(prompt_toks)+sum(completion_toks))/n:.0f}")
    print(f"              27例总计 : {sum(prompt_toks)+sum(completion_toks)}")


if __name__ == "__main__":
    main()
