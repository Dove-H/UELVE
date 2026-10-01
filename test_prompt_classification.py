# 提示词分类效果测试脚本
# 覆盖五类编辑 + 关键判据边界（静止替换 vs 运动替换 vs 大结构替换）
# 用法: python test_prompt_classification.py

import deepseek

# (用例名, source, target, 期望 model_adaption 应包含的关键词集合)
# 关键词全部命中才判 PASS；None 表示仅观察、不计分
CASES = [
    ("属性修改(静止,小结构)", "a dog is eating pizza on the floor", "a cat is eating pizza on the floor",
     ["mask", "zero-shot"]),
    ("属性修改(颜色)", "a man in a red jacket is riding a bicycle on a forest road",
     "a man in a blue jacket is riding a bicycle on a forest road",
     ["mask", "zero-shot"]),
    ("大结构替换(静止物体)", "a man is cutting a watermelon on the table",
     "a man is cutting a pizza on the table",
     ["mask", "one-shot", "temporal"]),
    ("运动物体替换", "a dog is running on the grass", "a cat is running on the grass",
     ["mask", "one-shot", "temporal"]),
    ("全局风格迁移", "a realistic city street with cars and pedestrians",
     "an anime-style city street with cars and pedestrians",
     ["no-mask", "zero-shot"]),
    ("动作编辑(带信号map文件名)", "a girl is dancing", "a girl is dancing (dance_pose_map_03.gif)",
     ["no-mask", "one-shot", "Motion"]),
    ("动作编辑(prompt改动作,无map)", "a man is standing still in a park", "a man is jumping in a park",
     ["one-shot", "Motion"]),
    ("背景替换(观察用)", "a woman is sitting on a chair in an office",
     "a woman is sitting on a chair on a beach",
     None),
]


def check(model_adaption, expected):
    if expected is None:
        return "OBSERVE"
    text = model_adaption.lower()
    return "PASS" if all(kw.lower() in text for kw in expected) else "FAIL"


def main():
    n_pass = n_fail = 0
    for name, src, tgt, expected in CASES:
        print("=" * 70)
        print(f"[CASE] {name}")
        print(f"  SOURCE: {src}")
        print(f"  TARGET: {tgt}")
        try:
            result = deepseek.LLM(source_prompt=src, target_prompt=[tgt])
        except Exception as e:
            print(f"  API/解析异常: {e}\n  -> ERROR")
            n_fail += 1
            continue
        if not isinstance(result, dict) or "model_adaption" not in result:
            print(f"  原始返回解析失败: {result!r}\n  -> FAIL(解析)")
            n_fail += 1
            continue
        thinking = result.get("Thinking", "")
        # 只打印 Thinking 每步的首行，便于快速浏览
        for line in thinking.splitlines():
            s = line.strip()
            if s[:2] in ("1.", "2.", "3.", "4."):
                print(f"  | {s}")
        adaption = result["model_adaption"]
        verdict = check(adaption, expected)
        print(f"  EXPECTED: {expected}")
        print(f"  GOT model_adaption: {adaption!r}")
        print(f"  -> {verdict}")
        if verdict == "PASS":
            n_pass += 1
        elif verdict == "FAIL":
            n_fail += 1
    total = n_pass + n_fail
    print("=" * 70)
    print(f"结果: {n_pass}/{total} PASS" + ("（观察用例不计分）" if total < len(CASES) else ""))


if __name__ == "__main__":
    main()
