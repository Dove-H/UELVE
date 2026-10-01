# 提示词分类效果测试脚本（扩展集：20 个新用例）
# 覆盖：材质/颜色/静止替换/运动替换/大结构替换/全局风格/季节/昼夜/
#       背景/动作编辑(带map/纯prompt/两者叠加)/运动主体上的属性修改/边界case
# 用法: python test_prompt_classification_20.py

import deepseek

CASES = [
    # (用例名, source, target, 期望关键词；None=仅观察)
    ("01 材质修改(静止)", "a wooden table in the kitchen", "a marble table in the kitchen",
     ["mask", "zero-shot"]),
    ("02 颜色修改(静止车辆)", "a red car is parked on the street", "a blue car is parked on the street",
     ["mask", "zero-shot"]),
    ("03 同容器内容替换(轮廓不变)", "a glass of orange juice on the table", "a glass of apple juice on the table",
     ["mask", "zero-shot"]),
    ("04 静止大结构替换", "a man is holding a basketball", "a man is holding a tennis racket",
     ["mask", "one-shot", "temporal"]),
    ("05 运动物体替换(动物)", "a horse is galloping on the beach", "a zebra is galloping on the beach",
     ["mask", "one-shot", "temporal"]),
    ("06 运动物体替换(车辆)", "a red bus is driving down the city street", "a blue truck is driving down the city street",
     ["mask", "one-shot", "temporal"]),
    ("07 运动主体人物替换", "a boy is skateboarding in the park", "a girl is skateboarding in the park",
     ["mask", "one-shot", "temporal"]),
    ("08 全局风格(油画)", "a photo of a mountain landscape at sunset", "an oil painting of a mountain landscape at sunset",
     ["no-mask", "zero-shot"]),
    ("09 全局季节变化", "a green forest in summer", "a snow-covered forest in winter",
     ["no-mask", "zero-shot"]),
    ("10 全局昼夜变化", "a city skyline during the day", "a city skyline at night",
     ["no-mask", "zero-shot"]),
    ("11 背景替换(观察)", "a man is playing guitar on a stage", "a man is playing guitar on a beach",
     None),
    ("12 背景替换2(观察)", "a woman is reading a book in a cafe", "a woman is reading a book in a library",
     None),
    ("13 动作编辑(带信号map)", "a woman is doing yoga in a room", "a woman is doing yoga in a room (yoga_pose_map_11.gif)",
     ["no-mask", "one-shot", "Motion"]),
    ("14 动作编辑(纯prompt改动作)", "a man is walking his dog along the river", "a man is running with his dog along the river",
     ["one-shot", "Motion"]),
    ("15 动作编辑(改动作+带map)", "a cat is sitting on the windowsill", "a cat is jumping on the windowsill (cat_jump_map_02.gif)",
     ["no-mask", "one-shot", "Motion"]),
    ("16 运动主体上的属性修改", "a man in a white shirt is playing tennis", "a man in a black shirt is playing tennis",
     ["mask", "zero-shot"]),
    ("17 静止小结构替换(食物)", "a bowl of soup on the table", "a bowl of salad on the table",
     ["mask", "zero-shot"]),
    ("18 静止大结构替换(动物)", "a bird is perched on a branch", "a squirrel is perched on a branch",
     ["mask", "one-shot", "temporal"]),
    ("19 物体添加(观察,超出当前分类)", "an empty street in the rain", "a street in the rain with a red umbrella",
     None),
    ("20 局部风格化(观察,边界case)", "a girl is riding a bike in the park", "a girl is riding a bike in the park in Van Gogh style",
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
        for line in result.get("Thinking", "").splitlines():
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
