# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI

client = OpenAI(api_key="sk-c662b517835346348120c427a45701cf", base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "执行一次基于stable diffusion模型的视频编辑任务，分析一下视频编辑任务的类型。source prompt为原视频的描述，target prompt为编辑目标的描述。source prompt：“一只鸡在草地上跑”  target prompt：“一只猫在草地上跑，梵高风格”。局部编辑请返回edit_class=0 全局编辑请返回edit_class=1。编辑主体轮廓结构涉及较大的图像结构变化，返回fine-tune=1，否则返回fine-tune=0。仅返回固定答案，不要添加其他文字"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False
)

print(response.choices[0].message.content)