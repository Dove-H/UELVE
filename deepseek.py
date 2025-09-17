from openai import OpenAI
import json
import ast

def LLM(source_prompt, target_prompt):
    print("source_prompt=",source_prompt)
    print("target_prompt=",target_prompt[0])
    client = OpenAI(api_key="sk-c662b517835346348120c427a45701cf", base_url="https://api.deepseek.com")

    # 分行处理system提示内容，并插入传入的参数
    content = (
        "执行一次基于stable diffusion模型的视频编辑任务，分析一下视频编辑任务的类型。"
        f"source prompt为原视频的描述: {source_prompt}，"
        f"target prompt为编辑目标的描述: {target_prompt}。"
        "局部编辑请返回local_editing=True，全局编辑请返回local_editing=False。"
        "编辑主体轮廓结构涉及较大的图像结构变化，返回fine-tune=True，否则返回fine-tune=False。"
        "如果是局部编辑，请根据prompt的描述帮我设置参数的值："
        "介绍一下背景：我们保存DDIM Inversion阶段的attenion feature map,因为其包含原视频的特征信息，随后在编辑阶段通过掩码将编辑阶段的背景区域的attention feature map 替换为前面保存的DDIM Inversion阶段的attention featu map，来达到只修改局部区域而保持背景不变的效果"
        "参数1：self_replace_steps，作用是决定从什么时候开始替换self-attention的attention feature map，返回self_replace_steps=（小数）"
        "仅返回固定答案,以python字典的格式返回,不要添加其他文字"
    )

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": content},
            {"role": "user", "content": "Hello"},
        ],
        stream=False
    )
  
    result=parse_llm_response(response.choices[0].message.content)

    return result


def parse_llm_response(response):
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