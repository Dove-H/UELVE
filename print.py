# ANSI终端颜色标识
YELLOW = "\033[1;33m"
BLUE = "\033[1;34m"
END = "\033[0m"

# 三段原始文本
part_system = '''{"role": "system","content": '...... '}'''

part_user = '''{
"role": "user","content":
source prompt: A rabbit is eating the watermelon, 
target prompt: A rabbit is eating the pizza.
}'''
part_result = '''{
"role": "result",
"Thinking": 
   '1.Prompt comparsion analysis: 
         The difference between source and target prompts is: the
         subject "watermelon" is replaced with "pizza".  

   2.Editing type: 
         The change of subject in prompts involves an object 
         replacement in spatial domain of video frames, which belongs
         to the editing type: Local Editing. 
   
   3.Mask mechanism: 
         Therefore, we need a mask mechanism to segment the areas
         that are originally occupied by "watermelon" in the source
         frames and then generate new object "pizza" there. We select
         off-the-shelf FastSAM to do it.
   
   4.Fine-tuning: 
         Since this shape change between "watermelon" and "pizza" may
         be significant, which involves structure modification, we'd
         also like to add a temporal module to the base model and
         fine-tune it to smooth the complex changes across video
         frames.'
  
"model_adaption": 'mask(FastSAM), one-shot(temporal module)'
}'''

# 关键词替换上色函数
def mark_keyword(text, keyword, color):
    return text.replace(f'"{keyword}"', f'{color}"{keyword}"{END}')



# 分别处理每一段
line_sys = mark_keyword(part_system, "system", YELLOW)
line_usr = mark_keyword(part_user, "user", YELLOW)

line_res = mark_keyword(part_result, "result", YELLOW)
line_res = mark_keyword(line_res, "Thinking", BLUE)
line_res = mark_keyword(line_res, "Prompt comparsion analysis", BLUE)
line_res = mark_keyword(line_res, "Mask mechanism", BLUE)
line_res = mark_keyword(line_res, "Editing type", BLUE)
line_res = mark_keyword(line_res, "Fine-tune", BLUE)
line_res = mark_keyword(line_res, "model_adaption", BLUE)

# 分段打印，中间空行分隔
print("  ")
print("  ")

print(">>>>>>>>>>>>>>The input Prompts:")
print("-----------------------------------------------")
print(line_sys)
print(line_usr)
print(" ")
print("  ")
print(">>>>>>>>>>>>>>The output Results:")
print("-----------------------------------------------")
print(line_res)
print("-----------------------------------------------")
print("  ")
print("  ")


print(">>>>>>>>>>>>>>Implement corresponding model adaptions:")
print("-----------------------------------------------")
print("  ")
print("  ")