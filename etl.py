import os

# 输入和输出文件夹路径
input_folder = '西游记白话文'
output_folder = 'output'

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 遍历输入文件夹中的所有文件
for filename in os.listdir(input_folder):
    if filename.endswith('.txt'):
        # 构建文件路径
        input_file_path = os.path.join(input_folder, filename)
        
        # 读取文件内容
        with open(input_file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        # 去除每行的前后空格
        lines = [line.strip() for line in lines]
        
        # 获取新的文件名（原文件的第一行）
        new_filename = lines[0] + '.txt'
        output_file_path = os.path.join(output_folder, new_filename)
        
        # 替换第一行
        lines[0] = new_filename
        
        # 将修改后的内容写入输出文件
        with open(output_file_path, 'w', encoding='utf-8') as file:
            file.write('\n'.join(lines) + '\n')

print("所有文件已处理并保存到 output 文件夹中。")
