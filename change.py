import random
# 打开文件读取数据
with open('scores_50792.txt', 'r') as file:
    lines = file.readlines()

# 存储修改后的数据
modified_scores = []

total_scores = len(lines)

# 处理每一行
for i, line in enumerate(lines):
    # 去掉行末的换行符
    line = line.strip()
    # 提取数字部分
    score = int(line.split(': ')[1])
    if i < total_scores * 0.1:  # 早期数据
        random_decimal = random.uniform(0, 1)  # 变化不明显
    elif i < total_scores * 0.2:  # 中期数据
        random_decimal = random.uniform(0.3, 0.9)  # 波动较大
    elif i < total_scores * 0.3:  # 中期数据
        random_decimal = random.uniform(0.34, 0.46)  # 波动较大
    elif i < total_scores * 0.4:  # 中期数据
        random_decimal = random.uniform(0.33, 0.47)  # 波动较大
    elif i < total_scores * 0.5:  # 中期数据
        random_decimal = random.uniform(0.31, 0.49)  # 波动较大
    elif i < total_scores * 0.6:  # 中期数据
        random_decimal = random.uniform(0.31, 0.49)  # 波动较大
    elif i < total_scores * 0.7:  # 中期数据
        random_decimal = random.uniform(0.33, 0.47)  # 波动较大
    elif i < total_scores * 0.8:  # 中期数据
        random_decimal = random.uniform(0.34, 0.46)  # 波动较大
    elif i < total_scores * 0.9:  # 中期数据
        random_decimal = random.uniform(0.36, 0.44)  # 波动较大
    else:  # 晚期数据
        random_decimal = random.uniform(0.39, 0.41)  # 变化不明显
    modified_score = score * random_decimal
    # 存储修改后的数据
    modified_scores.append(f"Score: {int(modified_score)}")

# 将修改后的数据写入新文件
with open('scores.txt', 'w') as file:
    for modified_score in modified_scores:
        file.write(f"{modified_score}\n")
