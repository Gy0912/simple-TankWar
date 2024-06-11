import random

# 读取原始数据
with open('scores_50792.txt', 'r') as file:
    lines = file.readlines()

# 解析原始数据中的分数
scores = [int(line.split(': ')[1]) for line in lines]

# 生成200个随机分布的分数，范围可以根据原始数据的范围来设定
min_score = min(scores)
max_score = max(scores)
new_scores = [random.randint(min_score, max_score) for _ in range(100)]

# 随机插入新分数到原始数据中
for new_score in new_scores:
    position = random.randint(0, len(scores))
    scores.insert(position, new_score)

# 构建新的数据格式
modified_scores = [f"Score: {score}" for score in scores]

# 将新数据写入文件
with open('scores.txt', 'w') as file:
    for score in modified_scores:
        file.write(score + '\n')
