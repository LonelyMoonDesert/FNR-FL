import numpy as np

# 生成十个随机数，范围在0.6到0.75之间
random_numbers = np.random.uniform(0.6, 0.75, 10)

# 保留4位小数
random_numbers = np.round(random_numbers, 4)

# 对随机数进行排序
sorted_numbers = np.sort(random_numbers)

for i in range(len(sorted_numbers)):
    print(sorted_numbers[i])
