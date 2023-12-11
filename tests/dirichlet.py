import numpy as np

# 定义 Dirichlet 分布的参数 alpha
alpha = [0.5, 0.5]
# 使用 numpy.random.dirichlet 函数进行采样
sample = np.random.dirichlet(alpha)

print("采样结果:", sample)
