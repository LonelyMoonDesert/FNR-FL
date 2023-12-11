import matplotlib.pyplot as plt
import numpy as np

# 模拟数据，这里使用随机数据，你需要根据你的情况替换为实际的数据
global_model = np.array([1.0, 2.0, 3.0])  # 全局模型参数
client_updates = [
    np.array([1.1, 2.3, 2.8]),  # 客户端1的模型参数更新
    np.array([1.2, 2.1, 2.9]),  # 客户端2的模型参数更新
]

# 创建一个图形
plt.figure()

# 绘制全局模型参数
plt.quiver(0, 0, global_model[0], global_model[1], angles='xy', scale_units='xy', scale=1, color='b', label='Global Model')

# 绘制每个客户端的更新方向
for i, update in enumerate(client_updates):
    plt.quiver(0, 0, update[0], update[1], angles='xy', scale_units='xy', scale=1, color='r', label=f'Client {i+1}')

plt.xlim(-0.5, 2.5)
plt.ylim(-0.5, 3.0)
plt.legend()
plt.grid()
plt.show()
