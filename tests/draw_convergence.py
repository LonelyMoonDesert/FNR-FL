import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

# 数据
rounds = ['round1', 'round2', 'round3', 'round4', 'round5', 'round6', 'round7', 'round8', 'round9', 'round10']
data = {
    'FedFNC': [0.1004, 0.8682, 0.9772, 0.9877, 0.9929, 0.9945, 0.9953, 0.9962, 0.9971, 0.997],
    'FedAvg': [0.1003, 0.7609, 0.7921, 0.815, 0.8433, 0.8594, 0.8849, 0.8792, 0.8976, 0.885],
    'FedProx': [0.1, 0.687, 0.7493, 0.7659, 0.791, 0.8298, 0.8646, 0.874, 0.8806, 0.8773],
    'SCAFFOLD': [0.1029, 0.7725, 0.8569, 0.8907, 0.9005, 0.9066, 0.905, 0.9133, 0.909, 0.9077],
    'MOON': [0.1, 0.3353, 0.5001, 0.1226, 0.4705, 0.5905, 0.5186, 0.6275, 0.6247, 0.6515],
    'FedNova': [0.1, 0.5626, 0.7551, 0.7968, 0.8419, 0.8522, 0.8729, 0.8711, 0.8963, 0.9043]
}


# 使用tab10颜色循环
colors = plt.cm.tab10.colors

# 设置图例标签
labels = ['FedFNC', 'FedAvg', 'FedProx', 'SCAFFOLD', 'MOON', 'FedNova']

# 绘制收敛曲线
plt.figure(figsize=(10, 6))

for i, label in enumerate(labels):
    x = np.linspace(0, len(rounds) - 1, 100)
    y = make_interp_spline(range(len(rounds)), data[label])(x)
    plt.plot(x, y, color=colors[i], label=label)
    plt.scatter(range(len(rounds)), data[label], color=colors[i], s=30,  zorder=5)

# 添加图例
plt.legend(loc='best')

# 添加标题和标签
plt.title('Label Distribution skew')
plt.xlabel('Rounds')
plt.ylabel('Test Accuracy')

# 显示网格
plt.grid(True, linestyle='--', alpha=0.6)

# 设置横轴刻度
plt.xticks(range(len(rounds)), rounds)
# 设置纵轴刻度
plt.yticks(np.arange(0, 1.1, 0.1))

# 显示图
plt.show()
