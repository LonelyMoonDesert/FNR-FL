import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 创建数据
data = {
    'value': [0.532, 0.688, 0.192, 0.276, 0.574, 0.581, 0.547, 0.634, 0.568, 0.618, 0.877, 0.797, 0.687, 0.611, 0.882, 0.889, 0.96, 0.804, 0.736, 0.987],
    'treat': ['Before', 'Before', 'Before', 'Before', 'Before', 'Before', 'Before', 'Before', 'Before', 'Before', 'After', 'After', 'After', 'After', 'After', 'After', 'After', 'After', 'After', 'After'],
    'class': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck', 'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
}

df = pd.DataFrame(data)

# 将 "Before" 和 "After" 分成两组
before = df[df['treat'] == 'Before']
after = df[df['treat'] == 'After']

# 获取类别和相应的数量
classes = df['class'].unique()
num_classes = len(classes)

# 设置画布大小
plt.figure(figsize=(12, 7))

# 设置颜色
colors = ['#A9DFBF', '#85C1E9']

# 设置宽度和间隔
bar_width = 0.4
bar_spacing = 0.4

# 计算 x 坐标
x = np.arange(num_classes)

# 绘制条形柱状图
plt.bar(x - bar_spacing/2, before.groupby('class')['value'].mean(), bar_width, label='Before', color=colors[0])
plt.bar(x + bar_spacing/2, after.groupby('class')['value'].mean(), bar_width, label='After', color=colors[1])

# 设置 x 轴标签
plt.xlabel('Class')
plt.xticks(x, classes, rotation=45)
plt.ylabel('Test Accuracy')
plt.title('Comparison of Model Accuracy Before and After Feature Norm Regularization')
plt.legend()
plt.show()
