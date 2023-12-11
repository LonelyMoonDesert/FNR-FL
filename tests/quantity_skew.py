import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Define your data
data = {
    'Class': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
    'P1': [389, 391, 373, 422, 361, 435, 385, 387, 369, 389],
    'P2': [1, 2, 5, 3, 1, 3, 1, 4, 2, 3],
    'P3': [20, 15, 12, 20, 23, 17, 17, 17, 19, 14],
    'P4': [338, 294, 321, 309, 305, 289, 287, 318, 305, 313],
    'P5': [254, 283, 302, 269, 270, 269, 255, 269, 288, 311],
    'P6': [918, 962, 925, 902, 982, 896, 947, 946, 935, 961],
    'P7': [635, 606, 619, 610, 599, 648, 664, 591, 609, 675],
    'P8': [534, 505, 521, 487, 544, 508, 527, 539, 502, 473],
    'P9': [163, 173, 187, 178, 171, 149, 162, 199, 169, 190],
    'P10': [1748, 1769, 1735, 1800, 1744, 1786, 1755, 1730, 1802, 1671]
}

# Create a DataFrame
df = pd.DataFrame(data)
df.set_index('Class', inplace=True)

# Replace 0 with NaN for white cells
df = df.replace(0, np.nan)

# Create a heatmap
plt.figure(figsize=(14, 8))
ax = sns.heatmap(df, cmap='YlGnBu', annot=True, fmt='.0f', cbar=True)

# Set the colorbar label
cbar = ax.collections[0].colorbar
cbar.set_label('Count', rotation=270, labelpad=20)

plt.title('CIFAR-10 Data Distribution Across Parties')
plt.xlabel('Parties')
plt.ylabel('Classes')

plt.show()
