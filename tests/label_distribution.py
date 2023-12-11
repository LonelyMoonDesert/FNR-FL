import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Define your data
data = {
    'Class': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
    'P1': [953, 142, 141, 75, 695, 819, 0, 2482, 0, 0],
    'P2': [16, 43, 902, 1650, 86, 182, 0, 693, 110, 6],
    'P3': [9, 8, 290, 769, 841, 283, 119, 1044, 1014, 58],
    'P4': [395, 1200, 48, 68, 896, 681, 90, 17, 1351, 301],
    'P5': [504, 2917, 570, 721, 121, 356, 0, 0, 0, 0],
    'P6': [1262, 71, 325, 119, 1560, 14, 1, 85, 366, 0],
    'P7': [9, 273, 1657, 40, 1, 130, 1911, 624, 1160, 0],
    'P8': [722, 3, 281, 738, 22, 974, 624, 1, 82, 878],
    'P9': [1127, 153, 680, 500, 698, 1139, 92, 23, 1, 3665],
    'P10': [3, 190, 106, 320, 80, 422, 2163, 634, 916, 92]
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
