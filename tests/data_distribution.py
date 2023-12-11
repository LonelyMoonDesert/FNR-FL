import pandas as pd

data = {
    0: {0: 4917, 1: 1118, 2: 3666, 3: 4308, 4: 2208, 5: 3205, 6: 127, 7: 4191, 8: 4737},
    1: {0: 83, 1: 3882, 2: 1334, 3: 692, 4: 2792, 5: 1795, 6: 4873, 7: 809, 8: 263, 9: 5000}
}

# 创建一个 DataFrame
df = pd.DataFrame(data)

# 重置索引以将类别作为新的一列
df = df.reset_index()

# 重命名列名
df.columns = ['Class', 0, 1]

# 打印 DataFrame
print(df)
