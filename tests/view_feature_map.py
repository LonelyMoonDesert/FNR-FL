import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt

# 加载一个预训练的图像分类模型，例如ResNet
model = models.resnet18(pretrained=True)
model.eval()  # 设置为评估模式

# 选择一个层（可以根据需要进行更改）
layer = model.layer1[2]

# 创建一个函数，用于提取特征图
def get_interested_feature_map(x):
    activations = []
    def hook(module, input, output):
        activations.append(output)
    hook_handle = layer.register_forward_hook(hook)
    model(x)
    hook_handle.remove()
    return activations[0]

# 加载一张图像并进行预处理
from torchvision import transforms
from PIL import Image

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])
image = Image.open('bird.jpg')
image = preprocess(image).unsqueeze(0)  # 添加批次维度

# 获取特征图
feature_map = get_interested_feature_map(image)

# 可视化特征图
plt.figure()
for i in range(feature_map.size(1)):
    plt.subplot(4, 4, i + 1)
    plt.imshow(feature_map[0, i].detach().numpy())
plt.show()
