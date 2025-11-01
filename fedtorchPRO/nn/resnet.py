import torch
import torch.nn as nn
import torchvision.models as models


# 加载预训练的ResNet-50模型
model = models.resnet50(pretrained=True)

# 修改最后的全连接层，以适应100分类
num_classes = 100
model.fc = nn.Linear(model.fc.in_features, num_classes)

# 保存模型的权重
torch.save(model.state_dict(), './LLMs/resnet/resnet50_100class.pth')

print("模型权重已保存！")
