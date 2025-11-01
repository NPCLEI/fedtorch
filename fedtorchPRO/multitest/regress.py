import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# 生成数据：反比例函数 Y = k / X + 噪声
k = 1  # 反比例系数
X = np.linspace(0.01, 10, 100)  # 避免X为0
Y = k / (X) + np.random.normal(0, 4, X.shape)   # 添加噪声

# 转换为PyTorch张量
X_tensor = torch.tensor(X, dtype=torch.float32).view(-1, 1)
Y_tensor = torch.tensor(Y, dtype=torch.float32).view(-1, 1)

# 定义神经网络模型
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.U = nn.Sequential(
                nn.Linear(1, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 1),
        )
        self.K = nn.Sequential(
                nn.Linear(1, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 1),
        )
        self.B = nn.Sequential(
                nn.Linear(1, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 1),
        )

    def forward(self, x):
        K = self.K(x)
        B = self.B(x)
        U = self.U(x)
        return K/U * x + B

# 实例化模型、定义损失函数和优化器
model = NeuralNetwork()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    
    optimizer.zero_grad()  # 清空梯度
    Y_pred = model(X_tensor)  # 前向传播
    loss = criterion(Y_pred, Y_tensor)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数

# 预测
model.eval()
Y_fit = model(X_tensor).detach().numpy()

# 绘图
plt.scatter(X, Y, color='blue', label='Data Points')
plt.plot(X, Y_fit, color='red', label='Fitted Curve')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Fitting Inverse Function with PyTorch')
plt.legend()
plt.savefig('./test.png')
