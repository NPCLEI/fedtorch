import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 下载和解压数据集（确保你已经设置了 Kaggle API Token）
def download_and_prepare_data():
    os.makedirs('stackoverflow', exist_ok=True)
    os.system('kaggle datasets download -d datafiniti/stack-overflow-questions')
    os.system('unzip stack-overflow-questions.zip -d stackoverflow')

# 读取数据集
def load_data():
    # 假设你有一个 CSV 文件名为 'questions.csv'
    data = pd.read_csv('stackoverflow/questions.csv')
    return data

# 数据预处理
def preprocess_data(data):
    # 假设我们使用特征 'feature_column' 和标签 'label_column'
    X = data[['feature_column']].values  # 替换为实际特征列
    y = data['label_column'].values        # 替换为实际标签列

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 标准化特征
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

# 定义逻辑回归模型
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# 训练模型
def train_model(X_train, y_train):
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    input_dim = X_train.shape[1]
    model = LogisticRegression(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    return model

# 测试模型
def test_model(model, X_test, y_test):
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        predicted = (test_outputs > 0.5).float()
        accuracy = (predicted.eq(y_test_tensor)).sum().item() / y_test_tensor.size(0)
        print(f'Accuracy: {accuracy:.4f}')

# 主程序
def main():
    download_and_prepare_data()
    data = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(data)
    model = train_model(X_train, y_train)
    test_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
