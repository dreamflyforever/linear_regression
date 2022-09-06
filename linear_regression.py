import random
import torch
import torch.nn as nn

## set y = 2X + 4.2
## 人造数据集
def create_data(w, b, nums_example):
    X = torch.normal(0, 1, (nums_example, len(w)))
    y = torch.matmul(X, w) + b
    print("y_shape:", y.shape)
    y += torch.normal(0, 0.01, y.shape)  # 加入噪声
    return X.reshape(-1, 1), y.reshape(-1, 1)  # y从行向量转为列向量


true_w = torch.tensor([2], dtype=torch.float)
true_b = 4.2
features, labels = create_data(true_w, true_b, 1000)

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    # 前向传播
    def forward(self, x):
        out = self.linear(x)
        return out

input_dim = 1
output_dim = 1
model = LinearRegressionModel(input_dim, output_dim)
print("model:\t", model)
epochs = 1000
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate) # 优化器SGD
criterion = nn.MSELoss() # 定义一个损失函数

#训练模型
for epoch in range(epochs):
    epoch += 1
    inputs = features
    labels = labels
    # 梯度要清零每一次迭代 (缺省情况梯度是累加的，梯度反向传播前，先需把梯度清空)
    optimizer.zero_grad()
    # 前向传播(把输入数据传入神经网络NET实例化对象model中，自动执行forward函数，得到output的值)
    outputs = model.forward(inputs)
    # 计算损失
    loss = criterion(outputs, labels)
    # 反向传播
    loss.backward()
    # 更新权重参数
    optimizer.step()
    if epoch % 50 == 0:
    	print("epoch {}， loss {}".format(epoch, loss.item()))

x = torch.tensor([1], dtype=torch.float)
y = model.forward(x)
print("x = 1, y = {}".format(y))
