## Lec 3. 卷积神经网络 Convolution Neural Network[](http://localhost:8888/lab/tree/JupterLab/Lec_3.ipynb?#Lec-3.-%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C-Convolution-Neural-Network)

- 空间不变性 `Spatial Invariance`
    
- 卷积 `Convolution` -> 互相关运算 `Correlation`
    
- 具体模型结构
    
    - Feature Extraction 特征提取：卷积神经网络
    - Classifier 分类器：线性神经网络
### 3.1 互相关运算
```
import torch
import torch.nn as nn

def corr2d(X, kernel): #kernel:卷积核
	"""二维互相关运算函数"""
	h_k, w_k = kernel.shape
	h_x, w_x = X.shape
	feature_map = torch.Zeors(h_x - h_k + 1, w_x - w_k + 1)

	for i in feature_map.shape[0]:
		for j in feature_map.shape[1]:
			feature_map[i,j] = (X([i:i + h_k, j:j + h_w])*kernel).sum()
	return feature_map

```

```
X = torch.tensor([[0, 1, 2, 3], 
                  [1, 2, 3, 4], 
                  [2, 3, 4, 5], 
                  [3, 4, 5, 6]]) #4x4

kernel = torch.tensor([[0, 1, 2], 
                       [3, 4, 5], 
                       [6, 7, 8]]) #3x3
corr2d(X, kernel)
```
```
输出结果为：tensor([[ 96., 132.],
        [132., 168.]]) #(4-3+1 x 4-3+1) = 2x2
```
### 3.2 卷积神经网络层的简易实现
```
def corr2d(X, kernel):
    """ 二维互相关运算函数 """
    h_k, w_k = kernel.shape
    h_x, w_x = X.shape
    feature_map = torch.zeros((h_x - h_k + 1, w_x - w_k + 1))

    for i in range(feature_map.shape[0]):
        for j in range(feature_map.shape[1]):
            feature_map[i, j] = (X[i: i + h_k, j: j + w_k] * kernel).sum()
    return feature_map


class Conv2d(nn.Module):
	def __init__(self, kernel_size):
		super().__init__()
		self.weight = nn.Parameter(torch.rand(kernel_size))
		self.bias = nn.Parameter(torch.zeros(1))

	def forward(self, x):
		"""向前传播方法"""
		feature_map = corr2d(x, self.weight) + self.bias
		return feature_map
```
```
#尝试调用简易方法
conv2d = Conv2d(kernel_size=(3, 3))
conv2d
实际输出：Conv2d()

feature_map = conv2d(X)
feature_map
实际输出：tensor([[ 9.1574, 13.5972],
        [13.5972, 18.0370]], grad_fn=<AddBackward0>)
```

### 3.3 卷积神经网络的使用[](http://localhost:8888/lab/tree/JupterLab/Lec_3.ipynb?#3.3-%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%9A%84%E4%BD%BF%E7%94%A8)

- nn.Conv2d 卷积层模块
- nn.AvgPool2d 池化层模块
```
import torch.nn as nn

module = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = (2,2), stride = 1, padding = 0)
module

实际输出：Conv2d(3, 16, kernel_size=(2, 2), stride=(1, 1))


X.float().view(1, 1, 4, 4).shape  # [batch_size, channels, height, width]
实际输出：torch.Size([1, 1, 4, 4])

x = torch.rand(1, 3, 4, 4)
module(x).shape
实际输出：torch.Size([1, 16, 3, 3]) #这里x的in_channels与module里设置的in_channels要相等
```

### 3.4 经典卷积神经网络 (LeNet)[](http://localhost:8888/lab/tree/JupterLab/Lec_3.ipynb?#3.4-%E7%BB%8F%E5%85%B8%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C-\(LeNet\))

- Yann Lecun
- MNIST 手写数字图像识别
    - 逻辑回归、决策树、随机森林
    - 神经网络
        - MLP: Linear
        - CNN: LeNet:
![[Pasted image 20250211200234.png]]
###### 图中的汇聚层就是池泳层

```
# 1.构建数据集
imort pandas as pd
import torch
import torch.nn as nn

df_dataset = pd.read_csv("/Users/vegeta/JupterLab/data_set/fashion-mnist_test.csv") #这里使用的是相对路径(因为用的mac...)

train_set = df_dataset.sample(frac = 0.7)
test_set = df_dataset[~df_dataset.index.isin(train_set.index)]

from torch.utils.data import Dataset, DataLoader

class FasionMNIST(Dataset):
	"""构建FashionMINIST图像数据集"""
	def __init__(self, df_dataset):
		self.y = df_dataset.label.values
		self.x = df_dataset.iloc[:, 1:].values

	def __len__(self):
		return len(self.y)

	def __getitem__(self, idx):
		y = torch.LongTensor([self.y[idx]])
		x = torch.Tensor(self.x[idx])
		return y,x
```
```
train_dataset = FashionMINIST(train_set)
test_dataset = FashionMINIST(test_set)

train_dataloader = DataLoader(train_dataset, batch_size = 256, shuffle = True)
test_dataloader = DataLoader(test_dataset, batch_size)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
```
```
 #2.构建模型(与上图相对应)
lenet = nn.Sequential(nn.Conv2d(1, 6, kernel_size = 5, padding = 2),
                      nn.Sigmoid(),
                      nn.AvgPool2d(kernel_size = 2, stride = 2),
                      
                      nn.Conv2d(6, 16, kernel_size = 5)
                      nn.Sigmoid(),
                      nn.AvgPool2d(kernel_size = 2, stride = 2),
                      
                      nn.Flatten(),
                      nn.Linear(16*5*5, 120),
                      nn.Sigmoid(),
                      
                      nn.Linear(120, 84),
                      nn.Sigmoid(),
                      
                      nn.Linear(84, 10))

lenet.to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lenet.parameters(), lr = 1e-4)
```
```
 #3.定义训练模型
 def train_model(model, train_dataloader, loss_func, optimizer):
	 total_loss = 0

	lenet.train() #打开训练模式，使得参数可以被调整
	for y,x in train_dataloader():
		# model input: [batch_size, channels, height, width]
		y_hat = model(x.view(x.shape[0], 1, 28, 28).to(device))
		loss = loss_func(y_hat, y.to(device).view(y.shape[0]))

		optimizer.zero_grad()
        loss.backward()
        optimizer.step()

		total_loss += loss.item()
	print(f"Total loss: {total_loss/len(train_dataloader): 0.4f}")
    model.eval()
    return total_loss/len(train_dataloader)
	
```
```
 #4. 定义测试模型
 def test_model(model, test_dataloader, loss_func):
    """ 训练模型 """
    total_loss = 0.

    model.eval() # 打开模型验证模式，所有参数冻结不可被调整
    for (y, x) in test_dataloader:
        y_hat = model(x.view(x.shape[0], 1, 28, 28).to(device))
        loss = loss_func(y_hat, y.to(device).view(y.shape[0]))
        
        total_loss += loss.item()
    print(f"Total loss: {total_loss/len(test_dataloader): 0.4f}")
    model.train()
    return total_loss/len(test_dataloader)
```
 ```
 #训练模型
epoch = 200

train_loss_records = []
test_loss_records = []

for i in range(epoch):
	train_loss = train_model(lenet, train_dataloader, loss_func, optimizer)
	test_loss = test_model(lenet, test_dataloader, loss_func)

	train_loss_records.append(train_loss)
	test_loss_records.append(test_loss)
	
```
```
 #绘制图像
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']= 'True'

plt.figure(figsize=(8, 4))
plt.plot(train_loss_records, label="train loss")
plt.plot(test_loss_records, label="test loss")
plt.title("Lenet Model Train Loss")
plt.legend()

plt.grid()
plt.show()
```