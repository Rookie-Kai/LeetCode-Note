### 1.SoftMax运算
- 对多维`Tensor`按维度操作
  - dim=0:同一列操作
  - dim=1:同一行操作
 ```
 X = torch.tensor([[1, 2, 3], [4, 5, 6]])
 print(X.sum(dim=0, keepdim=True))
 print(X.sum(dim=1, keepdim=True))
 ```
 - 输出为：
   ```
   tensor([[5, 7, 9]])
   tensor([[6], [15]])
   ```
- exp( )函数
  - 对每个元素进行指数运算
 
---

### 2.损失函数
- gather( )函数
  - 沿给定轴 dim ,将输入索引张量 index 指定位置的值进行聚合.
 `torch.gather(input, dim, index, out=None) → Tensor`
   - 参数：
     - input(Tensor) - 源张量
     - dim(int) - 索引的轴 dim=1/0
     - index(LongTensor) - 聚合元素的下标，必须为<code>**torch.LongTensor**</code>类型
     - out(Tensor,optional) - 目标张量
```
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = torch.LongTensor([0, 2])
y_hat.gather(1, y.view(-1, 1))
# gather(1, y.view(-1, 1))中，1是指按行取值
# 因为y = torch.LongTensor([0, 2])，所以y.view(-1, 1) = ([[0],[2]])
# 因此取值为第一行的第0个，第二行的第2个
# 输出为tensor([[0.1000], [0.5000]])
```
- view( )函数
  - 将一个多行的`Tensor`拼为一行
```
# tensor([[0.0000, 0.0000, 0.0000],
#         [0.0000, 0.0000, 0.0000]])
print(a.view(1,-1))
# tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]])
```
---
### 3.计算分类准确率
- 详见如下内容
  - [计算分类准确率](https://tangshusen.me/Dive-into-DL-PyTorch/#/chapter03_DL-basics/3.6_softmax-regression-scratch?id=_366-%e8%ae%a1%e7%ae%97%e5%88%86%e7%b1%bb%e5%87%86%e7%a1%ae%e7%8e%87 "计算分类准确率")
- **mean( )** 函数
  - 求平均值
- **item( )** 函数
  - 一个张量`Tensor`可通过`item( )`函数获得元素值

---

### 4.训练模型
在训练模型时，迭代周期数`num_epochs`和学习率`lr`都是可以调的超参数。改变它们的值可能会得到分类更准确的模型。可用softmax回归做多类别分类

| epoch| loss  | train acc  | test acc  |
| :------------: | :------------: | :------------: | :------------: |
|  周期数 | 损失值  | 训练准确率  | 测试准确率|

- 最开始因为数据随机生成，训练准确率可能低于测试准确率
- 中途随着优化的进行，训练准确率开始高于测试准确率
- 最后由于训练完成，优化趋于完善，训练准确率与测试准确率的值很接近

---

### 5.简洁实现softmax分类
- softmax回归的输出层是一个全连接层，所以用一个线性模块就可以了。因为数据返回的每个batch样本`x`的形状为(batch_size, 1, 28, 28), 故先用`view()`将`x`的形状转换成(batch_size, 784)再送入全连接层。

- `FlattenLayer`对`X`的形状进行转换，将$28 \times 28$的形状变为$784$

- **CrossEntropyLoss( )函数**
 pytorch提供的包括softmax运算和交叉熵损失计算的函数
  - <code>**loss = nn.CrossEntropyLoss()**</code>