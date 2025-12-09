# NumPy 详解

## 一、简介

**NumPy (Numerical Python)** 是Python科学计算的基础库，提供了高性能的多维数组对象和相关工具。

### 特点

* ⚡ 高效的多维数组运算
* 🔢 丰富的数学函数库
* 🚀 比Python原生列表快10-100倍
* 🧮 支持线性代数、傅里叶变换等
* 🔗 其他科学计算库的基础（Pandas、Matplotlib等）

## 二、安装

```bash
pip install numpy
# 或
conda install numpy
```

```python
import numpy as np  # 标准导入方式
print(np.__version__)  # 查看版本
```

## 三、核心概念：ndarray

**ndarray** (N-dimensional array) 是NumPy的核心数据结构。

### ndarray vs Python列表

```python
# Python列表
list_a = [1, 2, 3, 4]

# NumPy数组
arr_a = np.array([1, 2, 3, 4])

# 性能对比
import time

# 列表运算
python_list = list(range(1000000))
start = time.time()
python_list = [x * 2 for x in python_list]
print(f"列表耗时: {time.time() - start:.4f}秒")

# NumPy数组运算
numpy_array = np.arange(1000000)
start = time.time()
numpy_array = numpy_array * 2
print(f"NumPy耗时: {time.time() - start:.4f}秒")
```

## 四、创建数组

### 1. **从Python数据结构创建**

```python
# 从列表创建
arr1 = np.array([1, 2, 3, 4, 5])
print(arr1)  # [1 2 3 4 5]

# 从嵌套列表创建多维数组
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print(arr2)
# [[1 2 3]
#  [4 5 6]]

# 指定数据类型
arr3 = np.array([1, 2, 3], dtype=np.float64)
print(arr3)  # [1. 2. 3.]
```

### 2. **使用内置函数创建**

```python
# 全零数组
zeros = np.zeros((3, 4))
print(zeros)

# 全一数组
ones = np.ones((2, 3))

# 空数组（未初始化）
empty = np.empty((2, 2))

# 单位矩阵
identity = np.eye(3)

# 等差数列
arange_arr = np.arange(0, 10, 2)  # [0 2 4 6 8]

# 线性空间
linspace_arr = np.linspace(0, 1, 5)  # [0.   0.25 0.5  0.75 1.  ]

# 填充特定值
full = np.full((2, 3), 7)  # 全部填充7

# 创建与现有数组形状相同的数组
x = np.array([[1, 2], [3, 4]])
zeros_like = np.zeros_like(x)
ones_like = np.ones_like(x)
```

### 3. **随机数组**

```python
# 随机数（0-1均匀分布）
rand = np.random.rand(3, 3)

# 标准正态分布
randn = np.random.randn(3, 3)

# 指定范围的随机整数
randint = np.random.randint(0, 10, size=(3, 3))

# 设置随机种子
np.random.seed(42)
```

## 五、数组属性

```python
arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]])

print(arr.shape)      # (3, 4) - 形状
print(arr.ndim)       # 2 - 维度数
print(arr.size)       # 12 - 元素总数
print(arr.dtype)      # int64 - 数据类型
print(arr.itemsize)   # 8 - 每个元素字节数
print(arr.nbytes)     # 96 - 总字节数
```

## 六、索引和切片

### 1. **基本索引**

```python
arr = np.array([10, 20, 30, 40, 50])

print(arr[0])      # 10 - 第一个元素
print(arr[-1])     # 50 - 最后一个元素
print(arr[1:4])    # [20 30 40] - 切片
print(arr[::2])    # [10 30 50] - 步长为2
```

### 2. **多维数组索引**

```python
arr2d = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

print(arr2d[0, 0])       # 1 - 第一行第一列
print(arr2d[1])          # [4 5 6] - 第二行
print(arr2d[:, 1])       # [2 5 8] - 第二列
print(arr2d[0:2, 1:3])   # [[2 3] [5 6]] - 切片
```

### 3. **布尔索引**

```python
arr = np.array([1, 2, 3, 4, 5, 6])

# 条件过滤
mask = arr > 3
print(mask)        # [False False False True True True]
print(arr[mask])   # [4 5 6]

# 直接使用条件
print(arr[arr > 3])       # [4 5 6]
print(arr[(arr > 2) & (arr < 5)])  # [3 4]
```

### 4. **花式索引**

```python
arr = np.array([10, 20, 30, 40, 50])

# 使用索引数组
indices = [0, 2, 4]
print(arr[indices])  # [10 30 50]

# 二维数组
arr2d = np.arange(12).reshape(3, 4)
rows = [0, 2]
cols = [1, 3]
print(arr2d[rows, cols])  # [1 11]
```

## 七、数组运算

### 1. **算术运算**

```python
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

print(a + b)      # [11 22 33 44]
print(a - b)      # [-9 -18 -27 -36]
print(a * b)      # [10 40 90 160]
print(a / b)      # [0.1 0.1 0.1 0.1]
print(a ** 2)     # [1 4 9 16]
print(a + 10)     # [11 12 13 14] - 广播
```

### 2. **通用函数（ufunc）**

```python
arr = np.array([1, 4, 9, 16])

print(np.sqrt(arr))       # [1. 2. 3. 4.]
print(np.exp(arr))        # 指数函数
print(np.log(arr))        # 自然对数
print(np.sin(arr))        # 正弦
print(np.abs(arr))        # 绝对值

# 四舍五入
arr = np.array([1.2, 2.5, 3.7])
print(np.round(arr))      # [1. 2. 4.]
print(np.floor(arr))      # [1. 2. 3.]
print(np.ceil(arr))       # [2. 3. 4.]
```

### 3. **聚合函数**

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

print(np.sum(arr))        # 21 - 总和
print(np.mean(arr))       # 3.5 - 平均值
print(np.std(arr))        # 1.707... - 标准差
print(np.var(arr))        # 2.916... - 方差
print(np.min(arr))        # 1 - 最小值
print(np.max(arr))        # 6 - 最大值

# 沿轴计算
print(arr.sum(axis=0))    # [5 7 9] - 按列求和
print(arr.sum(axis=1))    # [6 15] - 按行求和

# 累积函数
print(np.cumsum(arr))     # [ 1  3  6 10 15 21] - 累积和
```

## 八、形状操作

### 1. **改变形状**

```python
arr = np.arange(12)

# reshape - 改变形状
arr2d = arr.reshape(3, 4)
print(arr2d)

# reshape自动计算维度
arr3d = arr.reshape(2, -1, 2)  # -1自动计算

# ravel - 展平为一维
flat = arr2d.ravel()
print(flat)  # [ 0  1  2  3  4  5  6  7  8  9 10 11]

# flatten - 展平（返回副本）
flat_copy = arr2d.flatten()

# 转置
print(arr2d.T)
```

### 2. **合并数组**

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# 垂直堆叠
v_stack = np.vstack([a, b])
print(v_stack)
# [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]]

# 水平堆叠
h_stack = np.hstack([a, b])
print(h_stack)
# [[1 2 5 6]
#  [3 4 7 8]]

# concatenate
concat = np.concatenate([a, b], axis=0)  # 等同于vstack
```

### 3. **分割数组**

```python
arr = np.arange(16).reshape(4, 4)

# 垂直分割
v_split = np.vsplit(arr, 2)

# 水平分割
h_split = np.hsplit(arr, 2)

# split
split_arr = np.split(arr, [1, 3], axis=0)  # 在索引1和3处分割
```

## 九、广播机制

```python
# 不同形状的数组运算
a = np.array([[1, 2, 3],
              [4, 5, 6]])  # (2, 3)

b = np.array([10, 20, 30])  # (3,)

print(a + b)
# [[11 22 33]
#  [14 25 36]]

# 更复杂的广播
a = np.arange(3).reshape(3, 1)  # (3, 1)
b = np.arange(3)                 # (3,)

print(a + b)
# [[0 1 2]
#  [1 2 3]
#  [2 3 4]]
```

## 十、线性代数

```python
# 矩阵乘法
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# 点积
dot_product = np.dot(a, b)
# 或使用 @
dot_product = a @ b

# 转置
transpose = a.T

# 行列式
det = np.linalg.det(a)

# 逆矩阵
inv = np.linalg.inv(a)

# 特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(a)

# 求解线性方程组 Ax = b
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])
x = np.linalg.solve(A, b)
print(x)  # [2. 3.]
```

## 十一、实用技巧

### 1. **条件操作**

```python
arr = np.array([1, 2, 3, 4, 5])

# where - 条件选择
result = np.where(arr > 3, arr, 0)
print(result)  # [0 0 0 4 5]

# 替换值
arr[arr > 3] = 99
print(arr)  # [ 1  2  3 99 99]
```

### 2. **唯一值和计数**

```python
arr = np.array([1, 2, 2, 3, 3, 3, 4])

# 唯一值
unique = np.unique(arr)
print(unique)  # [1 2 3 4]

# 唯一值和计数
values, counts = np.unique(arr, return_counts=True)
print(dict(zip(values, counts)))  # {1: 1, 2: 2, 3: 3, 4: 1}
```

### 3. **数组排序**

```python
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])

# 排序（返回副本）
sorted_arr = np.sort(arr)

# 原地排序
arr.sort()

# 返回排序索引
indices = np.argsort(arr)

# 二维数组排序
arr2d = np.array([[3, 1], [2, 4]])
sorted_2d = np.sort(arr2d, axis=1)  # 按行排序
```

### 4. **数组比较**

```python
a = np.array([1, 2, 3])
b = np.array([1, 2, 4])

# 元素级比较
print(a == b)  # [ True  True False]

# 数组相等
print(np.array_equal(a, b))  # False

# 近似相等
print(np.allclose([1.0, 2.0], [1.0, 2.000001]))  # True
```

## 十二、性能优化建议

```python
# ✅ 好的做法：向量化操作
arr = np.arange(1000000)
result = arr * 2

# ❌ 避免：循环
result = np.array([x * 2 for x in arr])

# ✅ 使用内置函数
np.sum(arr)

# ❌ 避免
sum(arr)

# ✅ 预分配数组
result = np.empty(1000)
for i in range(1000):
    result[i] = i ** 2

# ✅ 使用视图而非副本
view = arr[::2]  # 视图
copy = arr[::2].copy()  # 副本
```

## 十三、常用示例

### 示例1：数据标准化

```python
data = np.random.randn(100, 5)

# Z-score标准化
mean = data.mean(axis=0)
std = data.std(axis=0)
normalized = (data - mean) / std
```

### 示例2：移动平均

```python
def moving_average(arr, window=3):
    return np.convolve(arr, np.ones(window)/window, mode='valid')

data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
ma = moving_average(data, 3)
print(ma)  # [2. 3. 4. 5. 6. 7. 8.]
```

### 示例3：找出最大值的索引

```python
arr = np.array([[1, 5, 3],
                [4, 2, 6]])

# 全局最大值索引
max_idx = np.argmax(arr)
print(max_idx)  # 5

# 转换为二维索引
max_pos = np.unravel_index(max_idx, arr.shape)
print(max_pos)  # (1, 2)
```

## 十四、总结

### 核心要点

1. **ndarray是基础** - 理解数组结构
2. **向量化操作** - 避免显式循环
3. **广播机制** - 高效处理不同形状数组
4. **视图vs副本** - 注意内存使用
5. **丰富的函数库** - 熟悉常用函数

需要我深入讲解某个特定主题吗？
