# NumPy 详解
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


# 创建与现有数组形状相同的数组
x = np.array([[1, 2], [3, 4]])
zeros_like = np.zeros_like(x)
ones_like = np.ones_like(x)
```

### 五、数组属性

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

```

### 十、线性代数

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
