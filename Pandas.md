# Pandas 详解（完整版）

## 一、简介

**Pandas** 是Python中最强大的数据分析和处理库，提供了高效的数据结构和数据分析工具。

### 特点

* 📊 强大的数据结构：Series和DataFrame
* 🔧 高效的数据处理能力
* 📈 与NumPy、Matplotlib无缝集成
* 📁 支持多种文件格式读写
* 🚀 适合处理表格数据和时间序列

## 二、安装和导入

```python
# 安装
pip install pandas

# 标准导入方式
import pandas as pd
import numpy as np

# 查看版本
print(pd.__version__)

# 设置显示选项
pd.set_option('display.max_rows', 100)      # 最多显示100行
pd.set_option('display.max_columns', 50)    # 最多显示50列
pd.set_option('display.width', 1000)        # 显示宽度
pd.set_option('display.precision', 2)       # 小数精度
```

## 三、核心数据结构

### 1. Series（一维数据）

```python
# 创建Series
# 方法1：从列表创建
s1 = pd.Series([1, 2, 3, 4, 5])
print(s1)
"""
0    1
1    2
2    3
3    4
4    5
dtype: int64
"""

# 方法2：指定索引
s2 = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])
print(s2)
"""
a    1
b    2
c    3
d    4
e    5
dtype: int64
"""

# 方法3：从字典创建
data_dict = {'a': 1, 'b': 2, 'c': 3}
s3 = pd.Series(data_dict)

# 方法4：从标量创建
s4 = pd.Series(5, index=['a', 'b', 'c'])
print(s4)
"""
a    5
b    5
c    5
dtype: int64
"""

# Series属性
s = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])

print(s.values)     # array([1, 2, 3, 4, 5])
print(s.index)      # Index(['a', 'b', 'c', 'd', 'e'])
print(s.dtype)      # int64
print(s.shape)      # (5,)
print(s.size)       # 5
print(s.name)       # None

s.name = '数值'     # 设置名称

# Series索引（重要！）
s = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])

print(s['a'])           # 1 - 标签索引
print(s[0])             # 1 - 位置索引
print(s[['a', 'c']])    # 多个标签
print(s[0:3])           # 切片
print(s['a':'c'])       # 标签切片（包含结束）

# Series运算
s = pd.Series([1, 2, 3, 4, 5])

print(s + 10)           # 加法
print(s * 2)            # 乘法
print(s ** 2)           # 幂运算
print(s > 2)            # 布尔运算
print(s[s > 2])         # 条件过滤

# Series方法
s = pd.Series([1, 2, 3, 4, 5, 2, 3])

print(s.sum())          # 20 求和
print(s.mean())         # 2.857... 平均值
print(s.std())          # 标准差
print(s.min())          # 1 最小值
print(s.max())          # 5 最大值
print(s.median())       # 3.0 中位数
print(s.unique())       # 唯一值
print(s.value_counts()) # 值计数
print(s.isnull())       # 是否为空
print(s.notnull())      # 是否非空
```

### 2. DataFrame（二维数据）

```python
# 创建DataFrame
# 方法1：从字典创建（推荐）
data = {
    'name': ['张三', '李四', '王五', '赵六'],
    'age': [25, 30, 35, 28],
    'city': ['北京', '上海', '广州', '深圳'],
    'salary': [8000, 12000, 15000, 10000]
}
df1 = pd.DataFrame(data)
print(df1)
"""
  name  age city  salary
0   张三   25   北京    8000
1   李四   30   上海   12000
2   王五   35   广州   15000
3   赵六   28   深圳   10000
"""

# 方法2：从列表的列表创建
data = [
    ['张三', 25, '北京'],
    ['李四', 30, '上海'],
    ['王五', 35, '广州']
]
df2 = pd.DataFrame(data, columns=['name', 'age', 'city'])

# 方法3：从NumPy数组创建
arr = np.random.rand(4, 3)
df3 = pd.DataFrame(arr, columns=['A', 'B', 'C'])

# 方法4：从Series字典创建
data = {
    'col1': pd.Series([1, 2, 3]),
    'col2': pd.Series([4, 5, 6])
}
df4 = pd.DataFrame(data)

# 指定索引
df = pd.DataFrame(data, index=['row1', 'row2', 'row3', 'row4'])

# DataFrame属性（重要！）
df = pd.DataFrame(data)

print(df.shape)         # (4, 4) 形状
print(df.size)          # 16 元素总数
print(df.ndim)          # 2 维度
print(df.columns)       # 列名
print(df.index)         # 索引
print(df.dtypes)        # 数据类型
print(df.values)        # NumPy数组
print(df.info())        # 信息概览
print(df.describe())    # 统计描述

# 查看数据
print(df.head())        # 前5行
print(df.head(3))       # 前3行
print(df.tail())        # 后5行
print(df.tail(3))       # 后3行
print(df.sample(2))     # 随机2行
```

## 四、数据选择和索引（重要！考试高频）

### 1. 列选择

```python
df = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [5, 6, 7, 8],
    'C': [9, 10, 11, 12]
})

# 单列选择（返回Series）
print(df['A'])
print(df.A)             # 点语法（列名不能有空格）

# 多列选择（返回DataFrame）
print(df[['A', 'C']])

# 添加新列
df['D'] = df['A'] + df['B']
df['E'] = 100

# 删除列
df.drop('E', axis=1, inplace=True)  # axis=1表示列
del df['D']                          # 直接删除
```

### 2. 行选择

```python
df = pd.DataFrame({
    'name': ['张三', '李四', '王五', '赵六'],
    'age': [25, 30, 35, 28],
    'salary': [8000, 12000, 15000, 10000]
}, index=['a', 'b', 'c', 'd'])

# 切片选择
print(df[0:2])          # 前2行
print(df['a':'c'])      # 标签切片（包含结束）

# 条件过滤（重要！）
print(df[df['age'] > 28])               # 年龄>28
print(df[df['salary'] >= 10000])        # 薪水>=10000

# 多条件过滤
print(df[(df['age'] > 25) & (df['salary'] > 10000)])  # 与
print(df[(df['age'] < 26) | (df['salary'] > 14000)])  # 或

# isin方法
print(df[df['name'].isin(['张三', '李四'])])

# 字符串方法
df_str = pd.DataFrame({
    'name': ['张三', '李四', '王五']
})
print(df_str[df_str['name'].str.contains('张')])
```

### 3. loc和iloc（重要！考试必考）

```python
df = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [5, 6, 7, 8],
    'C': [9, 10, 11, 12]
}, index=['row1', 'row2', 'row3', 'row4'])

# loc - 基于标签的索引
print(df.loc['row1'])                   # 单行
print(df.loc['row1', 'A'])              # 单个值
print(df.loc['row1':'row3'])            # 行切片
print(df.loc['row1':'row3', 'A':'B'])   # 行列切片
print(df.loc[:, 'A'])                   # 所有行，A列
print(df.loc['row1', :])                # row1行，所有列
print(df.loc[['row1', 'row3'], ['A', 'C']])  # 指定多行多列

# iloc - 基于位置的索引
print(df.iloc[0])                       # 第1行
print(df.iloc[0, 0])                    # 第1行第1列
print(df.iloc[0:2])                     # 前2行
print(df.iloc[0:2, 0:2])                # 前2行前2列
print(df.iloc[:, 0])                    # 所有行，第1列
print(df.iloc[0, :])                    # 第1行，所有列
print(df.iloc[[0, 2], [0, 2]])          # 指定位置

# at和iat - 快速访问单个值
print(df.at['row1', 'A'])               # 标签
print(df.iat[0, 0])                     # 位置

# 条件选择
print(df.loc[df['A'] > 2])              # 条件过滤
print(df.loc[df['A'] > 2, ['A', 'B']])  # 条件+列选择
```

### 4. 修改数据

```python
df = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [5, 6, 7, 8]
})

# 修改单个值
df.loc[0, 'A'] = 100
df.iloc[1, 1] = 200

# 修改整列
df['A'] = [10, 20, 30, 40]

# 修改整行
df.loc[0] = [100, 200]

# 条件修改
df.loc[df['A'] > 20, 'B'] = 999

# where方法
df['A'] = df['A'].where(df['A'] > 20, 0)  # 小于20的设为0
```

## 五、数据读取和保存

### 1. CSV文件

```python
# 读取CSV
df = pd.read_csv('data.csv')

# 常用参数
df = pd.read_csv('data.csv',
                sep=',',              # 分隔符
                header=0,             # 表头行号
                index_col=0,          # 索引列
                names=['A', 'B'],     # 列名
                encoding='utf-8',     # 编码
                nrows=100,            # 读取行数
                skiprows=[1, 2],      # 跳过的行
                na_values=['NA', ''])  # 空值表示

# 保存CSV
df.to_csv('output.csv', 
         index=False,          # 不保存索引
         encoding='utf-8-sig', # 编码（避免中文乱码）
         sep=',')              # 分隔符
```

### 2. Excel文件

```python
# 读取Excel
df = pd.read_excel('data.xlsx')

# 指定工作表
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# 读取多个工作表
dfs = pd.read_excel('data.xlsx', sheet_name=['Sheet1', 'Sheet2'])

# 保存Excel
df.to_excel('output.xlsx', 
           sheet_name='数据', 
           index=False)

# 保存多个工作表
with pd.ExcelWriter('output.xlsx') as writer:
    df1.to_excel(writer, sheet_name='Sheet1')
    df2.to_excel(writer, sheet_name='Sheet2')
```

### 3. 其他格式

```python
# JSON
df = pd.read_json('data.json')
df.to_json('output.json', orient='records')

# SQL数据库
import sqlite3
conn = sqlite3.connect('database.db')
df = pd.read_sql('SELECT * FROM table', conn)
df.to_sql('table_name', conn, if_exists='replace')

# HTML
df = pd.read_html('http://example.com')[0]
df.to_html('output.html')

# 剪贴板
df = pd.read_clipboard()
df.to_clipboard()
```

## 六、数据清洗

### 1. 缺失值处理（重要！）

```python
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4],
    'B': [5, np.nan, np.nan, 8],
    'C': [9, 10, 11, 12]
})

# 检测缺失值
print(df.isnull())          # 是否为空
print(df.notnull())         # 是否非空
print(df.isnull().sum())    # 每列缺失值数量
print(df.info())            # 查看非空值数量

# 删除缺失值
df.dropna()                 # 删除含有缺失值的行
df.dropna(axis=1)           # 删除含有缺失值的列
df.dropna(how='all')        # 删除全为空的行
df.dropna(thresh=2)         # 至少有2个非空值的行
df.dropna(subset=['A'])     # 删除A列为空的行

# 填充缺失值
df.fillna(0)                # 用0填充
df.fillna({'A': 0, 'B': 100})  # 不同列不同值
df.fillna(method='ffill')   # 前向填充
df.fillna(method='bfill')   # 后向填充
df['A'].fillna(df['A'].mean())  # 用均值填充

# 插值
df.interpolate()            # 线性插值
df.interpolate(method='polynomial', order=2)  # 多项式插值
```

### 2. 重复值处理

```python
df = pd.DataFrame({
    'A': [1, 2, 2, 3, 3],
    'B': [5, 6, 6, 7, 7]
})

# 检测重复值
print(df.duplicated())      # 标记重复行
print(df.duplicated(subset=['A']))  # 基于某列检测

# 删除重复值
df.drop_duplicates()        # 删除重复行
df.drop_duplicates(subset=['A'])  # 基于某列删除
df.drop_duplicates(keep='first')   # 保留第一个
df.drop_duplicates(keep='last')    # 保留最后一个
df.drop_duplicates(keep=False)     # 全部删除
```

### 3. 数据类型转换

```python
df = pd.DataFrame({
    'A': ['1', '2', '3'],
    'B': ['4.5', '5.6', '6.7'],
    'C': ['2024-01-01', '2024-01-02', '2024-01-03']
})

# 转换数据类型
df['A'] = df['A'].astype(int)
df['B'] = df['B'].astype(float)
df['C'] = pd.to_datetime(df['C'])

# 批量转换
df = df.astype({'A': int, 'B': float})

# 转换为分类类型
df['category'] = df['category'].astype('category')

# 查看数据类型
print(df.dtypes)
```

### 4. 字符串处理

```python
df = pd.DataFrame({
    'name': ['  Zhang San  ', 'Li Si', 'WANG WU']
})

# 字符串方法
df['name'].str.lower()          # 转小写
df['name'].str.upper()          # 转大写
df['name'].str.strip()          # 去除空格
df['name'].str.replace('a', 'A')  # 替换
df['name'].str.split()          # 分割
df['name'].str.contains('Zhang')  # 包含
df['name'].str.startswith('L')   # 开头
df['name'].str.endswith('i')     # 结尾
df['name'].str.len()            # 长度
df['name'].str[0]               # 切片

# 正则表达式
df['name'].str.extract(r'(\w+)')  # 提取
df['name'].str.match(r'\w+')      # 匹配
```

### 5. 异常值处理

```python
df = pd.DataFrame({
    'value': [1, 2, 3, 100, 4, 5, 200, 6]
})

# 使用IQR方法检测异常值
Q1 = df['value'].quantile(0.25)
Q3 = df['value'].quantile(0.75)
IQR = Q3 - Q1

# 定义异常值范围
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 过滤异常值
df_clean = df[(df['value'] >= lower_bound) & (df['value'] <= upper_bound)]

# 使用Z-score方法
from scipy import stats
df['z_score'] = np.abs(stats.zscore(df['value']))
df_clean = df[df['z_score'] < 3]
```

## 七、数据操作

### 1. 排序

```python
df = pd.DataFrame({
    'A': [3, 1, 2],
    'B': [6, 4, 5],
    'C': [9, 7, 8]
})

# 按值排序
df.sort_values('A')                 # 按A列升序
df.sort_values('A', ascending=False)  # 降序
df.sort_values(['A', 'B'])          # 多列排序
df.sort_values(['A', 'B'], ascending=[True, False])

# 按索引排序
df.sort_index()                     # 索引升序
df.sort_index(ascending=False)      # 索引降序
```

### 2. 重命名

```python
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})

# 重命名列
df.rename(columns={'A': 'col1', 'B': 'col2'})

# 重命名索引
df.rename(index={0: 'row1', 1: 'row2', 2: 'row3'})

# 直接设置列名
df.columns = ['col1', 'col2']

# 直接设置索引
df.index = ['row1', 'row2', 'row3']
```

### 3. 添加和删除

```python
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})

# 添加列
df['C'] = [7, 8, 9]
df.insert(1, 'D', [10, 11, 12])  # 在位置1插入

# 删除列
df.drop('C', axis=1, inplace=True)
df.drop(['A', 'B'], axis=1)

# 添加行
new_row = pd.DataFrame({'A': [4], 'B': [7]})
df = pd.concat([df, new_row], ignore_index=True)

# 删除行
df.drop(0, axis=0, inplace=True)
df.drop([0, 1], axis=0)
```

### 4. 应用函数

```python
df = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': [5, 6, 7, 8]
})

# apply - 应用函数
df['A'].apply(lambda x: x ** 2)         # 对列应用
df.apply(lambda x: x.sum(), axis=0)     # 对列应用
df.apply(lambda x: x.sum(), axis=1)     # 对行应用

# map - 映射（仅Series）
df['A'].map({1: 'one', 2: 'two', 3: 'three', 4: 'four'})
df['A'].map(lambda x: x * 10)

# applymap - 对每个元素应用（DataFrame）
df.applymap(lambda x: x * 2)

# 自定义函数
def custom_function(x):
    if x > 2:
        return 'high'
    else:
        return 'low'

df['A'].apply(custom_function)
```

### 5. 替换值

```python
df = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': ['a', 'b', 'c', 'd']
})

# 替换单个值
df.replace(1, 100)

# 替换多个值
df.replace([1, 2], [100, 200])

# 字典替换
df.replace({'A': {1: 100, 2: 200}})

# 正则替换
df.replace(r'\d+', 'number', regex=True)
```

## 八、分组和聚合（重要！）

### 1. groupby基础

```python
df = pd.DataFrame({
    'department': ['销售', '销售', '技术', '技术', '人事'],
    'name': ['张三', '李四', '王五', '赵六', '钱七'],
    'salary': [8000, 12000, 15000, 18000, 10000],
    'age': [25, 30, 35, 28, 32]
})

# 单列分组
grouped = df.groupby('department')

# 查看分组
for name, group in grouped:
    print(f"组名：{name}")
    print(group)
    print()

# 聚合函数
print(grouped.sum())        # 求和
print(grouped.mean())       # 平均值
print(grouped.count())      # 计数
print(grouped.min())        # 最小值
print(grouped.max())        # 最大值
print(grouped.std())        # 标准差
print(grouped.median())     # 中位数

# 单列聚合
print(grouped['salary'].mean())

# 多列分组
df.groupby(['department', 'age']).mean()
```

### 2. agg聚合（重要！）

```python
# 单个聚合函数
grouped['salary'].agg('mean')

# 多个聚合函数
grouped['salary'].agg(['mean', 'sum', 'count'])

# 不同列不同函数
grouped.agg({
    'salary': ['mean', 'sum'],
    'age': ['min', 'max']
})

# 自定义聚合函数
def range_func(x):
    return x.max() - x.min()

grouped['salary'].agg(['mean', range_func])

# 重命名聚合列
grouped['salary'].agg([
    ('平均工资', 'mean'),
    ('总工资', 'sum')
])
```

### 3. transform和filter

```python
# transform - 保持原始形状
df['salary_mean'] = grouped['salary'].transform('mean')

# 计算与组均值的差值
df['diff_from_mean'] = df['salary'] - grouped['salary'].transform('mean')

# filter - 过滤组
# 只保留平均工资>10000的组
df_filtered = grouped.filter(lambda x: x['salary'].mean() > 10000)
```

### 4. pivot_table透视表

```python
df = pd.DataFrame({
    'date': ['2024-01', '2024-01', '2024-02', '2024-02'],
    'city': ['北京', '上海', '北京', '上海'],
    'sales': [100, 150, 120, 180],
    'profit': [20, 30, 25, 35]
})

# 创建透视表
pivot = pd.pivot_table(df,
                      values='sales',        # 值列
                      index='date',          # 行索引
                      columns='city',        # 列索引
                      aggfunc='sum')         # 聚合函数

print(pivot)
"""
city      北京   上海
date            
2024-01  100  150
2024-02  120  180
"""

# 多个值列
pivot = pd.pivot_table(df,
                      values=['sales', 'profit'],
                      index='date',
                      columns='city',
                      aggfunc='sum')

# 多个聚合函数
pivot = pd.pivot_table(df,
                      values='sales',
                      index='date',
                      columns='city',
                      aggfunc=['sum', 'mean'])

# 添加边际合计
pivot = pd.pivot_table(df,
                      values='sales',
                      index='date',
                      columns='city',
                      aggfunc='sum',
                      margins=True)
```

## 九、数据合并

### 1. concat连接

```python
df1 = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})

df2 = pd.DataFrame({
    'A': [7, 8, 9],
    'B': [10, 11, 12]
})

# 垂直连接（默认）
result = pd.concat([df1, df2])
result = pd.concat([df1, df2], ignore_index=True)  # 重置索引

# 水平连接
result = pd.concat([df1, df2], axis=1)

# 指定键
result = pd.concat([df1, df2], keys=['df1', 'df2'])
```

### 2. merge合并（重要！类似SQL的JOIN）

```python
df1 = pd.DataFrame({
    'key': ['A', 'B', 'C', 'D'],
    'value1': [1, 2, 3, 4]
})

df2 = pd.DataFrame({
    'key': ['B', 'D', 'E', 'F'],
    'value2': [5, 6, 7, 8]
})

# 内连接（默认）
result = pd.merge(df1, df2, on='key')
# 结果：只保留共同的键B和D

# 左连接
result = pd.merge(df1, df2, on='key', how='left')
# 结果：保留df1所有行

# 右连接
result = pd.merge(df1, df2, on='key', how='right')
# 结果：保留df2所有行

# 外连接
result = pd.merge(df1, df2, on='key', how='outer')
# 结果：保留所有行

# 不同列名合并
df1 = pd.DataFrame({'key1': ['A', 'B'], 'value': [1, 2]})
df2 = pd.DataFrame({'key2': ['A', 'B'], 'value': [3, 4]})
result = pd.merge(df1, df2, left_on='key1', right_on='key2')

# 多列合并
result = pd.merge(df1, df2, on=['key1', 'key2'])

# 使用索引合并
result = pd.merge(df1, df2, left_index=True, right_index=True)
```

### 3. join连接

```python
df1 = pd.DataFrame({
    'A': [1, 2, 3]
}, index=['a', 'b', 'c'])

df2 = pd.DataFrame({
    'B': [4, 5, 6]
}, index=['a', 'b', 'd'])

# 默认左连接
result = df1.join(df2)

# 指定连接方式
result = df1.join(df)
```
