# Matplotlib 详解

## 一、简介

**Matplotlib** 是Python最流行的数据可视化库，可以创建高质量的静态、动态和交互式图表。

### 特点

* 📊 功能强大，图表类型丰富
* 🎨 高度可定制化
* 📈 类似MATLAB的绘图接口
* 🔧 与NumPy、Pandas无缝集成
* 📖 文档完善，社区活跃

## 二、安装和导入

```python
# 安装
pip install matplotlib

# 标准导入方式
import matplotlib.pyplot as plt
import numpy as np

# Jupyter Notebook中显示图表
%matplotlib inline

# 查看版本
import matplotlib
print(matplotlib.__version__)
```

## 三、基础概念

### 1. 图表结构

```python
"""
Figure（画布）
  └─ Axes（坐标系/子图）
       ├─ x轴（X-axis）
       ├─ y轴（Y-axis）
       ├─ 标题（Title）
       ├─ 图例（Legend）
       └─ 数据图形（Line, Bar, etc.）
"""

# 创建图表的两种方式
# 方式1：pyplot接口（简单快速）
plt.plot([1, 2, 3, 4])
plt.ylabel('数值')
plt.show()

# 方式2：面向对象接口（推荐，更灵活）
fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4])
ax.set_ylabel('数值')
plt.show()
```

### 2. 基本绘图流程

```python
# 1. 准备数据
x = np.linspace(0, 10, 100)
y = np.sin(x)

# 2. 创建图表
plt.figure(figsize=(10, 6))  # 设置图表大小

# 3. 绘制图形
plt.plot(x, y)

# 4. 添加标签和标题
plt.xlabel('X轴')
plt.ylabel('Y轴')
plt.title('正弦函数图')

# 5. 显示图表
plt.show()
```

## 四、基础图表类型

### 1. 折线图 plot()

```python
import matplotlib.pyplot as plt
import numpy as np

# 基本折线图
x = np.linspace(0, 10, 50)
y1 = np.sin(x)
y2 = np.cos(x)

plt.figure(figsize=(10, 6))

# 绘制多条线
plt.plot(x, y1, label='sin(x)')
plt.plot(x, y2, label='cos(x)')

# 添加标签
plt.xlabel('X轴', fontsize=12)
plt.ylabel('Y轴', fontsize=12)
plt.title('三角函数图', fontsize=14)
plt.legend()  # 显示图例
plt.grid(True)  # 显示网格
plt.show()

# 线条样式定制
plt.plot(x, y1, 
         color='red',           # 颜色
         linestyle='--',        # 线型
         linewidth=2,           # 线宽
         marker='o',            # 标记
         markersize=5,          # 标记大小
         label='sin(x)')

# 常用线型
"""
'-'   实线
'--'  虚线
'-.'  点划线
':'   点线
''    无线条
"""

# 常用标记
"""
'o'   圆圈
's'   正方形
'^'   三角形
'*'   星号
'+'   加号
'x'   叉号
'D'   菱形
"""

# 常用颜色
"""
'r'   红色
'g'   绿色
'b'   蓝色
'c'   青色
'm'   品红
'y'   黄色
'k'   黑色
'w'   白色
或使用十六进制：'#FF5733'
"""

# 简写方式
plt.plot(x, y1, 'r--o', label='sin(x)')  # 红色虚线，圆圈标记
```

### 2. 散点图 scatter()

```python
# 基本散点图
x = np.random.rand(50)
y = np.random.rand(50)

plt.figure(figsize=(8, 6))
plt.scatter(x, y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('散点图')
plt.show()

# 高级散点图
x = np.random.rand(100)
y = np.random.rand(100)
colors = np.random.rand(100)  # 颜色值
sizes = 1000 * np.random.rand(100)  # 点大小

plt.scatter(x, y, 
           c=colors,           # 颜色
           s=sizes,            # 大小
           alpha=0.5,          # 透明度
           cmap='viridis',     # 颜色映射
           edgecolors='black', # 边缘颜色
           linewidth=1)

plt.colorbar()  # 显示颜色条
plt.title('彩色散点图')
plt.show()
```

### 3. 柱状图 bar()

```python
# 基本柱状图
categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 78, 32]

plt.figure(figsize=(8, 6))
plt.bar(categories, values)
plt.xlabel('类别')
plt.ylabel('数值')
plt.title('柱状图')
plt.show()

# 水平柱状图
plt.barh(categories, values)
plt.xlabel('数值')
plt.ylabel('类别')
plt.title('水平柱状图')
plt.show()

# 分组柱状图
x = np.arange(len(categories))
values1 = [23, 45, 56, 78, 32]
values2 = [34, 56, 67, 45, 43]

width = 0.35  # 柱子宽度

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, values1, width, label='组1')
plt.bar(x + width/2, values2, width, label='组2')

plt.xlabel('类别')
plt.ylabel('数值')
plt.title('分组柱状图')
plt.xticks(x, categories)
plt.legend()
plt.show()

# 堆叠柱状图
plt.bar(categories, values1, label='组1')
plt.bar(categories, values2, bottom=values1, label='组2')
plt.legend()
plt.title('堆叠柱状图')
plt.show()
```

### 4. 直方图 hist()

```python
# 基本直方图
data = np.random.randn(1000)

plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, edgecolor='black')
plt.xlabel('值')
plt.ylabel('频数')
plt.title('直方图')
plt.show()

# 高级直方图
plt.hist(data, 
         bins=50,              # 箱数
         density=True,         # 归一化
         alpha=0.7,            # 透明度
         color='skyblue',      # 颜色
         edgecolor='black',    # 边缘颜色
         cumulative=False)     # 是否累积

plt.xlabel('值')
plt.ylabel('概率密度')
plt.title('概率密度直方图')
plt.show()

# 多组直方图对比
data1 = np.random.randn(1000)
data2 = np.random.randn(1000) + 2

plt.hist(data1, bins=30, alpha=0.5, label='数据1')
plt.hist(data2, bins=30, alpha=0.5, label='数据2')
plt.legend()
plt.title('多组直方图对比')
plt.show()
```

### 5. 饼图 pie()

```python
# 基本饼图
labels = ['A', 'B', 'C', 'D']
sizes = [15, 30, 45, 10]

plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title('饼图')
plt.show()

# 高级饼图
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
explode = (0, 0.1, 0, 0)  # 突出第二个扇形

plt.pie(sizes, 
        explode=explode,      # 突出显示
        labels=labels,        # 标签
        colors=colors,        # 颜色
        autopct='%1.1f%%',    # 百分比格式
        shadow=True,          # 阴影
        startangle=90)        # 起始角度

plt.axis('equal')  # 保持圆形
plt.title('高级饼图')
plt.show()

# 环形图
plt.pie(sizes, labels=labels, autopct='%1.1f%%',
        wedgeprops=dict(width=0.5))  # 设置宽度
plt.title('环形图')
plt.show()
```

### 6. 箱线图 boxplot()

```python
# 基本箱线图
data = [np.random.normal(0, std, 100) for std in range(1, 4)]

plt.figure(figsize=(10, 6))
plt.boxplot(data, labels=['A', 'B', 'C'])
plt.ylabel('值')
plt.title('箱线图')
plt.show()

# 水平箱线图
plt.boxplot(data, labels=['A', 'B', 'C'], vert=False)
plt.xlabel('值')
plt.title('水平箱线图')
plt.show()

# 美化箱线图
bp = plt.boxplot(data, 
                 labels=['A', 'B', 'C'],
                 patch_artist=True,      # 填充颜色
                 notch=True,             # 显示凹口
                 showmeans=True)         # 显示均值

# 设置颜色
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')

plt.title('美化箱线图')
plt.show()
```

### 7. 热力图 imshow()

```python
# 基本热力图
data = np.random.rand(10, 10)

plt.figure(figsize=(8, 6))
plt.imshow(data, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title('热力图')
plt.show()

# 带标签的热力图
import matplotlib.pyplot as plt
import numpy as np

data = np.random.rand(5, 5)
labels_x = ['A', 'B', 'C', 'D', 'E']
labels_y = ['1', '2', '3', '4', '5']

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(data, cmap='YlOrRd')

# 设置刻度
ax.set_xticks(np.arange(len(labels_x)))
ax.set_yticks(np.arange(len(labels_y)))
ax.set_xticklabels(labels_x)
ax.set_yticklabels(labels_y)

# 添加数值标签
for i in range(len(labels_y)):
    for j in range(len(labels_x)):
        text = ax.text(j, i, f'{data[i, j]:.2f}',
                      ha="center", va="center", color="black")

plt.colorbar(im)
plt.title('带标签的热力图')
plt.show()
```

## 五、图表美化

### 1. 颜色和样式

```python
# 使用样式
plt.style.use('seaborn-v0_8')  # 使用内置样式

# 查看所有可用样式
print(plt.style.available)

# 常用样式
"""
'default'
'classic'
'seaborn-v0_8'
'ggplot'
'bmh'
'fivethirtyeight'
'grayscale'
'dark_background'
"""

# 自定义颜色
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
x = [1, 2, 3, 4]
y = [1, 4, 2, 3]

plt.bar(x, y, color=colors)
plt.show()

# 颜色映射（colormap）
x = np.linspace(0, 10, 100)
for i in range(10):
    plt.plot(x, np.sin(x + i * 0.5), 
            color=plt.cm.viridis(i / 10))

plt.show()

# 常用colormap
"""
'viridis', 'plasma', 'inferno', 'magma'  # 感知均匀
'coolwarm', 'RdYlBu'                     # 发散
'Greys', 'Blues', 'Reds'                 # 顺序
'rainbow', 'jet'                          # 彩虹
"""
```

### 2. 标题和标签

```python
fig, ax = plt.subplots(figsize=(10, 6))

x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x))

# 标题
ax.set_title('这是标题', 
            fontsize=16,           # 字体大小
            fontweight='bold',     # 粗体
            color='navy',          # 颜色
            pad=20)                # 与图的距离

# 坐标轴标签
ax.set_xlabel('X轴标签', fontsize=14, fontweight='bold')
ax.set_ylabel('Y轴标签', fontsize=14, fontweight='bold')

# 刻度标签
ax.tick_params(axis='both', labelsize=12)

# 中文显示（重要！）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False     # 负号显示

plt.show()
```

### 3. 图例

```python
x = np.linspace(0, 10, 100)

plt.figure(figsize=(10, 6))
plt.plot(x, np.sin(x), label='sin(x)')
plt.plot(x, np.cos(x), label='cos(x)')
plt.plot(x, np.tan(x), label='tan(x)')

# 图例位置
plt.legend(loc='best')  # 自动选择最佳位置

# 图例位置选项
"""
'best'          自动
'upper right'   右上
'upper left'    左上
'lower left'    左下
'lower right'   右下
'right'         右侧
'center left'   左侧中心
'center right'  右侧中心
'lower center'  底部中心
'upper center'  顶部中心
'center'        中心
"""

# 图例美化
plt.legend(loc='upper right',
          fontsize=12,           # 字体大小
          frameon=True,          # 边框
          shadow=True,           # 阴影
          fancybox=True,         # 圆角
          ncol=2)                # 列数

plt.show()
```

### 4. 网格和边框

```python
fig, ax = plt.subplots(figsize=(10, 6))

x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x))

# 网格
ax.grid(True,                # 显示网格
       linestyle='--',       # 线型
       linewidth=0.5,        # 线宽
       alpha=0.7,            # 透明度
       color='gray')         # 颜色

# 主次网格
ax.grid(which='major', linestyle='-', linewidth=0.8)
ax.grid(which='minor', linestyle=':', linewidth=0.5)
ax.minorticks_on()  # 显示次刻度

# 边框样式
ax.spines['top'].set_visible(False)     # 隐藏顶部边框
ax.spines['right'].set_visible(False)   # 隐藏右侧边框

plt.show()
```

### 5. 坐标轴设置

```python
fig, ax = plt.subplots(figsize=(10, 6))

x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x))

# 设置坐标轴范围
ax.set_xlim(0, 10)
ax.set_ylim(-1.5, 1.5)

# 设置刻度
ax.set_xticks([0, 2, 4, 6, 8, 10])
ax.set_yticks([-1, -0.5, 0, 0.5, 1])

# 自定义刻度标签
ax.set_xticklabels(['零', '二', '四', '六', '八', '十'])

# 对数坐标
ax.set_xscale('log')
ax.set_yscale('log')

# 反转坐标轴
ax.invert_xaxis()
ax.invert_yaxis()

plt.show()
```

## 六、子图布局

### 1. subplot() 基础子图

```python
# 2x2子图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

x = np.linspace(0, 10, 100)

axes[0, 0].plot(x, np.sin(x))
axes[0, 0].set_title('sin(x)')

axes[0, 1].plot(x, np.cos(x))
axes[0, 1].set_title('cos(x)')

axes[1, 0].plot(x, np.tan(x))
axes[1, 0].set_title('tan(x)')

axes[1, 1].plot(x, x**2)
axes[1, 1].set_title('x²')

plt.tight_layout()  # 自动调整间距
plt.show()
```

### 2. subplot2grid() 复杂布局

```python
fig = plt.figure(figsize=(12, 8))

# 创建不同大小的子图
ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)
ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
ax4 = plt.subplot2grid((3, 3), (2, 0))
ax5 = plt.subplot2grid((3, 3), (2, 1))

x = np.linspace(0, 10, 100)
ax1.plot(x, np.sin(x))
ax2.plot(x, np.cos(x))
ax3.plot(x, np.tan(x))
ax4.plot(x, x)
ax5.plot(x, x**2)

plt.tight_layout()
plt.show()
```

### 3. GridSpec 高级布局

```python
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(12, 8))
gs = GridSpec(3, 3, figure=fig)

ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, :-1])
ax3 = fig.add_subplot(gs[1:, -1])
ax4 = fig.add_subplot(gs[-1, 0])
ax5 = fig.add_subplot(gs[-1, 1])

x = np.linspace(0, 10, 100)
ax1.plot(x, np.sin(x))
ax2.plot(x, np.cos(x))
ax3.plot(x, np.tan(x))
ax4.plot(x, x)
ax5.plot(x, x**2)

plt.tight_layout()
plt.show()
```

### 4. 嵌套子图

```python
fig = plt.figure(figsize=(12, 8))

# 主图
ax1 = plt.subplot(1, 1, 1)
x = np.linspace(0, 10, 100)
ax1.plot(x, np.sin(x))

# 嵌入小图
ax2 = fig.add_axes([0.6, 0.6, 0.25, 0.25])  # [left, bottom, width, height]
ax2.plot(x, np.cos(x), 'r')
ax2.set_title('放大图')

plt.show()
```

## 七、高级功能

### 1. 双Y轴

```python
fig, ax1 = plt.subplots(figsize=(10, 6))

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.exp(x / 5)

# 第一个Y轴
ax1.set_xlabel('X')
ax1.set_ylabel('sin(x)', color='b')
ax1.plot(x, y1, 'b-', label='sin(x)')
ax1.tick_params(axis='y', labelcolor='b')

# 第二个Y轴
ax2 = ax1.twinx()
ax2.set_ylabel('exp(x/5)', color='r')
ax2.plot(x, y2, 'r-', label='exp(x/5)')
ax2.tick_params(axis='y', labelcolor='r')

plt.title('双Y轴图表')
fig.tight_layout()
plt.show()
```

### 2. 填充区域

```python
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.figure(figsize=(10, 6))

# 填充两条线之间的区域
plt.fill_between(x, y1, y2, alpha=0.3, label='填充区域')
plt.plot(x, y1, label='sin(x)')
plt.plot(x, y2, label='cos(x)')

plt.legend()
plt.title('填充区域图')
plt.show()

# 条件填充
plt.figure(figsize=(10, 6))
plt.plot(x, y1)
plt.fill_between(x, 0, y1, where=(y1 > 0), 
                alpha=0.3, color='green', label='正值')
plt.fill_between(x, 0, y1, where=(y1 < 0), 
                alpha=0.3, color='red', label='负值')
plt.legend()
plt.show()
```

### 3. 误差条

```python
x = np.arange(0, 10, 1)
y = np.sin(x)
error = 0.1 + 0.2 * np.random.rand(len(x))

plt.figure(figsize=(10, 6))
plt.errorbar(x, y, yerr=error, 
            fmt='o-',              # 格式
            ecolor='red',          # 误差条颜色
            elinewidth=2,          # 误差条线宽
            capsize=5,             # 误差条帽大小
            capthick=2,            # 误差条帽粗细
            label='数据点')

plt.legend()
plt.title('误差条图')
plt.show()
```

### 4. 注释和箭头

```python
fig, ax = plt.subplots(figsize=(10, 6))

x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x))

# 添加文本
ax.text(5, 0.5, '这是文本', fontsize=12)

# 添加箭头注释
ax.annotate('最大值', 
           xy=(np.pi/2, 1),         # 箭头指向的点
           xytext=(3, 1.2),         # 文本位置
           arrowprops=dict(
               facecolor='red',
               shrink=0.05,
               width=2,
               headwidth=8),
           fontsize=12)

# 添加标注框
ax.annotate('重要点', 
           xy=(np.pi, 0),
           xytext=(5, -0.5),
           bbox=dict(boxstyle='round', 
                    facecolor='yellow', 
                    alpha=0.5),
           arrowprops=dict(arrowstyle='->'))

plt.title('注释示例')
plt.show()
```

### 5. 极坐标图

```python
# 极坐标图
theta = np.linspace(0, 2*np.pi, 100)
r = 1 + np.sin(theta)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='polar')
ax.plot(theta, r)
ax.set_title('极坐标图')
plt.show()

# 玫瑰图
theta = np.linspace(0, 2*np.pi, 8, endpoint=False)
radii = np.array([3, 5, 2, 4, 6, 7, 3, 4])
width = 2*np.pi / len(radii)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='polar')
bars = ax.bar(theta, radii, width=width, bottom=0.0)

# 自定义颜色
for r, bar in zip(radii, bars):
    bar.set_facecolor(plt.cm.viridis(r / 10))
    bar.set_alpha(0.8)

plt.title('玫瑰图')
plt.show()
```

## 八、保存图表

```python
fig, ax = plt.subplots(figsize=(10, 6))
x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x))

# 保存图片
plt.savefig('figure.png',           # 文件名
           dpi=300,                 # 分辨率
           bbox_inches='tight',     # 紧凑布局
           transparent=False,       # 透明背景
           facecolor='white')       # 背景颜色

# 支持的格式
"""
.png  - PNG格式（推荐）
.jpg  - JPEG格式
.pdf  - PDF格式
.svg  - SVG矢量图
.eps  - EPS格式
"""

plt.show()
```

## 九、实用示例

### 示例1：数据对比图

```python
# 多组数据对比
categories = ['1月', '2月', '3月', '4月', '5月']
product_A = [23, 45, 56, 78, 90]
product_B = [34, 56, 67, 45, 67]
product_C = [12, 23, 34, 45, 56]

x = np.arange(len(categories))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))

bars1 = ax.bar(x - width, product_A, width, label='产品A', color='#FF6B6B')
bars2 = ax.bar(x, product_B, width, label='产品B', color='#4ECDC4')
bars3 = ax.bar(x + width, product_C, width, label='产品C', color='#45B7D1')

# 在柱子上添加数值
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.0f}',
               ha='center', va='bottom', fontsize=9)

ax.set_xlabel('月份', fontsize=12)
ax.set_ylabel('销量', fontsize=12)
ax.set_title('产品销量对比图', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
```

### 示例2：时间序列图

```python
import pandas as pd

# 生成时间序列数据
dates = pd.date_range('2024-01-01', periods=100)
values = np.cumsum(np.random.randn(100)) + 100

fig, ax = plt.subplots(figsize=(14, 6))

# 绘制主线
ax.plot(dates, values, linewidth=2, color='#2E86AB')

# 添加填充
ax.fill_between(dates, values, alpha=0.3, color='#2E86AB')

# 添加移动平均线
window = 10
moving_avg = pd.Series(values).rolling(window=window).mean()
ax.plot(dates, moving_avg, linewidth=2, 
       color='red', linestyle='--', label=f'{window}日移动平均')

ax.set_xlabel('日期', fontsize=12)
ax.set_ylabel('数值', fontsize=12)
ax.set_title
```
