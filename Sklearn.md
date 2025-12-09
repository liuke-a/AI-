# Scikit-learn (sklearn) 详解

## 一、简介

**Scikit-learn** 是Python中最流行的机器学习库，提供了简单高效的数据挖掘和数据分析工具。

### 特点

* 🎯 简单易用的API设计
* 📊 涵盖大部分经典机器学习算法
* 🔧 与NumPy、Pandas无缝集成
* 📖 文档完善，社区活跃
* 🆓 开源免费（BSD许可证）

## 二、安装

```bash
pip install scikit-learn
# 或
conda install scikit-learn
```

## 三、核心模块

### 1. **监督学习算法**

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 分类算法
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# 回归算法
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
```

### 2. **无监督学习算法**

```python
# 聚类
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

# 降维
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
```

### 3. **数据预处理**

```python
from sklearn.preprocessing import (
    StandardScaler,      # 标准化
    MinMaxScaler,        # 归一化
    LabelEncoder,        # 标签编码
    OneHotEncoder        # 独热编码
)

from sklearn.impute import SimpleImputer  # 缺失值处理
```

### 4. **模型选择与评估**

```python
from sklearn.model_selection import (
    train_test_split,    # 数据集划分
    cross_val_score,     # 交叉验证
    GridSearchCV,        # 网格搜索
    RandomizedSearchCV   # 随机搜索
)

from sklearn.metrics import (
    accuracy_score,      # 准确率
    precision_score,     # 精确率
    recall_score,        # 召回率
    f1_score,           # F1分数
    confusion_matrix,    # 混淆矩阵
    roc_auc_score       # AUC值
)
```

## 四、典型工作流程

### 完整示例：分类任务

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. 加载数据
iris = load_iris()
X, y = iris.data, iris.target

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. 数据预处理
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. 训练模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 5. 预测
y_pred = model.predict(X_test_scaled)

# 6. 评估
print("混淆矩阵:\n", confusion_matrix(y_test, y_pred))
print("\n分类报告:\n", classification_report(y_test, y_pred))
print("准确率:", model.score(X_test_scaled, y_test))
```

### 回归任务示例

```python
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 生成数据
X, y = make_regression(n_samples=100, n_features=1, noise=10)

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 训练
model = LinearRegression()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))
```

## 五、高级功能

### 1. **Pipeline（管道）**

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 创建管道
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC())
])

# 一步完成训练
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
```

### 2. **网格搜索调参**

```python
from sklearn.model_selection import GridSearchCV

# 定义参数网格
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.001, 0.01, 0.1],
    'kernel': ['rbf', 'linear']
}

# 网格搜索
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("最佳参数:", grid_search.best_params_)
print("最佳分数:", grid_search.best_score_)
```

### 3. **交叉验证**

```python
from sklearn.model_selection import cross_val_score

model = RandomForestClassifier()
scores = cross_val_score(model, X, y, cv=5)

print("交叉验证分数:", scores)
print("平均分数:", scores.mean())
```

### 4. **特征工程**

```python
from sklearn.feature_selection import SelectKBest, f_classif

# 选择K个最佳特征
selector = SelectKBest(f_classif, k=2)
X_new = selector.fit_transform(X, y)

# 查看被选择的特征
selected_features = selector.get_support(indices=True)
print("选择的特征索引:", selected_features)
```

## 六、常用算法对比

| 算法类型     | 适用场景  | 优点    | 缺点    |
| -------- | ----- | ----- | ----- |
| **逻辑回归** | 二分类   | 简单快速  | 线性假设  |
| **决策树**  | 分类/回归 | 可解释性强 | 易过拟合  |
| **随机森林** | 分类/回归 | 准确率高  | 训练慢   |
| **SVM**  | 高维数据  | 效果好   | 大数据集慢 |
| **KNN**  | 小数据集  | 无需训练  | 预测慢   |

## 七、实用技巧

### 1. **保存和加载模型**

```python
import joblib

# 保存
joblib.dump(model, 'model.pkl')

# 加载
model = joblib.load('model.pkl')
```

### 2. **处理不平衡数据**

```python
from sklearn.utils import resample

# 或使用class_weight参数
model = RandomForestClassifier(class_weight='balanced')
```

### 3. **查看特征重要性**

```python
# 树模型
importances = model.feature_importances_
for i, importance in enumerate(importances):
    print(f"特征 {i}: {importance}")
```

## 八、学习建议

1. **从简单算法开始**：先掌握线性回归、逻辑回归
2. **理解API设计**：fit、predict、transform模式
3. **注重数据预处理**：归一化、缺失值处理
4. **交叉验证**：避免过拟合
5. **实践项目**：Kaggle竞赛、真实数据集

需要我详细讲解某个特定部分吗？
