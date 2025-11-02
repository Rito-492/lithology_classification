# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. 参数设置 ====================
CSV_FILE_PATH = 'data/train_all.csv'  # ✅ 修改为你的 CSV 文件路径
MODEL_SAVE_PATH = 'models/xgboost_rock_model.pkl'  # 模型保存路径
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ==================== 2. 加载数据 ====================
print("🔍 正在加载数据...")
try:
    data = pd.read_csv(CSV_FILE_PATH)
except FileNotFoundError:
    raise FileNotFoundError(f"找不到文件：{CSV_FILE_PATH}，请检查路径！")

# 检查必要列
required_columns = ['DEPTH', 'SP', 'GR', 'AC', 'label']
missing_cols = [col for col in required_columns if col not in data.columns]
if missing_cols:
    raise ValueError(f"缺少必要列：{missing_cols}")

# 提取特征和标签
X = data[['DEPTH', 'SP', 'GR', 'AC']].copy()
y = data['label'].copy()

print(f"原始数据形状: {X.shape}")
print(f"标签分布:\n{y.value_counts().sort_index()}")

# ==================== 3. 数据预处理 ====================
print("🧹 数据预处理...")

# 去除 label 为 NaN 的行
valid_mask = y.notna()
X = X[valid_mask]
y = y[valid_mask].astype(int)

# 去除特征中的异常值（可选：使用 3σ 原则）
def remove_outliers(X_df, y_series, threshold=3):
    z_scores = np.abs((X_df - X_df.mean()) / X_df.std())
    no_outlier = (z_scores < threshold).all(axis=1)
    return X_df[no_outlier], y_series[no_outlier]

X, y = remove_outliers(X, y)
print(f"去除异常值后数据形状: {X.shape}")

# （可选）标准化：XGBoost 不需要，但这里提供开关
USE_SCALER = False
if USE_SCALER:
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print("已对特征进行标准化。")
else:
    X = X.values  # 转为 numpy 数组

y = y.values

# ==================== 4. 划分训练集和测试集 ====================
print("✂️  划分训练集和测试集...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y  # 分层抽样，保持类别比例
)

print(f"训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")
print(f"类别数量: {np.unique(y)}")

# ==================== 5. 训练 XGBoost 模型（带调参） ====================
print("🚀 训练 XGBoost 模型...")

# 基础模型
base_model = xgb.XGBClassifier(
    objective='multi:softprob',
    eval_metric='mlogloss',
    num_class=len(np.unique(y)),
    use_label_encoder=False,
    random_state=RANDOM_STATE
)

# 参数网格（可缩小搜索范围加快速度）
param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [4, 5, 6],
    'learning_rate': [0.05, 0.1, 0.15],
    'reg_lambda': [0.5, 1.0, 1.5],
    'gamma': [0.0, 0.1]
}

# 网格搜索（使用 Macro-F1 作为优化目标）
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring='f1_macro',
    cv=3,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# 最优模型
best_model = grid_search.best_estimator_
print(f"✅ 最优参数: {grid_search.best_params_}")

# ==================== 6. 预测与评估 ====================
print("📊 正在评估模型...")

y_pred = best_model.predict(X_test)

# 三种核心评估指标
accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average='macro')
f1_per_class = f1_score(y_test, y_pred, average=None)

print("\n✅ 三项核心评估结果：")
print(f"1. 准确率 (Accuracy):       {accuracy:.4f}")
print(f"2. Macro-F1 Score:          {macro_f1:.4f}")
print("3. 各类别 F1 分数:")
classes = np.unique(y)
for i, cls in enumerate(classes):
    print(f"   类别 {cls}: {f1_per_class[i]:.4f}")

# 详细报告
print("\n📋 详细分类报告（含 Precision, Recall）:")
target_names = [f'Rock_{c}' for c in classes]
print(classification_report(y_test, y_pred, target_names=target_names))

# ==================== 7. 特征重要性可视化 ====================
print("📈 绘制特征重要性...")

importance = best_model.feature_importances_
features = ['DEPTH', 'SP', 'GR', 'AC']

plt.figure(figsize=(8, 5))
indices = np.argsort(importance)[::-1]
plt.bar(range(len(importance)), importance[indices], 
        color=['skyblue', 'lightgreen', 'salmon', 'wheat'][indices[0]],
        edgecolor='black')
plt.xticks(range(len(importance)), [features[i] for i in indices])
plt.title('Feature Importance in XGBoost Model', fontsize=14)
plt.ylabel('Importance (Based on Gain)', fontsize=12)
plt.xlabel('Features', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# ==================== 8. 保存训练好的模型 ====================
print(f"💾 保存模型到 {MODEL_SAVE_PATH}...")

# 保存模型（含预处理器，如果用了）
model_package = {
    'model': best_model,
    'scaler': scaler if USE_SCALER else None,
    'features': ['DEPTH', 'SP', 'GR', 'AC'],
    'classes': np.unique(y),
    'preprocessing': 'StandardScaler' if USE_SCALER else 'None'
}
joblib.dump(model_package, MODEL_SAVE_PATH)
print("✅ 模型保存成功！")

# ==================== 9. 示例：如何用训练好的模型预测新数据 ====================
print("\n🧪 示例：使用保存的模型预测新数据...")

# 加载模型
loaded = joblib.load(MODEL_SAVE_PATH)
model = loaded['model']
scaler = loaded['scaler']

# 模拟一条新测井数据（DEPTH, SP, GR, AC）
new_data = np.array([[1250.0, 50.0, 75.0, 280.0]])  # 形状: (1, 4)

# 如果训练时用了标准化，预测时也要用
if scaler is not None:
    new_data = scaler.transform(new_data)

# 预测
pred_label = model.predict(new_data)[0]
pred_proba = model.predict_proba(new_data)[0]
print(f"新样本特征: DEPTH=1250, SP=50, GR=75, AC=280")
print(f"预测岩性类别: {pred_label}")
print(f"各类别概率: {dict(zip(loaded['classes'], pred_proba))}")