# ==========================
# train_random_forest.py
# ==========================
# 功能: 训练随机森林模型用于二分类（检测可疑流量）
# 使用说明:
#   python train_random_forest.py
# 依赖:
#   pip install pandas scikit-learn joblib matplotlib

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from config import processedTrainDatasetCSV, validDatasetCSV

# ========== 1. 读取数据 ==========
data_path = Path(processedTrainDatasetCSV)
df = pd.read_csv(data_path)

# 检查是否存在 target 列
if "target" not in df.columns:
    raise ValueError("❌ 数据中未找到 target 列，请确认列名。")

# ========== 2. 拆分特征和标签 ==========
X = df.drop(columns=["target"])
y = df["target"]

# ========== 3. 训练集 / 测试集划分 ==========
""" X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
) """

# ========== 4. 建立随机森林模型 ==========
rf = RandomForestClassifier(
    n_estimators=200,       # 树的数量
    max_depth=None,         # 自动确定深度
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1               # 全核并行加速
)

print("🚀 开始训练随机森林模型...")
rf.fit(X_train, y_train)
print("✅ 模型训练完成。")

# ========== 5. 模型评估 ==========
y_pred = rf.predict(X_test)

print("\n🎯 模型评估结果：")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ========== 6. 特征重要性可视化 ==========
importances = pd.Series(rf.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(20)

plt.figure(figsize=(8, 6))
top_features.plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("Top 20 Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig("feature_importance_top20.png", dpi=300)
plt.close()
print("📊 特征重要性图已保存为 feature_importance_top20.png")

# ========== 7. 保存模型 ==========
model_path = Path("random_forest_model.joblib")
joblib.dump(rf, model_path)
print(f"💾 模型已保存为 {model_path.resolve()}")

# ========== 8. 示例加载 ==========
# 重新加载模型示例:
# rf_loaded = joblib.load("random_forest_model.joblib")
# print(rf_loaded.predict(X_test[:5]))
