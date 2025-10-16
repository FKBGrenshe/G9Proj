import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path

def read_data_set(file_path) -> pd.DataFrame:
    """Reads a CSV file and returns a pandas DataFrame."""
    file_path = Path(file_path)
    return pd.read_csv(file_path, index_col=0)

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fills missing values with the median for simplicity."""
    df = df.copy()
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    return df

# ========== 1. 读取你的新数据集 ==========
print("Reading the processed dataset...")
file_path = r'D:\Project\NusSemester1BigDataProjectBETH\G9Proj\data\processed_train_data_all2Number_delProcessname_delMode.csv'
try:
    data = read_data_set(file_path)
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    print("Please ensure the script is in the same directory as the CSV file, or provide the full path.")
    exit()

# ========== 2. 预处理和特征提取 ==========
print("Preprocessing data...")
data = handle_missing_values(data)

# 在无监督场景下，我们不使用 'target' 列
if 'target' in data.columns:
    features_df = data.drop(columns=['target'])
else:
    features_df = data

print(f"Dataset contains {features_df.shape[0]} samples and {features_df.shape[1]} features.")

# ========== 3. 使用PCA进行特征选择 ==========
# 步骤 3.1: 数据标准化
print("\nStandardizing data for PCA...")
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_df)

# 步骤 3.2: 应用PCA
print("Applying Principal Component Analysis (PCA)...")
# n_components可以设置为一个小数，如0.95，代表保留95%方差的成分数
# 或者设置为一个整数，代表主成分的具体数量
pca = PCA(n_components=0.95) 
pca.fit(features_scaled)

print(f"PCA selected {pca.n_components_} components to explain 95% of the variance.")

# 步骤 3.3: 计算特征重要性
# 重要性可以通过特征在所有主成分上的绝对载荷之和来衡量
# pca.components_ 的形状是 (n_components, n_features)
most_important_features = np.abs(pca.components_).sum(axis=0)

feature_importance = pd.DataFrame({
    'feature': features_df.columns,
    'importance_score': most_important_features
}).sort_values('importance_score', ascending=False)


# ========== 4. 提取并显示最重要的特征 ==========
print("\nExtracting feature importances based on PCA...")

# 保存所有特征的重要性到CSV文件
feature_importance.to_csv('feature_importances_unsupervised_ranked.csv', index=False)
print("\nUnsupervised feature importances saved to 'feature_importances_unsupervised_ranked.csv'")

print("\nTop 30 Most Important Features (Unsupervised):")
print(feature_importance.head(30))
