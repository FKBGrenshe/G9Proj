import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import re

def read_data_set(file_path) -> pd.DataFrame:
    """Reads a CSV file and returns a pandas DataFrame."""
    file_path = Path(file_path)
    return pd.read_csv(file_path)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data: handles missing values and encodes categorical variables."""
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # Handle missing values
    numeric_columns = df.select_dtypes(include=np.number).columns
    categorical_columns = df.select_dtypes(include=['object']).columns

    if len(numeric_columns) > 0:
        for col in numeric_columns:
            df[col] = df[col].fillna(df[col].median())
    
    if len(categorical_columns) > 0:
        for col in categorical_columns:
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    # Encode categorical features
    le_dict = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
    
    return df

def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    清理特征集:
    1. 移除高基数的ID特征。
    2. 移除因独热编码等产生的重复特征。
    """
    print("  - Cleaning features...")
    
    # 步骤 1: 移除高基数ID特征
    ids_to_drop = ['processId', 'threadId']
    existing_ids_to_drop = [col for col in ids_to_drop if col in df.columns]
    if existing_ids_to_drop:
        df = df.drop(columns=existing_ids_to_drop)
        print(f"    Removed ID features: {existing_ids_to_drop}")

    # 步骤 2: 移除重复的Flag特征
    seen_base_names = set()
    cols_to_drop_redundant = []
    for col in df.columns:
        # 使用正则表达式获取基础名称, 例如 'flag_O_RDWR.1' -> 'flag_O_RDWR'
        base_name = re.sub(r'\.\d+$', '', col)
        if base_name in seen_base_names:
            cols_to_drop_redundant.append(col)
        else:
            seen_base_names.add(base_name)

    if cols_to_drop_redundant:
        df = df.drop(columns=cols_to_drop_redundant)
        print(f"    Removed {len(cols_to_drop_redundant)} redundant feature columns.")
        
    return df

# ========== 1. 读取数据 ==========
print("Reading data...")
# data = read_data_set(r'D:\Project\NusSemester1BigDataProjectBETH\G9Proj\data\sample_processes_train.csv')
# validation_data = read_data_set(r'D:\Project\NusSemester1BigDataProjectBETH\G9Proj\data\sample_processes_valid.csv')

data = read_data_set(r'D:\Project\NusSemester1BigDataProjectBETH\G9Proj\data\processed_train_data_all2Number_delProcessname_delMode.csv')
validation_data = read_data_set(r'D:\Project\NusSemester1BigDataProjectBETH\G9Proj\data\processed_valid_data_all2Number_delProcessname_delMode.csv')

# Display initial information
print("Initial training data shape:", data.shape)
print("\nFirst 5 rows of training data:")
print(data.head())
print("\nMissing values in training data:")
print(data.isnull().sum().loc[lambda x: x > 0]) # Only show columns with missing values

# ========== 2. 预处理数据 ==========
print("\nPreprocessing data...")
trainDataSet = preprocess_data(data)
validationDataSet = preprocess_data(validation_data)

# ========== 3. 特征-目标分离 ==========
# 假设 'target' 是目标列。同时，移除可能导致数据泄露的列，如 'sus' 或 'evil'。
leaky_columns = ['target', 'sus', 'evil'] 
print("\nProcessing Training Data...")
train_data_cleaned = clean_features(trainDataSet)
print("\nProcessing Validation Data...")
validation_data_cleaned = clean_features(validationDataSet)
existing_leaky_columns_train = [col for col in leaky_columns if col in train_data_cleaned.columns]
existing_leaky_columns_valid = [col for col in leaky_columns if col in validation_data_cleaned.columns]

# y_train 在这里仅用于后续计算 contamination (污染因子)，以及最终评估
y_train = trainDataSet['target']
X_train = trainDataSet.drop(columns=existing_leaky_columns_train)

y_validation = validationDataSet['target']
X_validation = validationDataSet.drop(columns=existing_leaky_columns_valid)

# Align columns - crucial if train/validation sets have different columns after preprocessing
train_cols = X_train.columns
valid_cols = X_validation.columns
shared_cols = list(set(train_cols) & set(valid_cols))
X_train = X_train[shared_cols]
X_validation = X_validation[shared_cols]

print(f"\nTraining with {X_train.shape[1]} features.")

# ========== 4. 训练 Isolation Forest 模型 (无监督) ==========
print("\nTraining Isolation Forest model...")
# contamination 指的是数据集中异常点的比例。'auto' 是一个常用选项，
# 也可以根据训练数据中的异常比例进行设置。
contamination_rate = 'auto' # or y_train.value_counts(normalize=True)[1]
if 1 in y_train.value_counts(normalize=True):
    contamination_rate = y_train.value_counts(normalize=True)[1]
    print(f"Using contamination rate from training data: {contamination_rate:.4f}")

iso_forest_model = IsolationForest(
    n_estimators=100,
    # contamination=contamination_rate,
    contamination=0.04,
    random_state=42,
    n_jobs=-1 # 使用所有可用的CPU核心
)

# 无监督学习：fit 的时候不需要 y_train
iso_forest_model.fit(X_train)

# ========== 5. 模型评估 ==========
print("\nMaking predictions on validation data...")
# predict 返回 -1 (异常) 和 1 (正常)
y_pred = iso_forest_model.predict(X_validation)

# 将预测结果映射到 0 和 1 (0: 正常, 1: 异常) 以便评估
y_pred_mapped = np.where(y_pred == -1, 1, 0)

print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_validation, y_pred_mapped))
print("\nConfusion Matrix:")
print(confusion_matrix(y_validation, y_pred_mapped))
print("\nClassification Report:")
print(classification_report(y_validation, y_pred_mapped, zero_division=0))

# 注意：标准的 IsolationForest 模型不提供直接的 'feature_importances_' 属性。
# 因此，我们在此不进行特征重要性分析。
