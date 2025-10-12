import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os


def preprocess_for_validation(input_filepath: str):
    """
    加载并预处理用于验证的数据。
    这个函数必须与训练时的预处理函数保持一致。
    """
    if not os.path.exists(input_filepath):
        print(f"错误: 验证文件 '{input_filepath}' 未找到。请确保文件名正确并且文件存在。")
        return None, None

    print("开始预处理验证数据...")
    df = pd.read_csv(input_filepath)

    # 检查所有必需的列是否存在
    required_cols = ['eventId', 'argsNum', 'processId', 'parentProcessId', 'userId', 'mountNamespace', 'returnValue',
                     'target']
    if not all(col in df.columns for col in required_cols):
        print(f"错误: 验证文件缺少必要的列。需要: {required_cols}")
        return None, None

    # 分离特征和真实标签
    y_true = df['target']
    features_df = df.drop('target', axis=1)

    processed_df = pd.DataFrame()

    # 应用与训练时完全相同的特征转换
    processed_df['eventId'] = features_df['eventId']
    processed_df['argsNum'] = features_df['argsNum']
    processed_df['processId_is_os'] = features_df['processId'].isin([0, 1, 2]).astype(int)
    processed_df['parentProcessId_is_os'] = features_df['parentProcessId'].isin([0, 1, 2]).astype(int)
    processed_df['userId_is_os'] = (features_df['userId'] < 1000).astype(int)
    processed_df['mountNamespace_is_default'] = (features_df['mountNamespace'] == 4026531840).astype(int)
    processed_df['returnValue_mapped'] = features_df['returnValue'].apply(
        lambda x: -1 if x < 0 else (0 if x == 0 else 1))

    print("验证数据预处理完成。")
    return processed_df, y_true


def validate_model(model_path: str, validation_data_path: str):
    """
    加载模型并在验证集上进行评估。
    """
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 '{model_path}' 未找到。")
        return

    # 1. 加载模型
    print(f"正在加载模型: {model_path}")
    model = joblib.load(model_path)

    # 2. 加载并预处理验证数据
    X_valid, y_valid = preprocess_for_validation(validation_data_path)

    if X_valid is None:
        print("由于数据预处理失败，验证中止。")
        return

    # 3. 进行预测
    print("\n模型正在对验证数据进行预测...")
    predictions = model.predict(X_valid)

    # 转换预测结果：孤立森林将异常标记为-1，正常为1。
    # 我们将其映射为与我们的标签一致：异常为1，正常为0。
    y_pred = np.array([1 if p == -1 else 0 for p in predictions])

    # 4. 评估性能
    print("\n" + "=" * 40)
    print("       模型在验证集上的性能评估")
    print("=" * 40)

    accuracy = accuracy_score(y_valid, y_pred)
    print(f"整体准确率 (Accuracy): {accuracy:.4f}")

    print("\n分类报告 (Classification Report):")
    print(classification_report(y_valid, y_pred, target_names=['正常 (class 0)', '异常 (class 1)']))

    print("混淆矩阵 (Confusion Matrix):")
    cm = confusion_matrix(y_valid, y_pred)
    print(cm)
    print("\n混淆矩阵解读:")
    print(f"[[ True Negatives (TN), False Positives (FP) ]]")
    print(f" [ False Negatives (FN),  True Positives (TP) ]]")
    print(f"正确预测为正常的数量 (TN): {cm[0][0]}")
    print(f"错误预测为异常的数量 (FP): {cm[0][1]}  (误报)")
    print(f"错误预测为正常的数量 (FN): {cm[1][0]}  (漏报)")
    print(f"正确预测为异常的数量 (TP): {cm[1][1]}")


# --- 主程序入口 ---
if __name__ == '__main__':
    # --- 请在这里配置您的文件名 ---
    MODEL_FILE = 'isolation_forest_model.joblib'
    VALIDATION_CSV = '/Users/hardycheng/Desktop/NUS_Sem1_course/1_Big-data/Projects/Full_Datasets/Track3/processes_train.csv'

    validate_model(MODEL_FILE, VALIDATION_CSV)