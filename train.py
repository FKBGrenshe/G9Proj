import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import os

def preprocess_for_training(input_filepath: str):
    """
    加载并预处理用于训练的数据。
    在无监督训练中，我们不需要 'target' 列。
    """
    if not os.path.exists(input_filepath):
        print(f"错误: 输入文件 '{input_filepath}' 未找到。请确保它和脚本在同一个目录下。")
        return None

    print("开始预处理数据...")
    df = pd.read_csv(input_filepath)

    # 在进行无监督学习训练时，我们不需要目标标签
    if 'target' in df.columns:
        df = df.drop('target', axis=1)

    # 检查所有必需的特征列是否存在
    required_features = ['eventId', 'argsNum', 'processId', 'parentProcessId', 'userId', 'mountNamespace',
                         'returnValue']
    if not all(col in df.columns for col in required_features):
        print(f"错误: CSV文件缺少必要的特征列。需要: {required_features}")
        return None

    processed_df = pd.DataFrame()

    # 根据论文进行特征转换
    processed_df['eventId'] = df['eventId']
    processed_df['argsNum'] = df['argsNum']
    processed_df['processId_is_os'] = df['processId'].isin([0, 1, 2]).astype(int)
    processed_df['parentProcessId_is_os'] = df['parentProcessId'].isin([0, 1, 2]).astype(int)
    processed_df['userId_is_os'] = (df['userId'] < 1000).astype(int)
    processed_df['mountNamespace_is_default'] = (df['mountNamespace'] == 4026531840).astype(int)
    processed_df['returnValue_mapped'] = df['returnValue'].apply(lambda x: -1 if x < 0 else (0 if x == 0 else 1))

    print("数据预处理完成。")
    return processed_df

def train_and_save_model(data: pd.DataFrame, model_save_path: str):
    """
    使用所有提供的数据训练孤立森林模型并保存。
    """
    if data is None:
        print("没有可用于训练的数据。")
        return

    print("\n开始在全部数据上训练孤立森林模型...")

    # 初始化模型
    # contamination='auto' 是一个稳健的默认值
    # n_jobs=-1 使用所有可用的CPU核心以加快训练速度
    model = IsolationForest(contamination='auto', random_state=42, n_jobs=-1)

    # 使用所有数据进行训练
    model.fit(data)

    print("模型训练完成。")

    # 使用 joblib 保存训练好的模型
    try:
        joblib.dump(model, model_save_path)
        print(f"模型已成功保存到: '{model_save_path}'")
    except Exception as e:
        print(f"保存模型时出错: {e}")


# --- 主程序入口 ---
if __name__ == '__main__':
    # 定义输入和输出文件名
    input_csv = '/Users/hardycheng/Desktop/NUS_Sem1_course/1_Big-data/Projects/Full_Datasets/Track3/processes_train.csv'
    model_output_file = 'isolation_forest_model.joblib'

    # 步骤 1: 预处理数据
    processed_training_data = preprocess_for_training(input_csv)

    # 步骤 2: 如果预处理成功，则训练并保存模型
    if processed_training_data is not None:
        train_and_save_model(processed_training_data, model_output_file)
    else:
        print("\n由于预处理失败，模型训练被中止。")