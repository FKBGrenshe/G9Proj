import pandas as pd


def preprocess_beth_data(input_filepath: str, output_filepath: str):
    """
    根据 BETH 论文附录A中的描述预处理网络安全日志数据。

    Args:
        input_filepath (str): 包含原始数据的CSV文件路径。
        output_filepath (str): 保存处理后数据的CSV文件路径。
    """
    try:
        # 1. 加载数据
        df = pd.read_csv(input_filepath)
        print("成功加载数据，原始数据的前5行：")
        print(df.head())
        print("\n原始数据信息：")
        df.info(verbose=False)
    except FileNotFoundError:
        print(f"错误：输入文件未找到，请确认路径 '{input_filepath}' 是否正确。")
        return

    # 创建一个新的DataFrame来存储处理后的特征
    processed_df = pd.DataFrame()

    # 2. 根据论文建议进行特征转换和选择
    # 保留 'eventId' 和 'argsNum'
    processed_df['eventId'] = df['eventId']
    processed_df['argsNum'] = df['argsNum']

    # processId: 如果ID是0, 1, or 2，则为1，否则为0
    processed_df['processId_is_os'] = df['processId'].isin([0, 1, 2]).astype(int)

    # parentProcessId: 规则同processId
    processed_df['parentProcessId_is_os'] = df['parentProcessId'].isin([0, 1, 2]).astype(int)

    # userId: 如果ID < 1000，则为1 (系统活动)，否则为0 (用户活动)
    processed_df['userId_is_os'] = (df['userId'] < 1000).astype(int)

    # mountNamespace: 如果值为 4026531840，则为1，否则为0
    processed_df['mountNamespace_is_default'] = (df['mountNamespace'] == 4026531840).astype(int)

    # returnValue: 映射为-1, 0, 1三种状态
    # <0 映射为 -1 (error)
    # =0 映射为 0 (success)
    # >0 映射为 1 (success with signal)
    processed_df['returnValue_mapped'] = df['returnValue'].apply(lambda x: -1 if x < 0 else (0 if x == 0 else 1))

    # 保留目标变量（如果存在），以便于后续模型评估
    if 'target' in df.columns:
        processed_df['target'] = df['target']

    print("\n数据预处理完成，处理后的数据前5行：")
    print(processed_df.head())
    print("\n处理后的数据信息：")
    processed_df.info()

    # 4. 保存处理后的数据
    try:
        processed_df.to_csv(output_filepath, index=False)
        print(f"\n处理后的数据已成功保存到: '{output_filepath}'")
    except Exception as e:
        print(f"保存文件时出错: {e}")


# --- 主程序入口 ---
if __name__ == '__main__':
    # 定义输入和输出文件名
    # 请确保 'processes_train测试.csv' 文件与此脚本位于同一目录下
    input_file = '/Users/hardycheng/Desktop/processes_train测试.csv'
    output_file = './processed_data.csv'

    # 执行预处理函数
    preprocess_beth_data(input_file, output_file)