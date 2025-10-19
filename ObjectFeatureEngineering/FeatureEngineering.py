# 针对文本特征的数据预处理
## 数据导入
import pandas as pd
import numpy as np
import ast
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def decode_or_split_flags(value):
        """
        同时支持字符串型和整型 flags:
        - 对字符串（'O_RDONLY|O_CLOEXEC'）自动拆分
        - 对整数（如 -1868620656）按 Linux open flags 解析
        - 对 NaN / 空值返回 []
        """

        # ========== 1) 空值直接返回 ==========
        if pd.isna(value):
            return []

        # ========== 2) 如果是字符串 ==========
        if isinstance(value, str):
            # 拆分多标志
            if "|" in value:
                return [v.strip() for v in value.split("|") if v.strip()]
            # 单个标志（如 "O_RDONLY"）
            elif value.strip():
                return [value.strip()]
            else:
                return []

        # ========== 3) 如果是数值型（含负数） ==========
        try:
            value = int(value) & 0xFFFFFFFF  # 转无符号 32 位
        except Exception:
            return []

        FLAG_MAP = {
            0x00000000: "O_RDONLY",
            0x00000001: "O_WRONLY",
            0x00000002: "O_RDWR",
            0x00000040: "O_CREAT",
            0x00000080: "O_EXCL",
            0x00000100: "O_NOCTTY",
            0x00000200: "O_TRUNC",
            0x00000400: "O_APPEND",
            0x00000800: "O_NONBLOCK",
            0x00001000: "O_DSYNC",
            0x00002000: "FASYNC",
            0x00004000: "O_DIRECT",
            0x00008000: "O_LARGEFILE",
            0x00010000: "O_DIRECTORY",
            0x00020000: "O_NOFOLLOW",
            0x00040000: "O_NOATIME",
            0x00080000: "O_CLOEXEC",
            0x00100000: "O_PATH",
            0x00200000: "O_TMPFILE"
        }

        result = [name for mask, name in FLAG_MAP.items() if value & mask]

        # 没有匹配项默认 O_RDONLY
        if not result:
            result.append("O_RDONLY")

        return result

def extract_stack_depth(x):
        """
        将 stackAddresses 解析成列表，并返回其深度（长度）。
        - 空值或格式错误的返回 0
        """
        if isinstance(x, str) and x.startswith('['):
            try:
                vals = ast.literal_eval(x)
                if isinstance(vals, list):
                    return len(vals)
            except:
                pass
        return 0

    # 转换成 {name: value} 的字典
def list_to_dict(lst):
    return {d['name']: d['value'] for d in lst}

def preprocess_text_features(input_filepath: str, output_filepath: str):
    """
    预处理文本特征，如 processName 和 eventName，使用标签编码进行转换。

    Args:
        input_filepath (str): 包含原始数据的CSV文件路径。
        output_filepath (str): 保存处理后数据的CSV文件路径。
    """


    

    train = pd.read_csv(input_filepath)

    # 解析args列
    train['args_parsed'] = train['args'].apply(ast.literal_eval)
    train['args_dict'] = train['args_parsed'].apply(list_to_dict)
    # 展开成 DataFrame
    args_df = pd.json_normalize(train['args_dict'])
    train = pd.concat([train.drop(columns=['args', 'args_parsed', 'args_dict']), args_df], axis=1)
    # 导出解析arg的CSV
    saved_parsed_csv = False;
    if saved_parsed_csv:
        train.to_csv(output_filepath, index=False)
        print("✅ 已经生成新的 CSV 文件: sample_processes_train_parsed.csv")

    #############################################################################
    ### 异常值处理 - 缺失值>95%删除
    print("异常值处理 - 缺失值>95%删除...")
    #############################################################################
    #############################################################################
    # 假设 df 是你的 DataFrame
    # 删除缺失比例超过 95% 的列
    print("筛选前剩余的列数:", train.shape[1])
    # 查看都有哪些列
    print("删除前都有哪些列:", train.columns)
    threshold = 0.95
    train = train.loc[:, train.isnull().mean() <= threshold]
    print("筛选后剩余的列数:", train.shape[1])
    print("删除后都有哪些列:", train.columns)
    #############################################################################
    # 查看当前train中的文本型特征
    cat_cols = train.select_dtypes(include=['object']).columns
    print("当前文本特征包括：", cat_cols)
    #############################################################################
    ### 文本特征处理 
    #############################################################################


    ########################################################
    ### processName、eventName
    print("文本特征处理 - processName、eventName...")
    ########################################################
    # 使用 onehotencoder 对 processName和 eventName 进行编码
    from sklearn.preprocessing import LabelEncoder
    for col in ["processName", "eventName"]:
        train[col] = train[col].astype(str)
        le = LabelEncoder()
        train[col + "_enc"] = le.fit_transform(train[col])


    ########################################################
    ### addrlen, dirp, stack, parent_tid, child_tid, statbuf等直接删除
    print("删除文本特征 - addrlen, dirp, stack, parent_tid, child_tid, statbuf...")
    ########################################################
    train = train.drop(columns=["statbuf"])
    print('查看删除statbuf后剩余的列数:', train.shape[1])


    ########################################################
    ### stackAddress列表，转成栈深度
    print("文本特征处理 - stackAddress列表，转成栈深度...")
    ########################################################
    # stackAddress 列是一个栈地址列表，使用栈深度（列表长度）作为数值特征
    train["stack_depth"] = train["stackAddresses"].apply(extract_stack_depth)


    ########################################################
    ### 进行 flags 列的解码
    print("文本特征处理 - 进行 flags 列的解码...")
    ########################################################
    train["decoded_flags"] = train["flags"].apply(decode_or_split_flags)
    # 查看前几行数据
    print(train[["flags", "decoded_flags"]].head())
    
    mlb = MultiLabelBinarizer()
    flags_encoded = mlb.fit_transform(train["decoded_flags"])
    flags_df = pd.DataFrame(flags_encoded, columns=[f"flag_{c}" for c in mlb.classes_])
    train = pd.concat([train, flags_df], axis=1)
    # 查看当前特征中都有哪些列
    print(train.columns)
    # 查看当前前几行数据
    print(train.head())


    ####################################################
    ### 暂时放弃 pathname 和 mode 数据列的预处理
    ### 保存最终处理后的数据
    print("删除文本特征 - pathname 和 mode...")
    ####################################################
    ### 删除pathname和mode列
    train = train.drop(columns=["pathname", "mode"])   
    # 查看还有哪些列是文本型
    cat_cols = train.select_dtypes(include=['object']).columns
    print("当前文本特征包括：", cat_cols)
    # 删除这些列
    train = train.drop(columns=cat_cols)
    # 对数据进行保存
    train.to_csv(output_filepath, index=False)
    print(f"✅ 已经生成最终处理后的 CSV 文件: {output_filepath}")



if __name__ == "__main__":
    # from config import trainDatasetCSV, validDatasetCSV, processedTrainDatasetCSV, processedValidDatasetCSV;
    trainDatasetCSV = r"D:\Project\NusSemester1BigDataProjectBETH\G9Proj\data\sample_processes_train.csv"
    validDatasetCSV = r"D:\Project\NusSemester1BigDataProjectBETH\G9Proj\data\sample_processes_valid.csv"
    processedTrainDatasetCSV = r"D:\Project\NusSemester1BigDataProjectBETH\G9Proj\data\processed_sample_processes_train.csv"
    processedValidDatasetCSV = r"D:\Project\NusSemester1BigDataProjectBETH\G9Proj\data\processed_sample_processes_valid.csv"
    # 预处理训练集
    preprocess_text_features(
        input_filepath=trainDatasetCSV,
        output_filepath=processedTrainDatasetCSV
    )
    # 预处理验证集
    preprocess_text_features(
        input_filepath=validDatasetCSV,
        output_filepath=processedValidDatasetCSV
    )