from config import *;
from train import *;
from valid import *;
from data_process import *;
import os


# 这是主程序入口
def main():
    print("请使用 'data_process.py' 进行数据预处理，或使用 'valid.py' 进行模型验证。")
    ## 输入trian 即重新训练一个模型并进行保存，输入valid 即使用已有模型进行验证，保存至saveModule目录中
    mode = input("请输入模式 (train/valid): ").strip().lower()
    if mode == 'train':
        # 重新训练模型

        # 获取当前脚本所在目录
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # 拼接相对路径
        trainDatasetCSV = os.path.join(base_dir, "data", "sample_processes_train.csv")
        print(trainDatasetCSV)

        # 1. 数据预处理
        processed_training_data = preprocess_for_training(trainDatasetCSV)
        if processed_training_data is None:
            print("数据预处理失败，训练中止。")
            return
        
        # 2. 训练并保存模型
        # 请输入模型名称
        model_name = input("请输入要保存的模型名称 (例如 'trained_model'): ").strip()
        moduleSavePath = f"{moduleSavePath}{model_name}.joblib"
        train_and_save_model(processed_training_data, moduleSavePath)

    elif mode == 'valid':
        # 从已经保存模型中进行验证
        # 输入模型名称，如果savedModule中没有该模型则报错
        model_name = input("请输入已训练模型的名称 (例如 'trained_model'): ").strip()
        model_file = f"{moduleSavePath}{model_name}.joblib"
        if not os.path.exists(model_file):
            print(f"模型文件 {model_file} 不存在，请检查模型名称。")
            return
        
        validate_model(model_file, validDatasetCSV)

    else:
        print("无效的模式输入，请输入 'train' 或 'valid'。")


# 运行主程序
if __name__ == '__main__':
    main()