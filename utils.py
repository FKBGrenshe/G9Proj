# utils.py
from pathlib import Path

# 如果 utils.py 在项目根目录：用 .parent
# 若在子目录（例如 utils/utils.py），就用 .resolve().parent.parent
PROJECT_ROOT = Path(__file__).resolve().parent

def project_path(*parts) -> Path:
    """
    基于项目根目录拼接路径。自动处理不同系统的分隔符。
    用法：project_path('data', 'sample_processes_train.csv')
    """
    # 把每一段都去掉前导斜杠，避免把 join 前缀截断
    clean = [str(p).lstrip("/\\") for p in parts]
    return PROJECT_ROOT.joinpath(*clean)

def get_full_path(relative_path: str) -> Path:
    """
    兼容你现在的接口：传入相对路径字符串，返回绝对路径 Path。
    注意：如果传入 '/data/xxx.csv'，会自动去掉前导斜杠。
    """
    rel = relative_path.lstrip("/\\")
    return PROJECT_ROOT / rel

if __name__ == "__main__":
    # 自检
    p1 = project_path('data', 'sample_processes_train.csv')
    p2 = get_full_path('/data/sample_processes_train.csv')   # 故意带/也OK
    p3 = get_full_path('/savedModule/')   # 故意带/也OK
    # print("PROJECT_ROOT =", PROJECT_ROOT)
    # print("project_path =", p1)
    # print("get_full_path=", p2)
    # print("exists? p1:", p1.exists(), "| p2:", p2.exists())
    print("get_full_path=", p3)