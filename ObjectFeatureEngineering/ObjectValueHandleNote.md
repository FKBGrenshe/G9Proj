# 缺失值处理
将缺失值>95%的列进行删除
```python
# 假设 df 是你的 DataFrame
# 删除缺失比例超过 95% 的列
print("筛选前剩余的列数:", train.shape[1])
# 查看都有哪些列
print("删除前都有哪些列:", train.columns)
threshold = 0.95
train = train.loc[:, train.isnull().mean() <= threshold]
print("筛选后剩余的列数:", train.shape[1])
print("删除后都有哪些列:", train.columns)
----------------------------------------------
筛选前剩余的列数: 53
删除前都有哪些列: Index(['index', 'target', 'timestamp', 'processId', 'threadId',
       'parentProcessId', 'userId', 'mountNamespace', 'processName',
       'hostName', 'eventId', 'eventName', 'stackAddresses', 'argsNum',
       'returnValue', 'domain', 'type', 'protocol', 'pathname', 'flags', 'dev',
       'inode', 'fd', 'statbuf', 'dirfd', 'mode', 'ruid', 'euid', 'rgid',
       'egid', 'cap', 'sockfd', 'addr', 'addrlen', 'dirp', 'count', 'stack',
       'parent_tid', 'child_tid', 'tls', 'option', 'arg2', 'arg3', 'arg4',
       'arg5', 'pid', 'sig', 'target.1', 'oldfd', 'newfd', 'uid', 'argv',
       'gid'],
      dtype='object')
筛选后剩余的列数: 23
删除后都有哪些列: Index(['index', 'target', 'timestamp', 'processId', 'threadId',
       'parentProcessId', 'userId', 'mountNamespace', 'processName',
       'hostName', 'eventId', 'eventName', 'stackAddresses', 'argsNum',
       'returnValue', 'pathname', 'flags', 'dev', 'inode', 'fd', 'statbuf',
       'dirfd', 'mode'],
      dtype='object')
```

# 文本数值预处理
针对不同文本特征的特性，使用相应的处理方法转为数值特征

- processName 
- eventName 
- stackAddress ： 
    - ==当前方法==：计算栈深度代替栈地址，（其他方法，栈最深地址，最浅地址，平均地址深度...）
    - 原因：
        1. 栈地址序列本质上反映了调用链的深度，深度越大往往代表系统行为更复杂或调用层次更深。→ 栈深度能保留大部分有用信息，而不必处理变长的复杂序列。兼容所有模型：直接转成一个数值特征（int）后，几乎所有模型（树模型、逻辑回归、神经网络）都能直接使用。
        2. 简化缺失处理：空列表 [] 的栈深度自然就是 0，直观且无需额外编码。

- domain （被删除）
- type（被删除）
- pathname
- flags:
    - 一个系统调用的多个“标志位”（flags），每条记录可能有 1~N 个 flag，以 | 分隔 
    - **这类多标签（multi-label categorical）特征需要拆解成多列二进制特征才能用**
- statbuf -- 被丢弃
- mode：==**异常值处理：存在部分数字->是因为在数据收集中，多个特征的合并，解码错误导致，需要进行还原**==
- cap（被删除）
- addr（被删除）
- addrlen（被删除）
- dirp（被删除）
- stack（被删除）
- parent_tid（被删除）
- child_tid（被删除）
- option（被删除）
- target.1（被删除）
- argv（被删除）


## 类别型categories
> 处理方式 label/one-hot编码

- processName：执行进程名
- eventName：系统调用名


## 序列型（栈地址列表）
> 处理方式 序列长度统计+hash embedding
- stackAddresses

## 枚举型 categories
> One-hot编码
- domain：socket 域类型（AF_INET 等）
- type：socket类型
- option：prctl选项

## 字符型路径
> 提取模式/层级/后缀特征
- pathname：文件路径

## 枚举+组合型
> 多标签二进制展开
- flags：open/clone标志位

## 指针或地址
> 通常丢弃或转为是否为空binary
- statbuf/ addrlen/dirp/stack/parent_tid/child_tid：低层地址指针，

## 枚举+数值混合
> 提取是否有
- mode：权限位或标志

## 权限名
> One-hot
cap：系统capability

## JSON-like结构
> 解析成字段（IP版本、端口、family等）
- addr：网络地址对象

## 列表型参数
> TF-IDF/Token embedding
- argv：执行参数

## 字符路径
> 类似pathname，提取文件层级或关键字特征
- target.1：输出目录