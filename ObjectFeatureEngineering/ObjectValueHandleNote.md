
- processName 
- eventName 
- stackAddress
- domain
- type
- pathname
- flags
- statbuf
- mode
- cap
- addr
- addrlen
- dirp
- stack
- parent_tid
- child_tid
- option
- target.1
- argv


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