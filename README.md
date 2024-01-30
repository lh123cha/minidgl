# miniDGL
该项目基于CMU 10-414课程的课程大作业框架，实现一个简单的图神经网络系统，以及一些ai框架的优化。

预期添加实现以下功能：
1.高效的通用矩阵乘法GEMM
1.Checkpoint实现
2.高效的softmax实现OneFlow
3.flash-attention加速优化
4.图数据加载(拓扑数据与特征数据)
5.图采样算法(基于CPU、GPU)
6.简单的图神经网络架构算法
# 一、CheckPoint实现

# 一、图数据加载


## 1.1、特征存储结构Frame
节点特征和边特征以Tensor存储，Column类存储特征。

## 1.2、图拓扑数据存储结构MinidglGraph














## 使用接口
接口设计同dgl一致：
### 创建图结构


