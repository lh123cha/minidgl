# miniDGL
该项目基于CMU 10-414课程的课程大作业框架，实现一个简单的图神经网络系统，以及一些ai框架的优化。

预期添加实现以下功能：
- 高效的通用矩阵乘法GEMM
- 高效的softmax、layernorm、elementwise实现
- flash-attention加速优化
- 图数据加载(拓扑数据与特征数据)
- 图采样算法(基于CPU、GPU)
- 简单的图神经网络架构算法
# 一、通用GEMM
具体实现参考
https://voltaic-turret-94c.notion.site/Minidgl-a60dabb385a344d69d018e7708fd5b45?pvs=4

# 一、图数据加载


## 1.1、特征存储结构Frame
节点特征和边特征以Tensor存储，Column类存储特征。

## 1.2、图拓扑数据存储结构MinidglGraph














## 使用接口
接口设计同dgl一致：
### 创建图结构


