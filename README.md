[百度架构师手把手教深度学习](https://aistudio.baidu.com/aistudio/education/group/info/888)
波士顿房间预测in7
- [x] 使用python和Numpy构建神经网络模型

https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/beginners_guide/basics/fit_a_line/README.cn.html

## 宝石识别_pandas_visualdl_losshistory_v3 2022.11.20
数据保存在百度云盘\学习\dp\自己训练结果\宝石识别用冻结和解冻
MyDNN-0.7272是单纯训练100代，没有解冻和冻结训练的参数，eval精度为1，预测4对4

selfmodel-20221120_123002.pdparams是冻结训练100代结果，eval为0.51，预测4对2

selfmodel-20221120_123002-melt.pdparam是冻结之后今后的解冻训练100代结果。eval为0.80，预测4对1

log-宝石识别冻结100代解冻100代数据的结果，解压后用visualdl --logdir log查看

![](imgs\eval.png)

![](imgs\train.jpg)