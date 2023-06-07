1. 训练数据和测试数据的存放路径：.\dataset
   如下图所示：

   ![image-20230606230826691](C:\Users\ChenVadder\Desktop\NLPCC2023-Task6\assets\image-20230606230826691.png)

2. CPU:AMD Ryzen 5 4600U
   GPU:Radeon Graphics 
   未使用CUDA

3. 模型简介：
   本项目基于Elastic Search反向索引和Sentence Transformer模型，提出利用反向索引的检索找出待链接实体的候选项，而后通过语义空间检索的方式取得待链接实体和候选实体的向量表示，并根据相似度进行实体链接的方法。

4. 复现步骤：运行run.sh



其他说明：

1. 需要部署Elastic Search，当前项目使用版本为：8.1.0
2. 本次所使用的sentence_transformers 为直接调用的初始参数值
3. 建立索引和检索的过程存在一定的随机性，多次重复时结果不一定完全相同，如下图所示，现在的时间点生成的数据和竞赛时生成的数据就有近两千个不同（在生成64000条数据的情况下）
   ![](C:\Users\ChenVadder\Desktop\NLPCC2023-Task6\assets\202306062248281.png)
