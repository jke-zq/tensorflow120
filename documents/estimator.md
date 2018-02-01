# 概述
本模块将会以经典的例子：MNIST手写数字识别 来带你快速入门tf.estimator，并了解如何在DLS上运行，包括：
- TensorFlow1.4新特性的介绍
- DLS的路径配置

本文档中涉及的演示代码和数据集来源于网络，你可以在这里下载到：[ESTIMATOR_MNIST.zip](https://s3.meituan.net/v1/mss_0e5a0056f3b64f79aa4749ffa68ce372/cms001/MNIST.zip)

# TensorFlow1.4新特性介绍
Google在2017.11.21宣布TensorFlow1.4正式公开发布，并宣称这是一次大更新，拥有很多令人兴奋的功能。
主要介绍和本示例有关的两个新特性：
## Estimator（估算器）
Estimators ，这是一个比较轻量化，并且在Google内部生产环境中广泛使用的 API ，其中 Estimators 提供了很多模型供大家使用，叫做 Canned Estimator ，他们的关系是这样的：Estimators 和 tf.keras 上层封装了一个 Canned Estimator ，可以用其来封装成各种模型。

<img src="https://s3.meituan.net/v1/mss_0e5a0056f3b64f79aa4749ffa68ce372/cms001/tf_api_arch-1513336367808.png">

## DatasetAPI（数据集）
Dataset API 已从 tf.contrib.data 迁移到核心软件包 tf.data 中。1.4 版的 Dataset API 还增加了对 Python 生成器的支持。Google强烈建议使用 Dataset API 为 TensorFlow 模型创建输入管道，因为：

*  与旧 API（feed_dict 或队列式管道）相比，Dataset API 可以提供更多功能。
* Dataset API 的性能更高。
* Dataset API 更简洁，更易于使用。


# DLS的路径配置
要在DLS训练模型，需要将数据文件路径和结果输出（模型保存和图表结果）文件路径重新指定为DLS上的存储的文件路径，目前以hdfs://开头。

因此强烈建议在代码中通过命令行参数的形式传入所需文件路径，默认值为本地测试所需的文件路径。这样在使用DLS时，只需要在**启动参数**一栏显示传入对应文件路径即可。

<img src="https://s3.meituan.net/v1/mss_0e5a0056f3b64f79aa4749ffa68ce372/cms001/estimator_config-1513336312298.png">

# 在DLS上进行训练

* 创建项目
* 创建任务
	*  计算资源配置：单机 CPU或者GPU
	*  启动参数：如上图所示
	*  TensorBoard：文件路径需要和启动参数的model_dir一致
* 任务日志查看：如下图-日志查看
* TensorBoard查看

日志查看：
<img src="https://s3.meituan.net/v1/mss_0e5a0056f3b64f79aa4749ffa68ce372/cms001/estimator_tb_log-1513336314584.png">

TensorBoard查看：
<img src="https://s3.meituan.net/v1/mss_0e5a0056f3b64f79aa4749ffa68ce372/cms001/estimator_tensorboard-1513336314785.png">
