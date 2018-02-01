# 概述

生成对抗网络甚是迷人，以至于深度学习三驾马车之一的Yann LeCun这样说：

>「对抗训练是继切片面包之后最酷的事情。」- Yann LeCun

GAN (Generative Adversarial Nets 生成对抗网络) 的前世今生：

2014年Goodfellow Ian在论文Generative Adversarial Nets中提出来的。是非监督式学习的一种方法，通过让两个神经网络相互博弈的方式进行学习。

本文主要介绍GAN的基本知识，以及在DLS上运行的注意事项。
本模块继续通过经典的MNIST数据集来讲解GAN，使用GAN生成一组手写字符串。并了解如何在DLS上运行，包括：

- GAN的前世今生
- GAN的基本原理
- 在DLS运行的注意事项


- TensorFlow多机多卡实现思路
- 在DLS运行的注意事项

本文档中涉及的演示代码和数据集来源于网络，你可以在这里下载到：[GAN_MNIST.zip](https://s3.meituan.net/v1/mss_0e5a0056f3b64f79aa4749ffa68ce372/cms001/GAN_MNIST.zip)

# GAN的前世今生

GAN，是Generative Adversarial Nets的缩写，中文叫生成对抗网络。

2014年Goodfellow Ian在论文Generative Adversarial Nets中提出来的。

生成对抗网络甚是迷人，以至于深度学习三驾马车之一的Yann LeCun这样说：

>「对抗训练是继切片面包之后最酷的事情。」- Yann LeCun

生成对抗网络（GAN）是一类在无监督学习中使用的神经网络，，通过让两个神经网络相互博弈的方式进行学习。其有助于解决按文本生成图像、提高图片分辨率、药物匹配、检索特定模式的图片等任务。

![alt text](https://s3.meituan.net/v1/mss_0e5a0056f3b64f79aa4749ffa68ce372/cms001/arithmetic-1513567979175.png =466x244 "GAN 实例")


# GAN的基本原理
###### 简单理解GAN

GAN包含两个模型，一个是生成模型G，一个是判别模型D。

生成模型G从一些假数据或者随机数据中生成新的数据，用来欺骗判别模型D。

判别模型D则不断的从一堆柔和了真实数据和G生成的假数据当中，识别区分出真假数据。

就像矛与盾一样，生成模型G和判别模型D不断的互相切磋，一决高下。但是在切磋过程中，G和D都不断的学习，不断的提高自己的生成和判别水平。


![alt text](https://s3.meituan.net/v1/mss_0e5a0056f3b64f79aa4749ffa68ce372/cms001/generative-adversarial-network-1513567980319.png =466x244 "GAN 示例图")

###### GAN的代码实现
从GAN的基本原理当中不难看出，实现GAN需要完成以下部分：

* 生成网络G的实现
* 判别网络D的实现
* 生成网络和判别网络的组合D_on_G
* 判别网络D的训练
* 网络组合D_on_G的训练

注意：在训练D_on_G的时候需要固定住D的训练参数，通过keras实现的model，很容易用model.trainalbe=False来实现。




# 在DLS运行的注意事项
由于DLs的文件读取写入都是直接对HDFS进行的，因此对于实例代码中用到的数据，有些API的读取写入是不支持HDFS的。因此，我们需要额外做一些工作，以便能让模型在DLS上运行。

######数据文件的读取

* 使用分布式缓存
* 使用tf.gfile进行预先读取

下面的示例代码就是将HDFS的HDFS_FILE_PATH文件通过tf.gfile读取到本地。


```
 with tf.gfile.Open(HDFS_FILE_PATH, 'rb') as in_file:
     with open(LOCAL_FILE_PATH, 'wb') as out_file:
         out_file.write(in_file.read())
 ```
 

######数据文件的写入

* 使用tf.gfile进行同步

下面的示例代码就是将本地的的LOCAL_FILE_PATH文件通过tf.gfile写入到hdfs上。


```
with open(LOCAL_FILE_PATH, 'rb') as in_file:
    with tf.gfile.Open(HDFS_FILE_PATH, 'wb') as out_file:
        out_file.write(in_file.read())
 
```



**当然示例代码中还实现了其他的一些功能，这里就不做详细的描述了。可以直接阅读代码，如果发现代码缺陷或者又不明白之处欢迎交流。**

**祝您TensorFlow之旅愉快，祝好！**
