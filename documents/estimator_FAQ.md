# 概述
本文介绍 Estimator 的常见问题。    
由于本人在工作中大力推广 Estimator + Dataset 的方式来实现分布式训练，同时帮忙排查了些问题，变将其记录下来。

## Evaluator 的 停止条件
问题描述：分布式情况下，使用 tf.estimator.train_and_evaluate 来训练任务，为什么 evaluator 不退出。    
解释：evaluator 是一种 worker 角色，专门用来测试模型在验证集的效果。
Google 的官方文档中讲述 worker 的退出条件是：数据跑完，或 global_step >= max_step 。    
但其实 Evaluator 的停止条件只有一个：global_step >= max_step 。因此想要改 worker 退出，只需要想办法满足这个条件。当然，如果是公司内部的 AI 平台，可以实现：当 chief worker 退出时，强制退出 Evaluator，这样就能满足那些只想通过数据 epoch 来配置训练的场景了。

## 分布式运行和单机运行的不同
tf.estimator.train_and_evaluate 既可以分布式也可以单机，这俩区分不同主要在退出条件。    
分布式的退出条件:数据跑完，或 global_step >= max_step 。      
单机退出条件：global_step >= max_step 。    
单机的运行机制类似于：    

~~~

while True:
	train()
	eval()
	if global_step >= max_step:
		break
	
~~~

## PS 是如何停止的
