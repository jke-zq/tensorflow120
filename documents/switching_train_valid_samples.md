# 概述
一般来说，模型会在训练数据集上训练出满意的 loss 值之后，再在验证数据集中验证其泛化能力。    
但有些算法工程师可能更愿意在训练数据集上进行训练的同时，不断的抽查目前模型在验证数据集上的效果。    

那么能否在工程上实现呢？答案是肯定的。总共有三种方式：

1. DataSet 切换
2. SubGraph 切换
3. Estimator 的 evaluator worker

下面将分别介绍这三种方式，并给出具体的实现思路。

* DataSet
	* from_string_handle
	* Using SessionRunHook
	* Why SessionRunHook
* SubGraph
	* SubGraph
	* SubGraph 的具体实现
    * DataSet 的数据切换简介
* Estimator
	* chief, worker, evaluator
	* 实现

示例代码为：

## 切换数据集

DataSet 的切换方式有好几种，下面简述两种方式：

* from_string_handle
* from_structure

from_structure 是可重新初始化的迭代器。需要事先获取初始化操作，并在切换 DataSet 的时候进行初始化操作。更多详细信息可参见文档【数据集和估算器】。    
但这种方式有一个问题，每次初始化操作完成后，数据都是从头开始读取。一旦切换数据集，都会从切换到的数据集的开头开始读取，并不会从上次读取的位置继续开始。虽然这种方式简单，但是考虑到它的灵活性不好，遂放弃。在能满足你的需求的情况下，建议优先考虑这种方式，因为简单易于理解，并且 TensorFlow 支持完善。    

from_string_handle 是另外一种可选取的方式。相对于 from_structure 的每次切换数据集都从头开始读取，这种方式能够灵活的从上次读取的位置继续读取，并且切换的方式通过 feed_dict 的方式来进行，并不需要每次切换前都要进行初始化操作。 
   
### from_string_handle
使用的方式如下：    

~~~
train_dataset = tf.data.Dataset(...)
train_handle_str = train_dataset.make_one_shot_iterator().string_handle()
valid_dataset = tf.data.Dataset(...)
valid_handle_str = valid_dataset.make_one_shot_iterator().string_handle()

handle = tf.placeholder(tf.string, shape=[])

iterator = tf.data.Iterator.from_string_handle(
    handle, train_dataset.output_types, train_dataset.output_shapes)

next_elements = iterator.get_next()

sess = tf.Session()
train_handle = sess.run(train_handle_str)
valid_handle = sess.run(valid_handle_str)

train_next_elements = sess.run(next_elements, feed_dict={handle: train_handle})
valid_next_elements = sess.run(next_elements, feed_dict={handle: valid_handle})

~~~

如上代码所示，使用步骤为：

1. 定义数据集
2. 定义可通过 handle 标识切换的迭代器，并迭代出样本数据
3. 定义 placeholder，用来通过 feed_dict 的方式来切换数据集
4. 计算出每个数据集 handle 的标识
5. 通过 feed_dict 将对应数据集的标识传入，并获取到对应数据集的样本数据


### Using SessionRunHook
但我们代码中并没有直接使用 sess.run 去计算 DataSet string_handle 的值，而是使用了 tf.train.MonitoredTrainingSession（下面简称 mot_sess ） 配合 SessionRunHook 来计算的。mot_sess 创建参数中有一个关键字参数：hooks 。hooks 是个 list 类型，里面是各种 hook class 的 instance ，例如：StopAtStepHook(...) 、SessionRunHook(...) 。    
SessionRunHook 是 mot_sess.run 的调用扩展。通过继承该类，我们可以自定义 session 的创建前后的行为，sess.run 的前后的行为，以及 session 被创建后的行为。具体实现上，我们继承了 SessionRunHook ，并实现了其 after_create_session 方法。该方法会在 session 创建后只被调用一次，恰好能满足我们只计算一次 DataSet string_handle 数值的需求。    
实现方式如下：

~~~
class DS_Handle_Hook(tf.train.SessionRunHook):
    def __init__(self, train_str, valid_str):
    	self.train_handle = None
    	self.valid_handle = None
   
    def after_create_session(self, session, coord):
        del coord
        if self.train_str is not None:
            self.train_handle, self.valid_handle = session.run([self.train_str,
                                                                self.valid_str])
        tf.logging.info('session run ds string-handle done....')
        
ds_handle_hook = DS_Handle_Hook(...)
hooks = [ds_handle_hook]
sess = tf.train.MonitoredTrainingSession(..., hooks=hooks)
train_vals = sess.run(vals, feed_dict={handle: ds_handle_hook.train_handle})
valid_vals = sess.run(vals, feed_dict={handle: ds_handle_hook.valid_handle})
~~~
### Why SessionRunHook
为什么不能直接使用 sess.run 来计算 DataSet 的 string_handle 呢？ 
   
简单回答就是 mot_sess 的第一次 sess.run 会计算运行整个 Graph 。因为我们使用了 placeholder ，所以需要我们 feed_dict 。但显然我们是需要先计算出这个值的，这就陷入了死循环了。而使用 SessionRunHook 就会避免这个问题。    
这个问题的存在应该属于缺陷，也是对 DataSet 支持不完善的地方。但这在未来或许会有更改，期待 ~ 。

## SubGraph
如果我们不是用 DataSet 的方式，那该如何实现数据集的切换呢。我们可以切换 Graph ，一个输入验证集的 validation_graph ， 一个输入训练集的 training_graph 。当我们来回切换 Graph 的时候，就能达到切换数据集的目的。但问题是两个 Graph 必须得共享可训练的变量才行，且 validation_graph 不能改变这些变量。

### SubGraph
如何切换 Graph ？    
按照 TensorFlow 的设计，每个 Session 的同一时间只能有一个默认的 Graph 。那如何在同一个 session 中来回切换俩 Graph 呢？这个可以通过在同一个 Graph 中定义俩个 SubGraph 来实现。即我们在一个 session 中来回切换一个 Graph 的不同的 SubGraph 。    

如何共享且不改变可训练参数?    
TensorFlow 的变量有 scope 的概念，只要我们在 scope 中设置可重用，就可以在定义 validation_graph 时获取到 training_graph 事先定义好的变量了。只要 validation_graph 不进行 BackPropagation 计算，就不会更改这些变量。
### SubGraph 的具体实现
那么该如何实现呢？下面将用一段伪代码的方式来介绍。    
代码如下：

~~~
train_model = Model()
train_op, train_loss, train_acc = train_model.train(...)
 # 一般只在 chief worker 上做数据集切换
if self.is_chief:
	tf.get_variable_scope().reuse_variables()
	valid_model = Model()
	_, valid_loss, valid_acc = valid_model.train(...)

with sess:
	sess.run([train_op, train_loss, train_acc])
	if condition and self.is_chief:
		sess.run([valid_loss, valid_acc])
~~~

## Estimator
Estimator 属于 TensorFlow 的高级 API 。只要配置好 TF_CONFIG 环境变量，就可以将单机训练任务无缝变成分布式训练任务。在 TF_CONFIG 中，会有各种角色，其中 evaluator 角色负责间歇性的加载 checkpoint 文件，并执行 eval 任务（即在验证数据集上进行泛化能力验证）。

因此，如果使用 Estimator 的话，只要启动 evaluator job 就可以在训练的同时，间歇性的验证模型在验证数据集上的泛化能力了。

这种方式的好处也很明显，就是在不干扰训练的情况下，另起一个 job 来进行。
### ps, chief, worker, evaluator

* ps ：parameter server job
* chief ：master worker job ，负责初始化变量、保存模型等任务
* worker ：slave worker job ，只负责训练任务
* evaluator ：evaluator job ，负责加载模型，在验证数据集上在 loss 计算。



### evaluator 具体实现
我通常的做法是，在下面各个 server 上依次启动：

~~~
 # server0 --> [ps]：
python main.py --ps_hosts=server0 --worker_hosts=server1,server2,server3 --job_name=ps --task_index=0
 # server1 --> [chief]：
python main.py --ps_hosts=server0 --worker_hosts=server1,server2,server3 --job_name=worker --task_index=0
 # server2 --> [worker]:
python main.py --ps_hosts=server0 --worker_hosts=server1,server2,server3 --job_name=worker --task_index=1
 # server3 --> [evaluator]:
python main.py --ps_hosts=server0 --worker_hosts=server1,server2,server3 --job_name=worker --task_index=2
~~~

生成和设置 TF_CONFIG 的代码如下：

~~~
ps_hosts = FLAGS.ps_hosts.split(",")
worker_hosts = FLAGS.worker_hosts.split(",")
assert len(worker_hosts) > 2, 'workers shoule be more than two.'
chief_hosts = worker_hosts[0:1]
worker_hosts = worker_hosts[1:]
print('Chief host is :%s' % chief_hosts)
print('PS hosts are: %s' % ps_hosts)
print('Worker hosts are: %s' % worker_hosts[-1:])
if FLAGS.job_name == 'worker' and FLAGS.task_index > 0:
    job_name = FLAGS.job_name
    task_index = FLAGS.task_index - 1
elif FLAGS.job_name == 'worker' and FLAGS.task_index == 0:
    job_name = 'chief'
    task_index = 0
else:
    job_name = FLAGS.job_name
    task_index = FLAGS.task_index
if FLAGS.job_name == 'worker' and task_index == len(worker_hosts) - 1:
    job_name = 'evaluator'
    task_index = 0
worker_hosts = worker_hosts[:-1]
print('job_name : %s' % job_name)
print('task_index : %d' % task_index)
cluster = {'chief': chief_hosts, "ps": ps_hosts,
                "worker": worker_hosts}
os.environ['TF_CONFIG'] = json.dumps(
        {'cluster': cluster,
         'task': {'type': job_name, 
                  'index': task_index}})
~~~









