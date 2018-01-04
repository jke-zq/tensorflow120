# -*- coding: utf-8 -*-

import os
print(os.path.abspath(__file__))
import sys
import time
import tempfile

import tensorflow as tf
import numpy as np

flags = tf.app.flags
flags.DEFINE_string("data_dir", "./", "Directory for storing mnist data")
flags.DEFINE_string("train_tfrecords", None, "Path for the train tfrecords file")
flags.DEFINE_string("test_tfrecords", None, "Path for the test tfrecords file")
flags.DEFINE_string("model_dir", "./", "Directory for storing mnist data")
flags.DEFINE_integer("task_index", None,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the master worker task the performs the variable "
                     "initialization ")
flags.DEFINE_integer("num_gpus", 2,
                     "Total number of gpus for each machine."
                     "If you don't use GPU, please set it to '0'")
flags.DEFINE_integer("total_step", 100, "total steps of training.")
flags.DEFINE_integer("replicas_to_aggregate", None,
                     "Number of replicas to aggregate before parameter update"
                     "is applied (For sync_replicas mode only; default: "
                     "num_workers)")
flags.DEFINE_string("ps_hosts", "localhost:2222",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "localhost:2223,localhost:2224",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("job_name", None, "job name: worker or ps")

FLAGS = flags.FLAGS
assert tf.__version__ == '1.1.0', ('This code requires TensorFlow v1.1, You have:%s' % tf.__version__)

IMAGE_PIXELS = 28


def read_image(file_queue):
    reader = tf.TFRecordReader()
    key, value = reader.read(file_queue)
    _, serialized_example = reader.read(file_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([784])
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)
    return image, label


def read_image_batch(file_queue, batch_size):
    img, label = read_image(file_queue)
    capacity = 1000 * batch_size
    image_batch, label_batch = tf.train.shuffle_batch([img, label], batch_size=batch_size, capacity=capacity, min_after_dequeue=0)
    # image_batch, label_batch = tf.train.batch([img, label], batch_size=batch_size)
    one_hot_labels = tf.to_float(tf.one_hot(label_batch, 10, 1, 0))
    return image_batch, one_hot_labels


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def get_loss(train_images, train_labels, scope, reuse_variables=None):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        y = inference(train_images)
        y_ = tf.to_float(train_labels)
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=y_, logits=y))
    return cross_entropy


def get_weight_variable(shape):
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    return weights


def inference(train_images):
    with tf.variable_scope('layer1'):
        W = get_weight_variable([784, 10])
        b = tf.get_variable("biases", [10], initializer=tf.constant_initializer(0.0))
        x = tf.reshape(train_images, [-1, 784])
        y = tf.matmul(x, W) + b
    return y


def create_done_queue(num_workers):
    with tf.device("/job:ps/task:0"):
        return tf.FIFOQueue(num_workers, tf.int32, shared_name="done_queue0")


def main(_):
    # distribution check
    if FLAGS.job_name is None or FLAGS.job_name == "":
        raise ValueError("Must specify an explicit `job_name`")
    if FLAGS.task_index is None or FLAGS.task_index =="":
        raise ValueError("Must specify an explicit `task_index`")
    if FLAGS.train_tfrecords is None:
        raise ValueError("Must specify an explicit `train_tfrecords`")
    if FLAGS.test_tfrecords is None:
        raise ValueError("Must specify an explicit `test_tfrecords`")

    print("job name = %s" % FLAGS.job_name)
    print("task index = %d" % FLAGS.task_index)

    ps_spec = FLAGS.ps_hosts.split(",")
    worker_spec = FLAGS.worker_hosts.split(",")
    num_workers = len(worker_spec)
    cluster = tf.train.ClusterSpec({
        "ps": ps_spec,
        "worker": worker_spec})
    kill_ps_queue = create_done_queue(num_workers)
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    if FLAGS.job_name == "ps":
        with tf.Session(server.target) as sess:
            for i in range(num_workers):
                sess.run(kill_ps_queue.dequeue())
        return

    is_chief = (FLAGS.task_index == 0)
    worker_device = "/job:worker/task:%d" % FLAGS.task_index
    with tf.device(tf.train.replica_device_setter(worker_device=worker_device, ps_device="/job:ps/cpu:0", cluster=cluster)):
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        reuse_variables = False
        tower_grads = []
        train_labels_list = []
        loss_list = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('GPU_%d' % i) as scope:
                        train_image_filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(FLAGS.train_tfrecords), shuffle=False)
                        train_images, train_labels = read_image_batch(train_image_filename_queue, 12 * FLAGS.num_gpus)
                        labels = tf.argmax(train_labels, 1)
                        train_labels_list.append(labels)
                        cross_entropy = get_loss(train_images, train_labels, scope, reuse_variables)
                        loss_list.append(cross_entropy)
                        reuse_variables = True
                        opt_gpu = tf.train.GradientDescentOptimizer(0.5)
                        grads = opt_gpu.compute_gradients(cross_entropy)
                        tower_grads.append(grads)
        with tf.device('/cpu:0'):
            opt = tf.train.GradientDescentOptimizer(0.5)
            if FLAGS.replicas_to_aggregate is None:
                replicas_to_aggregate = num_workers
            else:
                replicas_to_aggregate = FLAGS.replicas_to_aggregate
            opt = tf.train.SyncReplicasOptimizer(
                opt, use_locking=False,
                replicas_to_aggregate=replicas_to_aggregate,
                total_num_replicas=num_workers,
                name="mnist_sync_replicas")
            reduce_grads = average_gradients(tower_grads)
            train_step = opt.apply_gradients(reduce_grads, global_step=global_step)
        test_image_filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(FLAGS.train_tfrecords))
        test_images, test_labels = read_image_batch(test_image_filename_queue, 100)
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            y_pred = inference(test_images)
        y_test = tf.to_float(test_labels)
        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_test, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        chief_queue_runner = opt.get_chief_queue_runner()
        token_nums = max(replicas_to_aggregate - num_workers, 0)
        sync_init_op = opt.get_init_tokens_op(token_nums)

        init_op = tf.global_variables_initializer()
        kill_ps_enqueue_op = kill_ps_queue.enqueue(1)

        sv = tf.train.Supervisor(
            is_chief=is_chief,
            init_op=init_op,
            logdir=FLAGS.model_dir,
            recovery_wait_secs=1,
            global_step=global_step)

        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)

        if is_chief:
            print("Worker %d: Initializing session..." % FLAGS.task_index)
        else:
            print("Worker %d: Waiting for session to be initialized..." %
                  FLAGS.task_index)
        sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)

        print("Worker %d: Session initialization complete." % FLAGS.task_index)
        if is_chief:
            sv.start_queue_runners(sess, [chief_queue_runner])
            sess.run(sync_init_op)

        b_time = time.time()
        while not sv.should_stop():
            time.sleep(1)
            print('=======================================')
            loss_1, loss_2, _, step, labels_1, labels_2 = sess.run(loss_list + [train_step, global_step] + train_labels_list)
            print('global_step:%s, cost_time:%s, loss:%s-%s, ====data:%s-%s' % (step, time.time() - b_time, loss_1, loss_2, labels_1, labels_2))
            if step >= FLAGS.total_step:
                break

        print("accuracy: ", sess.run(accuracy))
        sess.run(kill_ps_enqueue_op)
        print('kill_ps_enqueue_op done....')
    sv.stop()

if __name__ == '__main__':
    tf.app.run(main=main)
