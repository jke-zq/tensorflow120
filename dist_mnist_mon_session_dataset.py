# -*- coding: utf-8 -*-

import os

print(os.path.abspath(__file__))
import sys
import time
import traceback

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string("data_dir", "./", "Directory for storing mnist data")
flags.DEFINE_string("train_tfrecords", './raw_data/train.tfrecords',
                    "Path for the train tfrecords file")
flags.DEFINE_string("model_dir", "./dist_mon_dataset", "Directory for storing "
                                                       "model data")
flags.DEFINE_integer("task_index", None,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the master worker task the performs the variable "
                     "initialization ")
flags.DEFINE_integer("num_gpus", 2,
                     "Total number of gpus for each machine."
                     "If you don't use GPU, please set it to '0'")
flags.DEFINE_integer("total_step", 10, "total steps of training.")
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
tf.logging.set_verbosity(tf.logging.DEBUG)
# assert tf.__version__ == '1.1.0', (
#     'This code requires TensorFlow v1.1, You have:%s' % tf.__version__)

IMAGE_PIXELS = 28


def generate_parse_fn(batch_size):
    def read_examples(examples):
        features = {}
        features['label'] = tf.FixedLenFeature([], tf.int64)
        features['image_raw'] = tf.FixedLenFeature([], tf.string)
        features = tf.parse_example(examples, features)
        images = tf.decode_raw(features['image_raw'], tf.uint8)
        images.set_shape([batch_size, 784])
        images = tf.cast(images, tf.float32) * (1. / 255) - 0.5
        labels = features['label']
        one_hot_labels = tf.to_float(tf.one_hot(labels, 10, 1, 0))
        return images, one_hot_labels

    return read_examples


def input_fn(file_names, batch_size, epoch=None):
    _parse_fn = generate_parse_fn(batch_size)
    files = tf.data.Dataset.list_files(file_names)
    # number_of_cpu is the value of worker.vcore in xml file
    dataset = files.apply(tf.contrib.data.parallel_interleave(
            tf.data.TFRecordDataset,
            cycle_length=4 * 2))
    # prefetch will buffer the previos op and improve the performance
    dataset = dataset.prefetch(buffer_size=batch_size)
    # times: user defined
    dataset = dataset.shuffle(buffer_size=batch_size * 4)
    # buffer the shuffle op and improve the perfromance
    dataset = dataset.prefetch(buffer_size=batch_size)
    dataset = dataset.repeat(epoch)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(_parse_fn, num_parallel_calls=4)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def get_loss_acc(train_images, train_labels, reuse_variables=None):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        y_ = inference(train_images)
        print(train_labels.shape)
        y = tf.to_float(train_labels)
        # print(y.shape)
        y.set_shape([128, 10])
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=y, logits=y_))
        correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return loss, accuracy


def get_weight_variable(shape):
    weights = tf.get_variable("weights", shape,
                              initializer=tf.truncated_normal_initializer(
                                      stddev=0.1))
    return weights


def inference(train_images):
    with tf.variable_scope('layer1'):
        w = get_weight_variable([784, 10])
        b = tf.get_variable("biases", [10],
                            initializer=tf.constant_initializer(0.0))
        # x = tf.reshape(train_images, [-1, 784])
        y = tf.matmul(train_images, w) + b
        print('ddddd%s' % y.shape)
    return y


def train(global_step):
    with tf.variable_scope(tf.get_variable_scope()):
        train_images, train_labels = input_fn(
                FLAGS.train_tfrecords, 128)
        loss, acc = get_loss_acc(train_images, train_labels)
        opt = tf.train.GradientDescentOptimizer(0.5)
        train_op = opt.minimize(loss, global_step=global_step)
        return train_op, loss, acc


class _QueueHook(tf.train.SessionRunHook):
    def __init__(self, enqueue_op):
        self.op = enqueue_op

    def end(self, session):
        session.run(self.op)
        tf.logging.info('kill_ps_enqueue_op done....')


class Training(object):
    def __init__(self):
        # distribution check
        if FLAGS.job_name is None or FLAGS.job_name == "":
            raise ValueError("Must specify an explicit `job_name`")
        if FLAGS.task_index is None or FLAGS.task_index == "":
            raise ValueError("Must specify an explicit `task_index`")
        if FLAGS.train_tfrecords is None:
            raise ValueError("Must specify an explicit `train_tfrecords`")
        # if FLAGS.test_tfrecords is None:
        #     raise ValueError("Must specify an explicit `test_tfrecords`")

        print("job name = %s" % FLAGS.job_name)
        print("task index = %d" % FLAGS.task_index)

        ps_spec = FLAGS.ps_hosts.split(",")
        worker_spec = FLAGS.worker_hosts.split(",")
        self.num_workers = len(worker_spec)
        self.cluster = tf.train.ClusterSpec({
            "ps": ps_spec,
            "worker": worker_spec})
        self.kill_ps_queue = self.create_done_queue(self.num_workers)
        self.server = tf.train.Server(self.cluster, job_name=FLAGS.job_name,
                                      task_index=FLAGS.task_index)
        self.is_chief = (FLAGS.task_index == 0)
        self.worker_device = "/job:worker/task:%d" % FLAGS.task_index
        self.sess = None

    def create_done_queue(self, num_workers):
        with tf.device("/job:ps/task:0"):
            return tf.FIFOQueue(num_workers, tf.int32,
                                shared_name="done_queue0")

    def create_session(self):
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                     device_filters=["/job:ps",
                                                     "/job:worker/task:%d" % FLAGS.task_index],
                                     log_device_placement=False)
        sess_config.gpu_options.allow_growth = True
        hooks = [tf.train.StopAtStepHook(num_steps=FLAGS.total_step),
                 _QueueHook(self.kill_ps_queue.enqueue(1))]
        self.sess = tf.train.MonitoredTrainingSession(
                master=self.server.target,
                is_chief=self.is_chief,
                checkpoint_dir=FLAGS.model_dir,
                scaffold=None,
                hooks=hooks,
                chief_only_hooks=None,
                save_checkpoint_secs=30,
                save_summaries_steps=100,
                # save_summaries_secs=USE_DEFAULT,
                config=sess_config,
                stop_grace_period_secs=120,
                log_step_count_steps=10,
                max_wait_secs=7200
        )
        return self.sess

    def create_session_wrapper(self, times=10):
        if times == 0:
            tf.logging.error('creating the session is out of times.')
            sys.exit(0)
        try:
            return self.create_session()
        except Exception as e:
            tf.logging.info(e)
            tf.logging.info('retry creating session:%s' % times)
            try:
                if self.sess is not None:
                    self.sess.close()
                else:
                    tf.logging.info('close session: sess is None!')
            except Exception as e:
                exc_info = traceback.format_exc(sys.exc_info())
                msg = 'creating session exception:%s\n%s' % (e, exc_info)
                tf.logging.warn(msg)
            return self.create_session_wrapper(times - 1)

    def do(self):
        if FLAGS.job_name == "ps":
            with tf.Session(self.server.target) as sess:
                for i in range(self.num_workers):
                    sess.run(self.kill_ps_queue.dequeue())
            return
        with tf.device(tf.train.replica_device_setter(
                worker_device=self.worker_device,
                ps_device="/job:ps/cpu:0",
                cluster=self.cluster)):
            global_step = tf.train.get_or_create_global_step()
            train_op, loss, acc = train(global_step)

        if self.is_chief:
            print("Worker %d: Initializing session..." % FLAGS.task_index)
        else:
            print("Worker %d: Waiting for session to be initialized..." %
                  FLAGS.task_index)

        b_time = time.time()
        self.create_session_wrapper()
        while not self.sess.should_stop():
            with self.sess as sess:
                time.sleep(1)
                print('=======================================')
                _, loss_val, acc_val, step = sess.run([train_op, loss, acc,
                                                       global_step])
                print('global_step:%s, cost_time:%s, loss:%s, acc:%s' % (
                    step, time.time() - b_time, loss_val, acc_val))
        print('Done!!')


def main(_):
    training = Training()
    training.do()


if __name__ == '__main__':
    tf.app.run(main=main)
