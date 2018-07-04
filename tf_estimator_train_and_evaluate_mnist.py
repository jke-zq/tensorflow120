
'''
usages(acting like using tf.train.Supervisor):
    for ps:
        python .py --job_name='ps' --task_index=0 --ps_hosts='host1:port1' --worker_hosts='host2:port3,host3:port3,host4:port4' --train_tfrecords='train.tfrecords' --test_tfrecords='test.tfrecords'
    for worker_01(chief):
        python .py --job_name='worker' --task_index=0 --ps_hosts='host1:port1' --worker_hosts='host2:port3,host3:port3,host4:port4' --train_tfrecords='train.tfrecords' --test_tfrecords='test.tfrecords'
    for worer_02:
        python .py --job_name='worker' --task_index=1 --ps_hosts='host1:port1' --worker_hosts='host2:port3,host3:port3,host4:port4' --train_tfrecords='train.tfrecords' --test_tfrecords='test.tfrecords'
    for worer_03(evaluator):
        python .py --job_name='worker' --task_index=1 --ps_hosts='host1:port1' --worker_hosts='host2:port3,host3:port3,host4:port4' --train_tfrecords='train.tfrecords' --test_tfrecords='test.tfrecords'

'''
# -*- coding: utf-8 -*-

import os
print(os.path.abspath(__file__))
import json

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string("train_tfrecords", './raw_data/train.tfrecords', "Path for the train tfrecords file")
flags.DEFINE_string("test_tfrecords", './raw_data/test.tfrecords', "Path for the train tfrecords file")
flags.DEFINE_string("ps_hosts", "localhost:2222", "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "localhost:2223,localhost:2224", "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("job_name", None, "job name: worker, evaluator or ps")
flags.DEFINE_integer("task_index", None, '')
flags.DEFINE_boolean("local", False, '')


FLAGS = flags.FLAGS
assert tf.__version__ >= '1.4.0', ('This code requires TensorFlow v1.4, You have:%s' % tf.__version__)

tf.logging.set_verbosity(tf.logging.DEBUG)


# def check_flags():
#     if FLAGS.job_name is None or FLAGS.job_name == "":
#         raise ValueError("Must specify an explicit `job_name`")
#     if FLAGS.task_index is None or FLAGS.task_index =="":
#         raise ValueError("Must specify an explicit `task_index`")
#     if FLAGS.train_tfrecords is None:
#         raise ValueError("Must specify an explicit `train_tfrecords`")
#     if FLAGS.test_tfrecords is None:
#         raise ValueError("Must specify an explicit `test_tfrecords`")


def get_config(local=False):
    if not local:
        print('distributed training...')
        ps_hosts = FLAGS.ps_hosts.split(",")
        worker_hosts = FLAGS.worker_hosts.split(",")
        assert len(worker_hosts) > 2, 'workers should be more than two.'
        chief_hosts = worker_hosts[0:1]
        worker_hosts = worker_hosts[1:]
        tf.logging.info('Chief host is :%s' % chief_hosts)
        tf.logging.info('PS hosts are: %s' % ps_hosts)
        tf.logging.info('Worker hosts are: %s' % worker_hosts[:-1])
        tf.logging.info('eval hosts are: %s' % worker_hosts[-1:])
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

        tf.logging.info('job_name : %s' % job_name)
        tf.logging.info('task_index : %d' % task_index)
        cluster = {'chief': chief_hosts, "ps": ps_hosts,
                        "worker": worker_hosts}
        os.environ['TF_CONFIG'] = json.dumps(
                {'cluster': cluster,
                 'task': {'type': job_name, 'index': task_index}})
        print('-----run config---------------')
        print(os.environ['TF_CONFIG'])
    else:
        print('local training...')
    return tf.estimator.RunConfig(save_checkpoints_steps=10)


def parser(serialized_example):
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
    return {'image': image}, label


def create_input_fun(file_path, batch_size=64, perform_shuffle=True,
                     shuffle_window=1024, repeat_count=None):
    def input_fun():
        dataset = tf.data.TFRecordDataset([file_path])
        dataset = dataset.map(parser)
        if perform_shuffle:
            dataset = dataset.shuffle(buffer_size=shuffle_window)
        dataset = dataset.batch(batch_size)
        if repeat_count is not None:
            dataset = dataset.repeat(repeat_count)
        else:
            dataset = dataset.repeat()
        iterator = dataset.make_one_shot_iterator()
        image_batch, label_batch = iterator.get_next()
        return image_batch, label_batch
    return input_fun


def create_validate_input_fun(input_fun):
    validate_images, validate_labels = input_fun()
    with tf.train.MonitoredTrainingSession() as sess:
        validate_image_vals, validate_label_vals = sess.run([validate_images, validate_labels])
    print('validate labels:')
    print(str(list(validate_label_vals)))

    def validate_input_fun():
        validate_dataset = tf.data.Dataset.from_tensor_slices(validate_image_vals['image'])

        def decode(image_raw):
            return {'image': image_raw}
        validate_dataset = validate_dataset.map(decode)
        validate_dataset = validate_dataset.batch(2)
        validate_dataset = validate_dataset.repeat(1)
        iterator = validate_dataset.make_one_shot_iterator()
        validate_image_batch = iterator.get_next()
        return validate_image_batch, None
    return validate_input_fun


def main(_):
    config = get_config(local=FLAGS.local)
    feature_columns = [tf.feature_column.numeric_column('image', shape=784)]
    estimator = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[784],
        n_classes=10,
        model_dir='./estimator_model_dir/',
        config=config)
    train_input_fn = create_input_fun(FLAGS.train_tfrecords,
                                      repeat_count=-1)
    eval_input_fn = create_input_fun(FLAGS.test_tfrecords, repeat_count=1, perform_shuffle=False)
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                        max_steps=100)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
                                      start_delay_secs=5, throttle_secs=10)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

if __name__ == '__main__':
    tf.app.run(main=main)
