# coding=utf-8

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string("train_tfrecords", './raw_data/train.tfrecords',
                    "Path for the train tfrecords file")
flags.DEFINE_string("test_tfrecords", './raw_data/test.tfrecords',
                    "Path for the test tfrecords file")
FLAGS = flags.FLAGS


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
        # label = tf.argmax(labels)
        # one_hot_labels = tf.to_float(tf.one_hot(labels, 10, 1, 0))
        return images, labels

    return read_examples


def get_dataset(file_names, batch_size=8, epoch=None):
    _parse_fn = generate_parse_fn(batch_size)
    files = tf.data.Dataset.list_files(file_names)
    dataset = files.apply(tf.contrib.data.parallel_interleave(
            tf.data.TFRecordDataset,
            cycle_length=4 * 2))
    dataset = dataset.prefetch(buffer_size=batch_size)
    # dataset = dataset.shuffle(buffer_size=batch_size * 4)
    dataset = dataset.prefetch(buffer_size=batch_size)
    dataset = dataset.repeat(epoch)
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(_parse_fn, num_parallel_calls=4)
    return dataset


def get_samples(handle):
    train_dataset = get_dataset(FLAGS.train_tfrecords)
    valid_dataset = get_dataset(FLAGS.test_tfrecords)
    _iter = tf.data.Iterator.from_string_handle(
            handle, train_dataset.output_types,
            train_dataset.output_shapes)
    images, labels = _iter.get_next()
    train_str = train_dataset.make_one_shot_iterator().string_handle()
    valid_str = valid_dataset.make_one_shot_iterator().string_handle()
    return train_str, valid_str, images, labels


class DSHandleHook(tf.train.SessionRunHook):
    def __init__(self, train_str, valid_str):
        self.train_str = train_str
        self.valid_str = valid_str
        self.train_handle = None
        self.valid_handle = None

    def after_create_session(self, session, coord):
        del coord
        if self.train_str is not None:
            self.train_handle, self.valid_handle = session.run([self.train_str,
                                                                self.valid_str])
        print('session run ds string-handle done....')


def main(_):
    handle = tf.placeholder(tf.string, shape=[])
    train_str, valid_str, images, labels = get_samples(handle)
    ds_handle_hook = DSHandleHook(train_str, valid_str)
    hooks = [ds_handle_hook]
    sess = tf.train.MonitoredTrainingSession(hooks=hooks)

    with sess:
        for __ in range(8):
            print('----train_handle------')
            image_vals, label_vals = sess.run([images, labels], feed_dict={
                handle: ds_handle_hook.train_handle})
            print(label_vals)
            print('----test_handle------')
            image_vals, label_vals = sess.run([images, labels], feed_dict={
                handle: ds_handle_hook.valid_handle})
            print(label_vals)


if __name__ == '__main__':
    tf.app.run(main=main)
