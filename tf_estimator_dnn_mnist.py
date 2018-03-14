# -*- coding: utf-8 -*-

import os
print(os.path.abspath(__file__))

import tensorflow as tf

assert tf.__version__ >= '1.4.0', ('This code requires TensorFlow v1.4, '
                                  'You have:%s' % tf.__version__)

flags = tf.app.flags
flags.DEFINE_string("train_tfrecords", './raw_data/train.tfrecords', "Path for the train tfrecords file")
flags.DEFINE_string("test_tfrecords", './raw_data/test.tfrecords', "Path for the train tfrecords file")

FLAGS = flags.FLAGS


tf.logging.set_verbosity(tf.logging.DEBUG)


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
                     shuffle_window=1024, repeat_count=1):
    def input_fun():
        dataset = tf.data.TFRecordDataset([file_path])
        dataset = dataset.map(parser)
        if perform_shuffle:
            dataset = dataset.shuffle(buffer_size=shuffle_window)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat(repeat_count)
        # iterator = dataset.make_one_shot_iterator()
        # image_batch, label_batch = iterator.get_next()
        # return image_batch, label_batch
        return dataset

    return input_fun


def create_validate_input_fun(input_fun):
    validate_images, validate_labels = input_fun().make_one_shot_iterator(
    ).get_next()
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
        # iterator = validate_dataset.make_one_shot_iterator()
        # validate_image_batch = iterator.get_next()
        # return validate_image_batch, None
        return validate_dataset

    return validate_input_fun


def main(_):
    feature_columns = [tf.feature_column.numeric_column('image', shape=784)]
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[784],
        n_classes=10,
        model_dir='./estimator_model_dir')
    train_input_fun = create_input_fun(FLAGS.train_tfrecords, repeat_count=2)
    classifier.train(input_fn=train_input_fun)
    test_input_fun = create_input_fun(FLAGS.test_tfrecords, perform_shuffle=False)
    validate_input_fun = create_validate_input_fun(test_input_fun)
    evaluate_result = classifier.evaluate(input_fn=test_input_fun)
    print("Evaluation results:")
    for key in evaluate_result:
        print("   {}, was: {}".format(key, evaluate_result[key]))

    predict_results = classifier.predict(input_fn=validate_input_fun)
    print("Predictions:")
    print(list(map(lambda x: x["class_ids"][0], predict_results)))

if __name__ == '__main__':
    tf.app.run(main=main)
