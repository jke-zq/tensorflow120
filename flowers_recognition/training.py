# -*- coding: utf-8 -*-

import tensorflow as tf

assert tf.__version__ >= '1.4.0', ('This code requires TensorFlow v1.4, '
                                   'You have:%s' % tf.__version__)

flags = tf.app.flags
flags.DEFINE_string("test_tfrecords", 'vgg_codes_lables_test',
                    "Path for the test tfrecords file")
flags.DEFINE_string("train_tfrecords", 'vgg_codes_lables_train',
                    "Path for the train tfrecords file")

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.DEBUG)

LABEL_INDEX_MAP = {
    'daisy': 0,
    'dandelion': 1,
    'roses': 2,
    'sunflowers': 3,
    'tulips': 4,
}


def parser(serialized_example):
    features = tf.parse_single_example(
            serialized_example,
            features={
                'label': tf.FixedLenFeature([], tf.int64),
                'code': tf.FixedLenFeature([4096], tf.float32),
            })

    label = features['label']
    # label = LABEL_INDEX_MAP[label]
    code = features['code']
    return {'code': code}, label


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
        code_b, label_b = iterator.get_next()
        return code_b, label_b

    return input_fun


def create_validate_input_fun(input_fun):
    validate_codes, validate_labels = input_fun()
    with tf.train.MonitoredTrainingSession() as sess:
        validate_code_vals, validate_label_vals = sess.run(
                [validate_codes, validate_labels])
    print('validate labels:')
    print(str(list(validate_label_vals)))

    def validate_input_fun():
        validate_dataset = tf.data.Dataset.from_tensor_slices(
                validate_code_vals['code'])

        def decode(image_raw):
            return {'code': image_raw}

        validate_dataset = validate_dataset.map(decode)
        validate_dataset = validate_dataset.batch(2)
        validate_dataset = validate_dataset.repeat(1)
        iterator = validate_dataset.make_one_shot_iterator()
        validate_code_batch = iterator.get_next()
        return validate_code_batch, None

    return validate_input_fun


def main(_):
    feature_columns = [tf.feature_column.numeric_column('code', shape=4096)]
    classifier = tf.estimator.DNNClassifier(
            feature_columns=feature_columns,
            hidden_units=[512],
            n_classes=5,
            model_dir='./estimator_model_dir')
    train_input_fun = create_input_fun(FLAGS.train_tfrecords, repeat_count=2)
    classifier.train(input_fn=train_input_fun)
    test_input_fun = create_input_fun(FLAGS.test_tfrecords, repeat_count=1,
                                      perform_shuffle=False)
    validate_input_fun = create_validate_input_fun(test_input_fun)
    evaluate_result = classifier.evaluate(input_fn=test_input_fun)
    print("Evaluation results:")
    for key in evaluate_result:
        print("   {}, was: {}".format(key, evaluate_result[key]))

    predict_results = classifier.predict(input_fn=validate_input_fun)
    print("Predictions:")
    print(list(map(lambda x: x["class_ids"][0], predict_results)))


def test_main(_):
    test_input_fun = create_input_fun(FLAGS.test_tfrecords, repeat_count=1,
                                      perform_shuffle=False)
    validate_images, validate_labels = test_input_fun()
    with tf.train.MonitoredTrainingSession() as sess:
        validate_image_vals, validate_label_vals = sess.run(
                [validate_images, validate_labels])
    print('validate labels:')
    print(validate_label_vals)
    print(validate_image_vals)


if __name__ == '__main__':
    tf.app.run(main=main)
