# coding=utf-8
# ref: https://github.com/tensorflow/models/blob/master/samples/core/get_started/custom_estimator.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer('batch_size', 100, 'batch_size')
flags.DEFINE_integer('train_steps', 1000, 'number of training steps')
flags.DEFINE_string('train_file_path', './raw_data/iris_training.csv',
                    'data for training')
flags.DEFINE_string('test_file_path', './raw_data/iris_test.csv',
                    'data for test')
FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.DEBUG)

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
CSV_TYPES = [[0.0], [0.0], [0.0], [0.0], [0]]


def _parse_line(line):
    fields = tf.decode_csv(line, record_defaults=CSV_TYPES)
    features = dict(zip(CSV_COLUMN_NAMES, fields))
    label = features.pop('Species')
    return features, label


def csv_input_fn(csv_path, batch_size, repeat=None):
    dataset = tf.data.TextLineDataset(csv_path).skip(1)
    dataset = dataset.map(_parse_line)
    dataset = dataset.shuffle(1000).repeat(count=repeat).batch(batch_size)
    return dataset


def eval_input_fn(features, labels, batch_size):
    features = dict(features)
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)
    # Return the dataset.
    return dataset


def my_model(features, labels, mode, params):
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
    logits = tf.layers.dense(net, params['n_classes'], activation=None)
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)
    assert mode == tf.estimator.ModeKeys.TRAIN
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def main(_):
    my_feature_columns = []
    for key in CSV_COLUMN_NAMES[:-1]:
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))
    classifier = tf.estimator.Estimator(
            model_fn=my_model,
            params={
                'feature_columns': my_feature_columns,
                'hidden_units': [10, 10],
                'n_classes': 3,
            })
    classifier.train(
            input_fn=lambda: csv_input_fn(FLAGS.train_file_path,
                                          FLAGS.batch_size),
            steps=FLAGS.train_steps)
    eval_result = classifier.evaluate(
            input_fn=lambda: csv_input_fn(FLAGS.test_file_path,
                                          FLAGS.batch_size,
                                          repeat=1))
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
    expected = ['Setosa', 'Versicolor', 'Virginica']
    predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }
    predictions = classifier.predict(
            input_fn=lambda: eval_input_fn(predict_x,
                                           labels=None,
                                           batch_size=FLAGS.batch_size))

    for pred_dict, expec in zip(predictions, expected):
        template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(SPECIES[class_id],
                              100 * probability, expec))


if __name__ == '__main__':
    tf.app.run(main=main)
