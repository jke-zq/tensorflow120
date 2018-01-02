# -*- coding: utf-8 -*-

import tensorflow as tf
import vgg16

import skimage
import skimage.io
import skimage.transform

flags = tf.app.flags
flags.DEFINE_string("imag_path", '.', "Path for the image")

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.DEBUG)

LABEL_INDEX_MAP = {
    'daisy': 0,
    'dandelion': 1,
    'roses': 2,
    'sunflowers': 3,
    'tulips': 4,
}

INDEX_LABEL_MAP = dict(zip(LABEL_INDEX_MAP.values(), LABEL_INDEX_MAP.keys()))


def load_image(path):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224),
                                           mode='constant')
    return resized_img


def create_validate_input_fun(codes):
    def validate_input_fun():
        validate_dataset = tf.data.Dataset.from_tensor_slices(codes)

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
    input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16.Vgg16()
    vgg.build(input_)
    with tf.Session() as sess:
        img = load_image(FLAGS.imag_path)
        img = img.reshape((1, 224, 224, 3))

        feed_dict = {input_: img}
        codes = sess.run(vgg.relu6, feed_dict=feed_dict)

    with tf.Session() as sess:
        feature_columns = [
            tf.feature_column.numeric_column('code', shape=4096)]
        classifier = tf.estimator.DNNClassifier(
                feature_columns=feature_columns,
                hidden_units=[512],
                n_classes=5,
                model_dir='./estimator_model_dir')
        # train_input_fun = create_input_fun(FLAGS.train_tfrecords, repeat_count=2)
        # classifier.train(input_fn=train_input_fun)
        # test_input_fun = create_input_fun(FLAGS.test_tfrecords, repeat_count=1, perform_shuffle=False)
        validate_input_fun = create_validate_input_fun(codes)
        # evaluate_result = classifier.evaluate(input_fn=test_input_fun)
        # print("Evaluation results:")
        # for key in evaluate_result:
        #     print("   {}, was: {}".format(key, evaluate_result[key]))

        predict_results = classifier.predict(input_fn=validate_input_fun)
        print("Predictions:")
        print(list(map(lambda x: INDEX_LABEL_MAP[x["class_ids"][0]],
                       predict_results)))


if __name__ == '__main__':
    tf.app.run(main=main)
