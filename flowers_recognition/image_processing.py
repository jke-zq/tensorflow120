# -*- coding: utf-8 -*-

import os
import time

import tensorflow as tf
import numpy as np
import vgg16

import skimage
import skimage.io
import skimage.transform

# synset = [l.strip() for l in open('synset.txt').readlines()]

LABEL_INDEX_MAP = {
    'daisy': 0,
    'dandelion': 1,
    'roses': 2,
    'sunflowers': 3,
    'tulips': 4,
}


# returns image of shape [224, 224, 3]
# [height, width, depth]
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


def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def write_to_tfrecord(codes, labels, tfrecord_file):
    """ This example is to write a sample to TFRecord file. If you want to write
    more samples, just use a loop.
    """
    writer = tf.python_io.TFRecordWriter(tfrecord_file)
    length = len(codes)
    for ith in range(length):
        code, label = codes[ith], LABEL_INDEX_MAP[labels[ith]]
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': _int64_feature(label),
            'code': _float_feature(code.tolist()),
        }))
        writer.write(example.SerializeToString())
    writer.close()


def read_from_tfrecord(filename):
    tfrecord_file_queue = tf.train.string_input_producer([filename],
                                                         name='queue')
    reader = tf.TFRecordReader()
    _, tfrecord_serialized = reader.read(tfrecord_file_queue)

    # label and image are stored as bytes but could be stored as 
    # int64 or float64 values in a serialized tf.Example protobuf.
    tfrecord_features = tf.parse_single_example(tfrecord_serialized,
                                                features={
                                                    'label': tf.FixedLenFeature(
                                                            [], tf.int64),
                                                    'code': tf.FixedLenFeature(
                                                            [4096],
                                                            tf.float32),
                                                }, name='features')
    # image was saved as uint8, so we have to decode as uint8.
    label = tfrecord_features['label']
    code = tfrecord_features['code']
    return label, code


def read_tfrecord(filename, batch_size=3):
    label, code = read_from_tfrecord(filename)
    capacity = batch_size
    min_after_dequeue = 0
    b_label, b_code = tf.train.shuffle_batch(
            [label, code],
            batch_size=batch_size,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue)
    return b_label, b_code


data_dir = 'flower_photos/'
tfrecord_file = os.path.join(data_dir, 'vgg_codes_lables')
contents = os.listdir(data_dir)
classes = [each for each in contents if os.path.isdir(data_dir + each)]

# Set the batch size higher if you can fit in in your GPU memory
batch_size = 8
b_imgs = []
lables = []
codes = None
r_label, r_code = read_tfrecord(tfrecord_file)
btime = time.time()
with tf.Session() as sess:
    vgg = vgg16.Vgg16()
    input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
    with tf.name_scope("content_vgg"):
        vgg.build(input_)
    for each in classes:
        print("Starting {} images".format(each))
        class_path = data_dir + each
        files = os.listdir(class_path)
        for ii, file in enumerate(files, 1):
            # Add images to the current batch
            # utils.load_image crops the input images for us, from the center
            img = load_image(os.path.join(class_path, file))
            b_imgs.append(img.reshape((1, 224, 224, 3)))
            lables.append(each)
            # Running the batch through the network to get the codes
            if ii % batch_size == 0 or ii == len(files):
                # Image batch to pass to VGG network
                images = np.concatenate(b_imgs)
                b_codes = sess.run(vgg.relu6, feed_dict={input_: images})
                if codes is None:
                    codes = b_codes
                else:
                    codes = np.concatenate((codes, b_codes))
                b_imgs = []
                print('Have processed %s images, cost time:%s secs'
                      % (ii + 1, time.time() - btime))

    print(len(lables), codes.shape)
    write_to_tfrecord(codes[:-100], lables[:-100], tfrecord_file + '_train')
    write_to_tfrecord(codes[-100:], lables[-100:], tfrecord_file + '_test')
