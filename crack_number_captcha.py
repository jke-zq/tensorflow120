# -*- coding: utf-8 -*-

"""
REF:http://blog.topspeedsnail.com/archives/10858#more-10858
"""

import random
import os

from captcha.image import ImageCaptcha  # pip install captcha
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

NUMBERS = map(str, range(10))

CHAR_SET = NUMBERS
CHAR_SET_LEN = len(CHAR_SET)

CHAR_POS_MAP = {v: k for k, v in enumerate(CHAR_SET)}
POS_CHAR_MAP = {v: k for k, v in CHAR_POS_MAP.iteritems()}

CAPTCHA_LEN = 4


def random_captcha_text(char_set=NUMBERS, captcha_size=4):
        return ''.join(random.choice(char_set) for _ in range(captcha_size))


def gen_captcha_text_and_image():
        image = ImageCaptcha()
        captcha_text = random_captcha_text()
        captcha = image.generate(captcha_text)
        # write the image to HD
        # image.write(captcha_text, captcha_text + '.jpg')
        captcha_image = Image.open(captcha)
        captcha_image = np.array(captcha_image)
        # captcha_grey_image = convert2gray(captcha_image)
        # im = Image.fromarray(captcha_grey_image).convert('RGB')
        # im.save('grey_' + captcha_text + '.jpeg')
        return captcha_text, captcha_image


def convert2gray(img):
        if len(img.shape) > 2:
                # gray = np.mean(img, -1)
                # 上面的转法较快，正规转法如下
                r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
                gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
                return gray
        else:
                return img


def text2vec(text):
        vec = np.zeros(CAPTCHA_LEN * CHAR_SET_LEN)
        for i, c in enumerate(text):
                pos = CHAR_POS_MAP[c] + i * CHAR_SET_LEN
                vec[pos] = 1
        return vec


def vec2text(vec):
        nonzero_pos = vec.nonzero()[0]
        chars = []
        for i, p in enumerate(nonzero_pos):
                chars.append(POS_CHAR_MAP[i])
        return ''.join(chars)

IMAGE_HEIGHT = 60
IMAGE_WIDTH = 160

"""
cnn在图像大小是2的倍数时性能最高, 如果你用的图像大小不是2的倍数，可以在图像边缘补无用像素。
np.pad(image,((2,3),(2,2)), 'constant', constant_values=(255,))  # 在图像上补2行，下补3行，左补2行，右补2行
"""


def get_next_batch(batch_size=128):
        batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
        batch_y = np.zeros([batch_size, CAPTCHA_LEN * CHAR_SET_LEN])

        def loop_get_text_and_image():
                while True:
                        text, image = gen_captcha_text_and_image()
                        if image.shape == (60, 160, 3):
                                return text, convert2gray(image)
                        else:
                                # print 'image shape is not expected:', image.shape
                                continue
        for i in range(batch_size):
                text, image = loop_get_text_and_image()
                batch_x[i, :] = image.flatten() / 255
                batch_y[i, :] = text2vec(text)
        return batch_x, batch_y


X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, CAPTCHA_LEN * CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32)


def build_crack_captcha_cnn():
        x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
        # 3 conv layer
        conv_1 = slim.conv2d(x, 32, [3, 3], 1, padding='SAME', scope='conv1')
        max_pool_1 = slim.max_pool2d(conv_1, [2, 2], [2, 2], padding='SAME')

        conv_2 = slim.conv2d(max_pool_1, 64, [3, 3], padding='SAME', scope='conv2')
        max_pool_2 = slim.max_pool2d(conv_2, [2, 2], [2, 2], padding='SAME')

        conv_3 = slim.conv2d(max_pool_2, 64, [3, 3], padding='SAME', scope='conv3')
        max_pool_3 = slim.max_pool2d(conv_3, [2, 2], [2, 2], padding='SAME')

        # Fully connected layer
        # with tf.variable_scope('layer4-fc1'):
        #         fc1_weights = tf.get_variable('weight', [8 * 20 * 64, 1024], initializer=tf.truncated_normal_initializer(stddev=0.1))
        #         # L1
        #         # tf.add_to_collection('losses', )
        #         fc1_biases = tf.get_variable('bias', [1024], initializer=tf.constant_initializer(0.1))
        #         dense = tf.reshape(max_pool_3, [-1, fc1_weights.get_shape().as_list()[0]])
        #         fc1 = tf.nn.relu(tf.matmul(dense, fc1_weights) + fc1_biases)
        #         fc1 = tf.nn.dropout(fc1, 0.5)

        # with tf.variable_scope('layer5-fc2'):
        #         fc2_weights = tf.get_variable('weights', [1024, CAPTCHA_LEN * CHAR_SET_LEN])
        #         fc2_biases = tf.get_variable('bias', [CAPTCHA_LEN * CHAR_SET_LEN], initializer=tf.constant_initializer(0.1))
        #         logits = tf.matmul(fc1, fc2_weights) + fc2_biases
        #         return logits
        flatten = slim.flatten(max_pool_3)
        fc1 = slim.fully_connected(slim.dropout(flatten, keep_prob), 1024, activation_fn=tf.nn.tanh, scope='fc1')
        logits = slim.fully_connected(fc1, CAPTCHA_LEN * CHAR_SET_LEN, activation_fn=None, scope='fc2')
        return logits


MODEL_SAVED_DIR = 'crack_captcha.saved'
MONITOR_BOARD_DIR = 'monitor_board'


def train_crack_captcha_cnn():
        output = build_crack_captcha_cnn()
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=tf.argmax(Y, 1)))
        global_step = tf.get_variable("step", [], dtype=tf.int64, initializer=tf.constant_initializer(0), trainable=False)
        rate = tf.train.exponential_decay(1e-3, global_step, decay_steps=200, decay_rate=0.97, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=rate).minimize(loss, global_step=global_step)

        predict = tf.reshape(output, [-1, CAPTCHA_LEN, CHAR_SET_LEN])
        max_idx_p = tf.argmax(predict, 2)
        max_idx_l = tf.argmax(tf.reshape(Y, [-1, CAPTCHA_LEN, CHAR_SET_LEN]), 2)
        correct_pred = tf.equal(max_idx_p, max_idx_l)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(MODEL_SAVED_DIR)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        merged = tf.summary.merge_all()
        with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                board_writer = tf.summary.FileWriter(MONITOR_BOARD_DIR, sess.graph)
                # TODO: board_writer.close()
                if ckpt:
                        saver.restore(sess, ckpt)
                        print("restore from the checkpoint {0}".format(ckpt))
                        start_step = int(ckpt.split('-')[-1])
                        print start_step, 'from this.'
                while True:
                        batch_x, batch_y = get_next_batch(256)
                        summary, _, loss_, step = sess.run([merged, optimizer, loss, global_step], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
                        board_writer.add_summary(summary, step)
                        print('global_step:%s, loss:%s' % (step, loss_))
                        # 每100 step计算一次准确率
                        if step % 100 == 1:
                                batch_x_test, batch_y_test = get_next_batch(64)
                                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                                print('global_step:%s, acc:%s' % (step, acc))
                                saver.save(sess, os.path.join(MODEL_SAVED_DIR, 'model'), global_step=global_step)
train_crack_captcha_cnn()
