import tensorflow as tf

def slice_shuffle_batch(list_inputs, transfer_fun=None, epochs=1):
    tensor_inputs = map(lambda input: tf.convert_to_tensor(input, dtype=tf.int64), list_inputs)
    input_queue = tf.train.slice_input_producer(tensor_inputs, num_epochs=1)
    if transfer_fun:
        input_queue = transfer_fun(input_queue)
    return tf.train.shuffle_batch(input_queue, batch_size=5, capacity=100, min_after_dequeue=20)

def transfer(tensor_lists):
    tensor_lists[0] = tf.one_hot(tensor_lists[0], 10)
    return tensor_lists

with tf.Session() as sess:
    one_hot_list, number_list = range(10), range(10, 20)
    one_hot_batch, number_batch = slice_shuffle_batch([one_hot_list, number_list], transfer)
    sess.run(tf.global_variables_initializer())
    # just for tf.train.slice_input_producer
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        while not coord.should_stop():
            test_one_hot, test_number = sess.run([one_hot_batch, number_batch])
            print test_one_hot, test_number
            import time
            time.sleep(5)
    except tf.errors.OutOfRangeError as e:
        print e, 'it\'s out of range.'
    finally:
        coord.request_stop()
    coord.join(threads)
