import tensorflow as tf

# random int tensor described by shap
def gen_random_tensor(sess, shap, minval, maxval):
    initializer = tf.random_uniform_initializer(minval=minval, maxval=maxval, dtype=tf.int64)
    random_tensor = tf.get_variable('random_tensor', shape=shap, initializer=initializer, dtype=tf.int64)
    # using session to init the global variables
    sess.run(tf.global_variables_initializer())
    return random_tensor
with tf.Session() as sess:

    # using tf.variable_scope to allow get_variable getting more
    with tf.variable_scope('foo'):
        random_output = sess.run(gen_random_tensor(sess, [2, 2, 2], 6, 10))
        print random_output
    with tf.variable_scope('foo1'):
        random_output = sess.run(gen_random_tensor(sess, [1, 2, 3], 1, 5))
        print random_output
        # outputs
        # [[[8 9]
        #   [8 7]]

        #  [[8 8]
        #   [6 9]]]


        # [[[2 2 3]
        #   [3 2 3]]]
