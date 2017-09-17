from __future__ import print_function

import random
import tensorflow as tf
import live_feature_tf as lftf
import my_live_features

x = {}
x['id'] = tf.constant(["Q%04d" % i for i in xrange(100)])
x['baz'] = tf.constant([random.random() for _ in xrange(100)])

filenames = tf.placeholder(tf.string, shape=[None])
dataset = tf.contrib.data.Dataset.from_tensor_slices(x)

expander = lftf.Expander('id', my_live_features, cache_fn=None)
dataset = expander.transform(dataset).batch(8)
iterator = dataset.make_one_shot_iterator()
next_elem = iterator.get_next()

with tf.Session() as sess:
    while True:
        try:
            x_val = sess.run(next_elem)
            print(x_val)
        except tf.errors.OutOfRangeError:
            break
