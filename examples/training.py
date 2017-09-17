from __future__ import print_function

import tensorflow as tf
import live_features_tf as lftf
import my_live_features

x = {}
x['id'] = tf.constant(["Q1"])
x['baz'] = tf.constant([1.5])
lftf.expand(x, my_live_features, cache=None)

with tf.Session() as sess:
    x_val = sess.run(x)
    print(x_val)
