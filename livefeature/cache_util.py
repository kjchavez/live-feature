"""
    Commandline utility for creating persistent caches from TFRecords.
"""
import os
import itertools
import logging
import tensorflow as tf
import threading

from livefeature import cache
from livefeature import feature_def

def _to_pyvalue(feature_proto):
    if feature_proto.HasField('bytes_list'):
        return [x for x in feature_proto.bytes_list.value]

    if feature_proto.HasField('int64_list'):
        return [x for x in feature_proto.int64_list.value]

    if feature_proto.HasField('float_list'):
        return [x for x in feature_proto.float_list.value]

def _read_features(features_proto):
    """ Returns python dict with loaded values from feature proto. """
    x = {}
    for name in features_proto.feature:
        x[name] = _to_pyvalue(features_proto.feature[name])
    return x

def _take(iterable, k):
    return list(itertools.islice(iterable, k))

def _parse_and_convert(serialized_example):
    example = tf.train.Example()
    example.ParseFromString(serialized_example)
    return _read_features(example.features)

def _streamed(tfrecord_filepattern):
    iterator = tf.python_io.tf_record_iterator(tfrecord_filepattern)
    return itertools.imap(_parse_and_convert, iterator)

def _batched(tfrecord_filepattern, batch_size=64):
    while True:
        batch = _take(_streamed(tfrecord_filepattern, batch_size))
        if not batch:
            return
        yield batch

def _create_single_cache(feature_def, tfrecord_filepattern, cachedir):
    key_fn = feature_def.key_func
    value_fn = feature_def.func
    new_cache = cache.PersistentCache(os.path.join(cachedir, feature_def.name), func=value_fn)
    for x in _streamed(tfrecord_filepattern):
        try:
            key = key_fn(x)
        except:
            logging.warning("Failed to get key from example: %s", x)
            continue

        new_cache.get(key)

def create_caches_from_tfrecord(feature_defs, tfrecord_filepattern, cachedir):
    if not os.path.exists(cachedir):
        os.makedirs(cachedir)

    threads = []
    for feature_def in feature_defs:
        t = threading.Thread(target=_create_single_cache, args=(feature_def, tfrecord_filepattern,
                                                             cachedir))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

""" Sample example.proto
features {
  feature {
    key: "BillTitle"
    value {
      int64_list {
        value: 8
        value: 29
        value: 2
        value: 7
        value: 5
        value: 6
        value: 1
        value: 2
        value: 7
        value: 5
        value: 27
        value: 6
        value: 4
        value: 28
      }
    }
  }
  feature {
    key: "Decision"
    value {
      bytes_list {
        value: "Aye"
      }
    }
  }
}
"""
