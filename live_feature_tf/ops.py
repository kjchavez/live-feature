import tensorflow as tf
import live_feature as lf

def tf_dtype(pytype):
    if pytype == str:
        return tf.string
    elif pytype == float:
        return tf.float
    elif pytype == int:
        return tf.int64


class Expander(object):
    def __init__(self, id_key, live_feature_module, cache_fn=None):
        self.expander = lf.Expander(live_feature_module, id_key=id_key, cache_fn=cache_fn)
        self.ordered_keys = self.expander.live_features.keys()
        self._id_key = id_key

    def _apply(self, tensor_x):
        def _apply_ordered(x):
            # This avoids fetching LiveFeatures that have already been folded into the
            # example source.
            feat = {key: None for key in tensor_x.keys()}
            feat[self._id_key] = x
            self.expander.apply(feat)
            return tuple(feat[key] for key in self.ordered_keys)

        dtypes = [tf_dtype(self.expander.live_features[key].feature_def.dtype) for key in
                  self.ordered_keys]
        expanded_dict = dict(tensor_x)
        expanded_dict.update(zip(self.ordered_keys, tf.py_func(_apply_ordered,
                                                               [tensor_x[self._id_key]],
                                                               dtypes)))
        return expanded_dict

    def transform(self, dataset):
        """ Returns a dataset expanded with live features. """
        return dataset.map(self._apply)
