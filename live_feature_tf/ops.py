import tensorflow as tf
import numpy as np
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

    def _apply(self, tensor_x, batched=True):
        def _apply_ordered(x):
            # This avoids fetching LiveFeatures that have already been folded into the
            # example source.
            feat = {key: None for key in tensor_x.keys()}
            # Handle the batched case.
            if batched:
                feat[self._id_key] = [i for i in x]
            else:
                feat[self._id_key] = x
            self.expander.apply(feat)
            return tuple(feat[key] for key in self.ordered_keys)

        dtypes = [tf_dtype(self.expander.live_features[key].feature_def.dtype) for key in
                  self.ordered_keys]
        expanded_dict = dict(tensor_x)
        expanded_dict.update(zip(self.ordered_keys, tf.py_func(_apply_ordered,
                                                               [tensor_x[self._id_key]],
                                                               dtypes,
                                                              stateful=False)))

        # We also need to set the appropriate shapes for these elements of the expanded_dict.
        # Going through a py_func makes it impossible to infer the shape from the graph.
        # The first dim is typically the batch size -- so it should just carry over.
        # For the other dims, technically, it can be arbitrary, so it should be defined by the
        # LiveFeatureDef.
        for key in self.expander.live_features.keys():
            if key in expanded_dict:
                shape = self.expander.live_features[key].feature_def.output_shape
                if batched:
                    shape = (tensor_x[self._id_key].shape[0],) + shape

                expanded_dict[key].set_shape(shape)

        return expanded_dict

    def transform(self, dataset, batched=True):
        """ Returns a dataset expanded with live features.
        Args:
            dataset: tf.contrib.data.Dataset object
            batched: if dataset is already batched

        Returns:
            Dataset object whose elements are augmented with features from this Expander.
        """
        return dataset.map(lambda x: self._apply(x, batched=batched))
