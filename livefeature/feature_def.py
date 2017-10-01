from livefeature import cache
from multiprocessing.dummy import Pool
import inspect
import logging

# Decorator to create LiveFeatureDefs out of standalone functions.
# Note, by default we assume that the feature outputs a single value for each transformation.
# However, if instead, the function produces, say a 32 x 32 black and white image,
# the decorator should set the output shape. E.g.
#
# @lf.feature("selfie", float, key="id_key", shape=(32,32))
# @lf.feature("selfie", key_fn=lf.key_fn("id_key"), float, shape=(32,32))
#
# General overview.
#
# There is an example, X. The typical case is that this is a dictionary of features.
# We generate a key, K, from X using the |key_fn|. The typical case is simply a lookup of a single
# dictionary entry.
# That key is passed to the function that was annotated with @lf.feature and should be used to
# retrieve the new feature. It will be used to create cache entries as well -- so a key should
# always map to the same resource (for a fixed time), and ideally, different keys actually point to
# different resources (not as important, but helpful for space efficiency).
class feature(object):
    def __init__(self, name, dtype, key=None, key_fn=None, shape=()):
        if not key and not key_fn:
            raise ValueError("Must specify one of |key| or |key_fn|.")
        if key is not None:
            def _key_fn(x):
                return x.get('key_id', "")

            self.key_fn = lambda x: x.get(key, "")[0]
        else:
            self.key_fn = key_fn

        self.name = name
        self.shape = shape
        self.dtype = dtype

    def __call__(self, f):
        f.__livefeature__feature_def = LiveFeatureDef(self.name, self.key_fn, f, self.shape, self.dtype)
        return f

def get_feature_defs(python_module):
    """ Returns all LiveFeatureDefs for functions defined in |python_module|. """
    defs = []
    for fn_name, fn in inspect.getmembers(python_module, inspect.isfunction):
        if hasattr(fn, "_feature__livefeature__feature_def"):
            defs.append(fn._feature__livefeature__feature_def)

    return defs

class LiveFeatureDef(object):
    def __init__(self, name, key_func, func, shape, dtype):
        self.name = name
        self.key_func = key_func
        self.func = func
        self.dtype = dtype
        self.output_shape = shape

    def __repr__(self):
        return "LiveFeatureDef(name=%s,func=%s,shape=%s,dtype=%s)" % (self.name, self.func.__name__,
                                                                      self.output_shape, self.dtype)

class LiveFeature(object):
    def __init__(self, feature_def, cache=None, num_workers=8):
        self.feature_def = feature_def
        if cache is None:
            self.cache = cache.PassthroughCache(feature_def)
        else:
            self.cache = cache

        self.pool = Pool(num_workers)

    def get(self, x):
        """ Gets the LiveFeature value for a single example |x|. """
        key = self.feature_def.key_func(x)
        return self.cache.get(key)

    def get_batch(self, batch):
        feature_batch = self.pool.map(self.get, [x for x in batch])
        return feature_batch


class Expander(object):
    def __init__(self, feature_module, create_cache=cache.PassthroughCache,
                 num_workers=10):
        self.live_features = {}
        feature_defs = get_feature_defs(feature_module)
        for f in feature_defs:
            _cache = create_cache(f)
            self.live_features[f.name] = LiveFeature(f, cache=_cache)
            logging.info("Using %s", f)

        self.pool = Pool(num_workers)

    def apply_batch(self, x_batch):
        for feature in self.live_features.keys():
            if feature not in x:
                x[feature] = self.live_features[feature].get_batch(x_batch)

    def apply(self, x):
        """ Expands example |x| with live features. """
        # TODO(kjchavez): This is another bottleneck here. There's no point in making all these
        # lookups sequentially.
        for feature in self.live_features.keys():
            if feature not in x:
                x[feature] = self.live_features[feature].get(x)

