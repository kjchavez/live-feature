from livefeature import cache
from multiprocessing.dummy import Pool
import inspect
import logging

# Decorator to create LiveFeatureDefs out of standalone functions.
# Note, by default we assume that the feature outputs a single value for each transformation.
# However, if instead, the function produces, say a 32 x 32 black and white image,
# the decorator should set the output shape. E.g.
#
# @lf.feature("selfie", int, shape=(32,32))
class feature(object):
    def __init__(self, name, dtype, shape=()):
        self.name = name
        self.shape = shape
        self.dtype = dtype

    def __call__(self, f):
        f.__livefeature__feature_def = LiveFeatureDef(self.name, f, self.shape, self.dtype)
        return f

def get_feature_defs(python_module):
    """ Returns all LiveFeatureDefs for functions defined in |python_module|. """
    defs = []
    for fn_name, fn in inspect.getmembers(python_module, inspect.isfunction):
        if hasattr(fn, "_feature__livefeature__feature_def"):
            defs.append(fn._feature__livefeature__feature_def)

    return defs

class LiveFeatureDef(object):
    def __init__(self, name, func, shape, dtype):
        self.name = name
        self.func = func
        self.dtype = dtype
        self.output_shape = shape

    def __repr__(self):
        return "LiveFeatureDef(name=%s,func=%s,shape=%s,dtype=%s)" % (self.name, self.func.__name__,
                                                                      self.output_shape, self.dtype)

class LiveFeature(object):
    def __init__(self, feature_def, num_workers=16, cache_fn=cache.PassthroughCache):
        self.feature_def = feature_def
        self.cache = cache_fn(feature_def.name, feature_def.func)
        self.pool = Pool(num_workers)

    def get_batch(self, batch):
        if not isinstance(batch, list):
            return self.cache.get(batch)

        feature_batch = self.pool.map(self.cache.get, [x for x in batch])
        return feature_batch


class Expander(object):
    def __init__(self, feature_module, cache_fn=None, id_key='id'):
        if cache_fn is None:
            cache_fn = cache.PassthroughCache

        self.id_key = id_key
        self.live_features = {}
        fns = set(zip(*inspect.getmembers(feature_module, inspect.isfunction))[1])
        logging.debug("All fns: %s", fns)
        feature_defs = get_feature_defs(feature_module)
        for f in feature_defs:
            self.live_features[f.name] = LiveFeature(f, cache_fn=cache_fn)
            logging.info("Using %s", f)

    def apply(self, x):
        """ Expands example |x| with live features. """
        for key in self.live_features.keys():
            if key not in x:
                x[key] = self.live_features[key].get_batch(x[self.id_key])

