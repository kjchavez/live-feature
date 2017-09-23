from livefeature import cache
from multiprocessing.dummy import Pool
import inspect
import logging

class LiveFeatureDef(object):
    all_instances = []
    def __init__(self, name, func, shape, dtype):
        self.name = name
        self.func = func
        self.dtype = dtype
        self.output_shape = shape
        LiveFeatureDef.all_instances.append(self)

    def __repr__(self):
        return "LiveFeatureDef(name=%s,func=%s,shape=%s,dtype=%s)" % (self.name, self.func.__name__,
                                                                      self.shape, self.dtype)

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
        feature_defs = LiveFeatureDef.all_instances
        for f in feature_defs:
            logging.debug("Checking feature def: %s", f)
            logging.debug("FN: %s", f.func)
            if f.func in fns:
                self.live_features[f.name] = LiveFeature(f, cache_fn=cache_fn)
                logging.info("Using %s", f)

    def apply(self, x):
        """ Expands example |x| with live features. """
        for key in self.live_features.keys():
            if key not in x:
                x[key] = self.live_features[key].get_batch(x[self.id_key])

