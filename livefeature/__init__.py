from livefeature.feature import LiveFeatureDef, LiveFeature, Expander
import cache

# Decorator to create LiveFeatureDefs out of standalone functions.
# Note, by default we assume that the feature outputs a single value for each transformation.
# However, if instead, the function produces, say a 32 x 32 black and white image,
# the decorator should set the output shape. E.g.
#
# @feature("selfie", int, shape=(32,32))
class feature(object):
    def __init__(self, name, dtype, shape=()):
        self.name = name
        self.shape = shape
        self.dtype = dtype

    def __call__(self, f):
        LiveFeatureDef(self.name, f, self.shape, self.dtype)
        return f
