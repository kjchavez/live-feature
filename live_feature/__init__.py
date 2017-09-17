from live_feature.feature import LiveFeatureDef, LiveFeature, Expander

# Decorator to create LiveFeatureDefs out of standalone functions.
class live_feature(object):
    def __init__(self, name, dtype):
        self.name = name
        self.dtype = dtype

    def __call__(self, f):
        LiveFeatureDef(self.name, f, self.dtype)
        return f
