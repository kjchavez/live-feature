import cachetools

class MemCache(object):
    def __init__(self, name, func, max_size=1000):
        self.name = name
        self.cache = cachetools.LRUCache(max_size=max_size, missing=func)

    def get(self, example_id):
        return self.cache[example_id]

class PassthroughCache(object):
    def __init__(self, name, func):
        self.name = name
        self.func = func

    def get(self, example_id):
        return self.func(example_id)
