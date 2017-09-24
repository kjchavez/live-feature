import cachetools
import json
import shelve
import os

class MemCache(object):
    def __init__(self, name, func, max_size=1000, load_json=None):
        self.name = name
        self.cache = cachetools.LRUCache(max_size, missing=func)
        if load_json:
            with open(load_json) as fp:
                data = json.load(fp)

            for key, value in data.items():
                self.cache[key] = value

    def get(self, example_id):
        return self.cache[example_id]

    def dump(self, filename):
        with open(filename, 'w') as fp:
            json.dump(dict(self.cache), fp)

class PassthroughCache(object):
    """ Always calls |func| to retrieve value. """
    def __init__(self, name, func):
        self.name = name
        self.func = func

    def get(self, example_id):
        return self.func(example_id)

class PersistentCache(object):
    def __init__(self, filename, func=lambda key: None):
        """ Cache  whose entries will be flushed to (and restored from) persistent memory.
        Args:
            Unique name that identifies this cache.
        """
        self.name = filename
        self.cache = shelve.open(filename)
        self.func = func

    def get(self, key):
        if not self.cache.has_key(key):
            value = self.func(key)
            if value is not None:
                self.cache[key] = value
            return value

        return self.cache[key]

    def __del__(self):
        self.cache.close()
