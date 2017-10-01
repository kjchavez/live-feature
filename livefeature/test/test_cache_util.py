import os
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'  # Tell tensorflow to quiet down.
import shutil
import logging

from livefeature import cache_util
from livefeature import cache
from livefeature import feature_def

import fake_feature_module

TEST_TFRECORD_FILENAME = os.path.join(os.path.dirname(__file__), 'test.tfrecord')

def test_smoke():
    if os.path.exists("/tmp/testcachedir"):
        shutil.rmtree("/tmp/testcachedir")
    cache_util.create_caches_from_tfrecord(feature_def.get_feature_defs(fake_feature_module),
                                           TEST_TFRECORD_FILENAME,
                                           "/tmp/testcachedir")

    cache1 = cache.PersistentCache("/tmp/testcachedir/foobar")
    cache2 = cache.PersistentCache("/tmp/testcachedir/baz")
    assert(cache1.get("democrat") == "foobar")
    assert(cache2.get("CA") == "baz")

if __name__ == "__main__":
    test_smoke()
