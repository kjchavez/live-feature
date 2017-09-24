import os
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'  # Tell tensorflow to quiet down.
import logging

from livefeature import cache_util
from livefeature import feature_def

import fake_feature_module

def test_smoke():
    os.remove("/tmp/testcachedir/fake1")
    cache_util.create_caches_from_tfrecord(feature_def.get_feature_defs(fake_feature_module),
                                           "../../legislation-project/vote_prediction/mini-data/test.tfrecord",
                                           "/tmp/testcachedir")

if __name__ == "__main__":
    test_smoke()
