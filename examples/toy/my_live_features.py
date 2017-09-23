import livefeature as lf
import logging
import string
import time

@lf.feature("foo", int)
def get_foo(idx):
    logging.debug("Received idx: %s", idx)
    return hash(idx) % 1000

@lf.feature("bar", str)
def get_bar(idx):
    logging.debug("Received idx: %s", idx)
    time.sleep(0.100)  # Pretend this is expensive.
    char_index = hash(idx) % 26
    return string.ascii_uppercase[char_index]
