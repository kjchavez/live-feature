from live_feature import live_feature
import string

@live_feature("foo", int)
def get_foo(idx):
    return hash(idx) % 1000

@live_feature("bar", str)
def get_bar(idx):
    time.sleep(0.100)  # Pretend this is expensive.
    char_index = hash(idx) % 26
    return string.ascii_uppercase[char_index]
