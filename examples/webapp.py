import live_feature as lf

import my_live_features


def test_expand():
    x = {'id': "Q001", 'baz': 1.5}
    lf.expand(x, my_live_features, cache=None)
    print x
    assert 'foo' in x
    assert 'bar' in x

if __name__ == "__main__":
    test_expand()
