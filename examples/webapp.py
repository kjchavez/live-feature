import live_feature as lf
import logging

import my_live_features


def test_expand():
    x = {'id': "Q001", 'baz': 1.5}
    lf.expand(x, my_live_features, cache=None)
    print x
    assert 'foo' in x
    assert 'bar' in x

def test_batch():
    x = {'id': ["Q001", "Q002"], 'baz': [1.5, 2.0]}
    expander = lf.Expander(my_live_features, id_key='id')
    expander.apply(x)
    print(x)
    assert 'foo' in x
    assert 'bar' in x
    assert isinstance(x['foo'], list)
    assert isinstance(x['bar'], list)

def test_single():
    x = {'id': "Q001", 'baz': 1.5}
    expander = lf.Expander(my_live_features, id_key='id')
    expander.apply(x)
    print(x)
    assert 'foo' in x
    assert 'bar' in x

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_batch()
    test_single()
