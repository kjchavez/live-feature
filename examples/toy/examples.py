from __future__ import print_function

import livefeature as lf
import json

import my_live_features

def _add_to_file(x, outfile):
    expander = lf.Expander(my_live_features, id_key='id')
    expander.apply(x)
    with open(outfile, 'a') as fp:
        print(json.dumps(x), file=fp)

x = {'id': "Q001"}
labels = {'baz': 1.5}
_add_to_file(x, 'examples.json')
