# LiveFeature

A lightweight framework for adding features fetched from the Web to TensorFlow models.
Ensures consistency between training and inference time.

## Intended API

```python
# my_live_features.py
@live_feature(“wiki_summary”, string)
def fetch_wiki_summary(idx):
  # do stuff…
  return str(summary)


# In tensorflow training file.
import my_live_features
import live_feature_tf as lftf
def input_fn():
  x = read_examples(...)
  lftf.expand(x, my_live_features, cache=lftf.FrozenCache(“/tmp/cache”))

# in webapp.py
import live_feature as lf
import my_live_features
def predict(x):
  lf.expand(x, my_live_features, cache=None)
```
