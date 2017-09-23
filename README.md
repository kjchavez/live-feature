# LiveFeature

A lightweight framework for adding features fetched from the Web to TensorFlow models.
Ensures consistency between training and inference time.

This is the base package. Should be the only necessary dependency at serving time.
For the training time extras, use [`livefeature_tf`](https://github.com/kjchavez/live-feature-tf)

## Intended API

```python
# my_live_features.py
import livefeature as lf
@lf.feature(“wiki_summary”, str)
def fetch_wiki_summary(idx):
  # do stuff...
  return str(summary)


# in webapp.py
import livefeature as lf
import my_live_features

expander = lf.Expander(my_live_features, idd_key='id')
def predict(x):
  expander.apply(x)  # expands in-place.
  return cloudml_predict(x)
```
