# End-to-end Example

## Train, eval, and export model.

```
python train.py
```

This will create a SavedModel in `/tmp/exportdir`.
You can inspect its signature with:

```bash
```

## Upload to Cloud ML Engine

First:

* Create a Cloud Platform project
* Create a bucket in Google Cloud Storage
* Create a new model in Cloud ML Engine

You'll have to modify some parameters in `upload_model.sh`, then run it.


## Test

Start up the webapp. Host defaults to localhost. 0.0.0.0 listens on all
available interfaces.

```
export FLASK_APP=webapp.py
flask run --host=0.0.0.0 --port=5000
```

In the browser, go to `localhost:5000/test?id=Q001`

The webapp runs the LiveFeature Expander to get `foo` and `bar` features, then
sends the full example to the model via `cloudml_client.predict_json`.

You should get back a predicted (albeit senseless) value for `baz_hat`.
