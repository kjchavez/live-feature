from flask import Flask
from flask import jsonify
from flask import request
import logging
import os
import yaml

import cloudml_client
import my_live_features
import livefeature as lf


app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.debug = True

expander = lf.Expander(my_live_features, id_key='id')

@app.route("/")
def home():
    return "Running!"

@app.route("/test")
def test():
    x = dict(request.args)
    if 'id' not in x:
        return "Must provide ID!"
    expander.apply(x)
    result = cloudml_client.predict_json("tf-livefeature", "test_prediction", [x])
    return jsonify(result)
