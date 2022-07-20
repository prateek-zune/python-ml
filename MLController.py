import string

import requests
from flask import Flask, jsonify, request
import sys, os
from math import exp
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QFont
import tensorflow as tf
import numpy as np
import json

app = Flask(__name__)


def read_tensor_from_image_url(url,
                               input_height=299,
                               input_width=299,
                               input_mean=0,
                               input_std=255):
    image_reader = tf.image.decode_jpeg(
        requests.get(url).content, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])

    with tf.Session() as sess:
        return sess.run(normalized)


@app.route('/image/')
def get_cargo():
    image = tf.image.decode_jpeg(
        requests.get(request.args.get('file_path')).content, channels=3,
        name="jpeg_reader")
    image_nose = tf.expand_dims(image, 0)
    image_nose = tf.image.crop_and_resize(image_nose, [[0, 0, 0.2, 0.25]], [0], [224, 224])
    image_nose = tf.squeeze(image_nose)
    image_nose_ds = tf.data.Dataset.from_tensor_slices([image_nose])
    image_nose_ds = image_nose_ds.batch(1)
    image_nose_ds = image_nose_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    pred_nose = model_nose.predict(image_nose_ds)

    logit_nose = pred_nose[0][0]
    odds_nose = exp(logit_nose)
    prob_nose = odds_nose / (1 + odds_nose)

    image_load = tf.expand_dims(image, 0)
    image_load = tf.image.crop_and_resize(image_load, [[0, 0, 1.0, 1.0]], [0], [224, 224])
    image_load = tf.squeeze(image_load)
    image_load_ds = tf.data.Dataset.from_tensor_slices([image_load])
    image_load_ds = image_load_ds.batch(1)
    image_load_ds = image_load_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    pred_load = model_full.predict(image_load_ds)
    p0 = pred_load[0][0]
    p1 = pred_load[0][1]
    p2 = pred_load[0][2]
    p3 = pred_load[0][3]

    if logit_nose <= 0:
        prob_empty = 1 - prob_nose
        self.label.setText("{:s} - conf = {:d}%".format("EMPTY", round(100 * prob_empty)))
    #           if p0 >= p1 and p0 >= p2 and p0 >= p3:
    #               self.label.setText("{:s} - conf = {:d}%".format("EMPTY", round(100*prob_empty)))
    #           else:
    #               self.label.setText("{:s} - conf = {:d}%".format("Back LOADED", round(100*(1-p0))))
    else:
        pcnt = (0 * p0 + 40 * p1 + 68 * p2 + 100 * p3) / (p0 + p1 + p2 + p3)
        if pcnt < 25:
            text = "Barely LOADED"
        elif pcnt < 54:
            text = "Partially LOADED"
        elif pcnt < 90:
            text = "Mostly LOADED"
        else:
            text = "Fully LOADED"

        value = {
            "state": text,
            "prediction_value": "{:d}%".format(round(pcnt)),
            "confidence_rating": "{:d}%".format(round(100 * prob_nose))
        }

        return json.dumps(value)


model_nose = tf.keras.models.load_model("cargo-nose.mod")
model_full = tf.keras.models.load_model("cargo-full.mod")

if __name__ == "__main__":
    app.run(debug=True)
    get_cargo()
