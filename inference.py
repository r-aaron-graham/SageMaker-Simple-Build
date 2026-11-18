"""Serve three-feature regression predictions using JSON requests."""
import json
from pathlib import Path

import joblib
import numpy as np


def model_fn(model_dir):
    return joblib.load(Path(model_dir) / 'model.joblib')


def input_fn(request_body, content_type):
    if content_type.split(';')[0].strip().lower() != 'application/json':
        raise ValueError('Expected application/json')
    values = np.asarray(json.loads(request_body), dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != 3:
        raise ValueError('Expected a nonempty array of three-feature rows')
    if not np.isfinite(values).all():
        raise ValueError('Features must be finite numbers')
    return values


def predict_fn(input_data, model):
    return model.predict(input_data)


def output_fn(prediction, accept):
    if accept.split(';')[0].strip().lower() != 'application/json':
        raise ValueError('Expected application/json response format')
    return json.dumps(prediction.tolist()), 'application/json'
