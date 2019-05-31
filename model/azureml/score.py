import numpy as np
import os
import pickle
import json
from keras.models import model_from_json
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform

from azureml.core.model import Model


def init():
    global model
    global index_map

    model_root = Model.get_model_path('keras_bf')
    with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        json_file = open(os.path.join(model_root, 'model.json'), 'r')
        model_json = json_file.read()
        json_file.close()
        model = model_from_json(model_json)
        model.load_weights(os.path.join(model_root, "model.h5"))
        #model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

    with open(os.path.join(model_root, "index_map.pkl"), 'rb') as f:
        index_map = pickle.load(f)


def run(raw_data):
    raw_data_dict = json.loads(raw_data)
    test_data = [[index_map[key][value]] for key, value in raw_data_dict.items()]
    return int(np.squeeze(model.predict(test_data)))
