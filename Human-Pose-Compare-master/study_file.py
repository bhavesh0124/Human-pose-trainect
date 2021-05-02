from __future__ import print_function

import os
import json
import glob
import time
import pickle
import io
import sys
import signal

import argparse
from calculations import get_Score

import shutil
import traceback
import flask
import pandas as pd
import tensorflow as tf
import Evaluate
import Models.UnetAudioSeparator
import numpy as np
import array
import base64
import zipfile

prefix = '/opt/ml/'


class ScoringService(object):
    model=None
    
    @classmethod
    def predict(video):

        lookup="lookup_test.pickle"
        activity="punch - side"
        g = get_Score(lookup) #intiliasing the score obeject and passing lookup.pickle as the input
        
        final_score,score_list = g.calculate_Score(video,activity)

        return(final_score,score_list)


app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])

def ping():
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])


def transformation():
    data = None
    
    if flask.request.content_type == 'application/x-recordio-protobuf':
        data = flask.request.data
        #input_path = "/tmp/audio_file_{}.mp3".format(time.time())
        #audio_file = open(input_path, 'wb')
        #audio_file.write(data)
        print("Input path :", input_path)

    else:
        return flask.Response(response='This predictor only supports audio data', status=415, mimetype='text/plain')

    # Do the prediction

    print('Endpint invoked')
    final_score,score_list = ScoringService.predict(data)
    out = StringIO()
    out.write(final_score,score_list)
    #result = out.getvalue()
    print(final_score,score_list)

    return flask.Response(response=final_score, status=200, mimetype='text/csv')

    return flask.Response(response=output_zipped, status=200, mimetype='application/x-recordio-protobuf')
