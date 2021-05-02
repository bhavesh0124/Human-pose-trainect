# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import json
import pickle
from io import StringIO
import signal
import flask
from io import BytesIO

import sys
import cv2
import numpy as np
import traceback
import darknet.python.darknet as dn
import keras
import os
from src.label                 import Label, lwrite, Shape, writeShapes, dknet_label_conversion, lread, readShapes
from os.path                 import splitext, basename, isdir, isfile
from os                     import makedirs
from src.utils                 import crop_region, image_files_from_folder, im2single, nms
from darknet.python.darknet import detect
from glob                         import glob
from src.keras_utils             import load_model, detect_lp
from src.drawing_utils            import draw_label, draw_losangle, write2img
from pdb import set_trace as pause
import time

input_dir  = "input_data"
output_dir = "output"

vehicle_weights = 'data/vehicle-detector/yolo-voc.weights'
vehicle_netcfg  = 'data/vehicle-detector/yolo-voc.cfg'
vehicle_dataset = 'data/vehicle-detector/voc.data'

vehicle_net  = dn.load_net(vehicle_netcfg, vehicle_weights, 0)
vehicle_meta = dn.load_meta(vehicle_dataset)

wpod_net_path = "data/lp-detector/wpod-net_update1.h5"
wpod_net = load_model(wpod_net_path)

ocr_weights = 'data/ocr/ocr-net.weights'
ocr_netcfg  = 'data/ocr/ocr-net.cfg'
ocr_dataset = 'data/ocr/ocr-net.data'

ocr_net  = dn.load_net(ocr_netcfg, ocr_weights, 0)
ocr_meta = dn.load_meta(ocr_dataset)


def vehicle_detection():
    try:
    
        vehicle_threshold = .5

        imgs_paths = image_files_from_folder(input_dir)
        imgs_paths.sort()

        if not isdir(output_dir):
            makedirs(output_dir)

        print('Searching for vehicles using YOLO...')

        for i,img_path in enumerate(imgs_paths):

            print('\tScanning %s' % img_path)

            bname = basename(splitext(img_path)[0])

            R,_ = detect(vehicle_net, vehicle_meta, img_path ,thresh=vehicle_threshold)

            R = [r for r in R if r[0] in ['car','bus']]

            print ('\t\t%d cars found' % len(R))

            if len(R):

                Iorig = cv2.imread(img_path)
                WH = np.array(Iorig.shape[1::-1],dtype=float)
                Lcars = []

                for i,r in enumerate(R):

                    cx,cy,w,h = (np.array(r[2])/np.concatenate( (WH,WH) )).tolist()
                    tl = np.array([cx - w/2., cy - h/2.])
                    br = np.array([cx + w/2., cy + h/2.])
                    label = Label(0,tl,br)
                    Icar = crop_region(Iorig,label)

                    Lcars.append(label)

                    cv2.imwrite('%s/%s_%dcar.png' % (output_dir,bname,i),Icar)

                lwrite('%s/%s_cars.txt' % (output_dir,bname),Lcars)

    except:
        traceback.print_exc()
        sys.exit(1)


def adjust_pts(pts,lroi):
    return pts*lroi.wh().reshape((2,1)) + lroi.tl().reshape((2,1))

def license_plate_det():
    try:
        
        lp_threshold = .5

        imgs_paths = glob('%s/*car.png' % output_dir)

        print ('Searching for license plates using WPOD-NET')

        for i,img_path in enumerate(imgs_paths):

            print ('\t Processing %s' % img_path)

            bname = splitext(basename(img_path))[0]
            Ivehicle = cv2.imread(img_path)

            ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
            side  = int(ratio*288.)
            bound_dim = min(side + (side%(2**4)),608)
            print ("\t\tBound dim: %d, ratio: %f" % (bound_dim,ratio))

            Llp,LlpImgs,_ = detect_lp(wpod_net,im2single(Ivehicle),bound_dim,2**4,(240,80),lp_threshold)

            if len(LlpImgs):
                Ilp = LlpImgs[0]
                Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
                Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)

                s = Shape(Llp[0].pts)

                cv2.imwrite('%s/%s_lp.png' % (output_dir,bname),Ilp*255.)
                writeShapes('%s/%s_lp.txt' % (output_dir,bname),[s])

    except:
        traceback.print_exc()
        sys.exit(1)


def ocr():

    try:
        
        ocr_threshold = .4

        imgs_paths = sorted(glob('%s/*lp.png' % output_dir))

        print('Performing OCR...')

        identified = []

        for i,img_path in enumerate(imgs_paths):

            print('\tScanning %s' % img_path)

            bname = basename(splitext(img_path)[0])

            R,(width,height) = detect(ocr_net, ocr_meta, img_path ,thresh=ocr_threshold, nms=None)

            if len(R):

                L = dknet_label_conversion(R,width,height)
                L = nms(L,.45)

                L.sort(key=lambda x: x.tl()[0])
                lp_str = ''.join([chr(l.cl()) for l in L])

                with open('%s/%s_str.txt' % (output_dir,bname),'w') as f:
                    f.write(lp_str + '\n')

                # print '\t\tLP: %s' % lp_str
                print("\n\nImage Name: {}".format(bname.split('_')[0]))
                print("Identified License Plate: {}".format(lp_str))
                identified.append(lp_str)

            else:
                lp_str = "No characters found" 
                print ('No characters found')
                identified.append(lp_str)


        return identified
        
    except:
        traceback.print_exc()
        sys.exit(1)


# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""

    status = 200 
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None
    filename = "input_data/image1.jpeg"
    if flask.request.content_type == 'image/jpeg':
        data = flask.request.data
        nparr = np.fromstring(BytesIO(data).read(),np.uint8)
        image = cv2.imdecode(nparr, 1)
        cv2.imwrite(filename, image)
        # print(type(image))
                             
    else:
        return flask.Response(response='This predictor only supports image/jpeg!', status=415, mimetype='text/plain')


    print('\nEndpoint invoked')

    vehicle_detection()
    license_plate_det()
    prediction = ocr()
    cmd1 = "rm {}/*".format(output_dir)
    cmd2 = "rm {}/*".format(input_dir)

    os.system(cmd1)
    os.system(cmd2)

    print(prediction)
    result = " , ".join(prediction)

    return flask.Response(response=result, status=200, mimetype='text/csv')
