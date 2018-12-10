from __future__ import division, print_function
# coding=utf-8
import sys
import os
import numpy as np
#import base64

# PIL
from PIL import Image 

# Skimage
from skimage.morphology import binary_opening, disk
from skimage.io import imsave
from skimage import img_as_uint,io
from skimage.measure import label, regionprops

# Keras
from keras.models import load_model

# Flask utils
from flask import Flask, redirect, url_for, request, render_template, make_response,send_file
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# OpenCV
import cv2



# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
model_path = './models/fullres_model_v3.h5'
global img_stream



model = load_model(model_path)
print('Model loaded. Check http://localhost:5000/')

def predictraw(image_path,model_path):
    model = load_model(model_path)
    c_img = Image.open(image_path)
    c_img = np.expand_dims(c_img, 0)/255.0
    cur_seg = model.predict(c_img)[0]
    return cur_seg
def smooth(pred_raw):
    return binary_opening(pred_raw>0.99, np.expand_dims(disk(2), -1))
def predict(image_path,model_path):
    predictimage = predictraw(image_path,model_path)
    return smooth(predictimage)
    
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    global file1_path
    if request.method == 'POST':
        # Get the file from post request
        file1_path = ''
        f = request.files['file']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        preds = predict(file_path,model_path)
        file1_path = os.path.join(basepath, 'results', secure_filename(f.filename))    
        img_mask = img_as_uint(preds[:,:,0])

        #for bounding box
        #img_0 = cv2.imread(file_path)
        lable_mask = label(img_mask)
        props = regionprops(lable_mask)
        img_out = cv2.imread(file_path)

        for prop in props:
            cv2.rectangle(img_out,(prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), (77, 255, 9), 2)

        #io.imshow(img_stream)
        #print(type(img_stream))
        imsave(file1_path,img_out)
        #result = "Successed! "
        #img_stream = base64.b64encode(img_as_uint(preds[:,:,0]))
        return file1_path
        
    if request.method =='GET':

        img_stream = open(file1_path,'rb').read()
        response = make_response(img_stream)
        response.headers['Content-Type'] = 'image/jpg'
        return response



############################################################################

@app.route('/unet',methods = ['GET'])
def unet():
    return send_file('unet.html')





@app.route('/favicon.ico',methods =['GET'])
def ico():
    ico_path = './ship.ico'
    ico_data = open(ico_path,'rb').read()
    response = make_response(ico_data)
    response.headers['Content-Type'] = 'image/ico'
    return response


@app.route('/header.jpg',methods =['GET'])
def header():
    header_path = './header.jpg'
    header_data = open(header_path,'rb').read()
    #print(type(header_data))
    response = make_response(header_data)
    response.headers['Content-Type'] = 'image/jpg'
    return response




if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    port = int(os.environ.get('PORT', 5000)) 
    http_server = WSGIServer(('0.0.0.0', port), app)
    http_server.serve_forever()
