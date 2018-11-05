from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

#PIL
from PIL import Image 

from skimage.morphology import binary_opening, disk
from skimage.io import imsave
from skimage import img_as_uint

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

file1_path = ""

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
model_path = '/Users/zifanwang/Documents/keras-flask-deploy-webapp/models/fullres_model.h5'

# Load your trained model
# model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
model = load_model(model_path)
print('Model loaded. Check http://127.0.0.1:5000/')

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
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        preds = predict(file_path,model_path)

        file1_path = os.path.join(
            basepath, 'results', secure_filename(f.filename))
        
        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        imsave(file1_path,img_as_uint(preds[:,:,0]))
        result = "Successed! "
        return result

    if request.method =="POST":
        
    

    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
