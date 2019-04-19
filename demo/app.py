import os
import sys
import time
sys.path.insert(0,'..')

from pathlib import Path
import numpy as np
import json
from shutil import copyfile, copytree, rmtree
from werkzeug.utils import secure_filename
from flask import Flask, request, redirect, url_for, render_template, send_from_directory, jsonify
from PIL import Image

import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.image as mpimg


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploaded_img'
app._static_folder = os.path.abspath("static")

# TODO: load model here
_MODEL_PATH = '../experiments/VGG16_based_classification/vgg16_catdog.tflite'
_LABEL = ['Cat','Dog']

interpreter = tf.lite.Interpreter(model_path=_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(path, size):
    """
    This function preprocess test image for prediction
    Arguments:
    -path: 
    -size:
    Returns:
    
    """
    # Read image from image path
    image_decoded = mpimg.imread(path)
    image = tf.image.convert_image_dtype(image_decoded, tf.float32) # convert to float values in [0, 1]
    # resize image to fit the input size
    resized_image = tf.image.resize_with_pad(image, size, size)
    # Scale to [0,255]
    resized_image = resized_image*255
    # Transpote from RGB to BGR then subtract for mean(imagenet)
    resized_image = resized_image[...,::-1] - [103.939,116.779,123.68]
    # Reshape tensor
    resized_image = tf.reshape(resized_image,[1,224,224,3])
    return resized_image


@app.route('/static/<path:filename>')
def serve_static(filename):
	'''
	serve static file for this project
	Args:
	- filename: string, the file request get from url
	Another information can get from global variables: request
	Return:
	- static content (js, image, html, css, etc....)
	Raise: None
	'''
	root_dir = app.root_path
	return send_from_directory(os.path.join(root_dir, 'static'), filename)

@app.route('/', methods=['GET', 'POST'])
def home_page():
	'''
	this function return response for any request from home page
	including POST and GET
	Args: None
	Return:
	- html: Response object, contain text value inside
	Raise: None
	'''
	return render_template('home.htm')


@app.route('/video', methods=['GET', 'POST'])
def videoRTC_page():
	'''
	this function return response for any request from video page
	including POST and GET
	Args: None
	Return:
	- html: Response object, contain text value inside
	Raise: None
	'''
	return render_template('video.htm')

@app.route('/upload', methods=['GET', 'POST'])
def upload_page():
	'''
	this function return response for any request from video page
	including POST and GET
	Args: None
	Return:
	- html: Response object, contain text value inside
	Raise: None
	'''
	result = {
	"image": "",
	"class": -1,
	"confidence": 0,
	"eslapse": 0
	}
	return render_template('upload.html', result=result)

@app.route('/history', methods=['GET', 'POST'])
def history_page():
	'''
	return the predict history, including image url, predict result and percent of confident
	Args: None
	the information of request can be access using global variable: request
	Return:
	- response: Response object, container of text
	Raise: None
	'''
	listfile = sorted(os.listdir(app.config["UPLOAD_FOLDER"]), reverse=True)
	listfile = list(filter(lambda x: '_predictted_' in x, listfile))
	data = []
	bg = 'http://1266b9k5nh047m4qk1509101-wpengine.netdna-ssl.com/wp-content/uploads/2017/02/dogcatcare1.jpg'
	for filename in listfile:
		sp = filename.split('_')
		if len(sp) < 6:
			continue
		predict = int(sp[2])
		confident = float(sp[3])
		eslape_time = float(sp[4])
		dic = {
			'img': url_for('static', filename='uploaded_img/'+filename),
			'predict': predict,
			'confident': confident,
			'eslape': eslape_time
		}
		data.append(dic)
		# if len(listfile) != 0:
		# 	bg = url_for(app.config['UPLOAD_FOLDER'], filename=listfile[0])
	return render_template('history.html', data=data, bg=bg)


def inference():
	result = {
	"image": "",
	"class": -1,
	"confidence": 0,
	"eslapse": 0
	}
	
	 # check if the post request has the file part
	if 'pic' not in request.files:
		return jsonify({
			'error': -1,
			'msg': 'no image to process'
		})
	file = request.files['pic']
	# if user does not select file, browser also
	# submit an empty part without filename
	if file.filename == '':
		return jsonify({
			'error': -2,
			'msg': 'image have no filename'
		})
	if file:
		# save file to upload folder
		start_time = time.time()
		folder = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'])
		filename = secure_filename(file.filename)
		filename = filename + str(start_time) + '.jpg'
		fullpath = os.path.join(folder, filename) 
		file.save(fullpath)
		print(type(file))

		print(filename)
		#load file and pre-processing image
		size = 64
		new_path = os.path.join(folder, 'resized_' + filename)
		print('new image will save at: ' + new_path)

		# TODO: pre-process image here
		#Image.open(fullpath).resize((size, size)).convert('L').save(new_path)

		# Read image to array (height,weight,3)
		# Preprocess image 
		test_image = preprocess_image(path=fullpath, size=224)		

		# Run model with test_image as input image
		interpreter.set_tensor(input_details[0]['index'],test_image)
		interpreter.invoke()
		# Predict 
		result_ = interpreter.get_tensor(output_details[0]['index'])

		class_ = result_.argmax()	# class of object from image
		confidence = result_.max() # confidence score
		# TODO predict
		# class_, confident = model.predict(X)

		eslape_time = (time.time() - start_time) * 1000

		#add predict result to render dict
		result["class"] = int(class_)
		result["confidence"] = confidence*100
		result["eslapse"] = eslape_time

		# save image as formated
		millis = int(round(start_time * 1000))
		new_filename = "{2}_predictted_{0}_{1}_{3}_".format(class_, confidence*100, millis, eslape_time) + filename
		new_path = os.path.join(folder, new_filename)
		os.rename(fullpath, new_path)

		#add image to render dict
		result["image"] = url_for('static', filename='uploaded_img/' + new_filename)

	return result

@app.route('/predict', methods=['POST'])
def predict_image():
	'''
	handle upload image POST request,
	1. check upload file existed
	2. save to upload folder
	3. gray scale and make input for model prediction
	4. prediction
	5. return to user the result of predict
	Args: None
	- file uploaded can be check in global variables: request
	Return:
	- response: Response object, contain text value inside
	Raise: None
	'''
	result = inference()
	#send response
	return jsonify({
		'error': 0,
		'data': result
	})

@app.route('/predict_upload', methods=['POST'])
def predict_uploaded_image():
	'''
	handle upload image POST request,
	1. check upload file existed
	2. save to upload folder
	3. gray scale and make input for model prediction
	4. prediction
	5. return to user the result of predict
	Args: None
	- file uploaded can be check in global variables: request
	Return:
	- response: Response object, contain text value inside
	Raise: None
	'''
	result = inference()
	#send response
	return render_template('upload.html', result=result)


if __name__ == "__main__":
	'''
	main function if you run this script from terminal
	We suggest that you should start server using "flask run" command instead this way
	'''
	app.run()
