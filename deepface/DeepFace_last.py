import warnings
warnings.filterwarnings("ignore")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import numpy as np
from tqdm import tqdm
import cv2
from deepface.basemodels import Facenet
from deepface.commons import functions, realtime, distance as dst

import tensorflow as tf
tf_version = int(tf.__version__.split(".")[0])
if tf_version == 2:
	import logging
	tf.get_logger().setLevel(logging.ERROR)

def build_model(model_name):

	"""
	This function builds a deepface model
	Parameters:
		model_name (string): face recognition or facial attribute model
			Facenet
	Returns:
		built deepface model
	"""

	global model_obj #singleton design pattern

	models = {
		'Facenet': Facenet.loadModel,
	}

	if not "model_obj" in globals():
		model_obj = {}

	if not model_name in model_obj.keys():
		model = models.get(model_name)
		if model:
			model = model()
			model_obj[model_name] = model
		else:
			raise ValueError('Invalid model_name passed - {}'.format(model_name))

	return model_obj[model_name]


def verify(img1_path, img2_path = '', model_name = None, distance_metric = "euclidean", model = None, enforce_detection = True, detector_backend = 'opencv', align = True, prog_bar = True, normalization = 'base'):

	"""
	This function verifies an image pair is same person or different persons.

	Parameters:
		img1_path, img2_path: exact image path, numpy array (BGR) or based64 encoded images could be passed. If you are going to call verify function for a list of image pairs, then you should pass an array instead of calling the function in for loops.

		e.g. img1_path = [
			['img1.jpg', 'img2.jpg'],
			['img2.jpg', 'img3.jpg']
		]

		model_name (string): Facenet

		distance_metric (string): cosine, euclidean, euclidean_l2

		enforce_detection (boolean): If any face could not be detected in an image, then verify function will return exception. Set this to False not to have this exception. This might be convenient for low resolution images.

		detector_backend (string): retinaface

		prog_bar (boolean): enable/disable a progress bar

	Returns:
		Verify function returns a dictionary


	"""

	#img1_path = cv2.imread(img1_path)


	img_list, bulkProcess = functions.initialize_input(img1_path, img2_path)

	resp_objects = []

	#--------------------------------

	if model_name == 'Ensemble':
		model_names = ["Facenet"]
		metrics = ["cosine", "euclidean", "euclidean_l2"]
	else:
		model_names = []; metrics = []
		model_names.append(model_name)
		metrics.append(distance_metric)

	#--------------------------------

	if model == None:
		if model_name == 'Ensemble':
			models = Boosting.loadModel()
		else:
			model = build_model(model_name)
			models = {}
			models[model_name] = model
	else:
		if model_name == 'Ensemble':
			Boosting.validate_model(model)
			models = model.copy()
		else:
			models = {}
			models[model_name] = model


	#------------------------------

	disable_option = (False if len(img_list) > 1 else True) or not prog_bar

	pbar = tqdm(range(0,len(img_list)), desc='Verification', disable = disable_option)

	for index in pbar:

		instance = img_list[index]

		if type(instance) == list and len(instance) >= 2:
			img1_path = instance[0]; img2_path = instance[1]

			ensemble_features = []

			for i in  model_names:

				custom_model = models[i]

				# img_path, model_name = 'VGG-Face', model = None, enforce_detection = True, detector_backend = 'mtcnn'
				img1_representation = represent(img_path = img1_path
						, model_name = model_name, model = custom_model
						, enforce_detection = enforce_detection, detector_backend = detector_backend
						, align = align
						, normalization = normalization
						)

				img2_representation = represent(img_path = img2_path
						, model_name = model_name, model = custom_model
						, enforce_detection = enforce_detection, detector_backend = detector_backend
						, align = align
						, normalization = normalization
						)
				cosine_distance = dst.findCosineDistance(img1_representation, img2_representation)

				euclidean_distance = dst.findEuclideanDistance(img1_representation, img2_representation)

				euclidean_l2_distance = dst.findEuclideanDistance(dst.l2_normalize(img1_representation), dst.l2_normalize(img2_representation))

				resp_obj = {
					"img1_path": img1_path.split("\\")[-1]
					, "img2_path": img2_path.split("\\")[-1]
					, "cosine_distance": cosine_distance
					, "euclidean_distance": euclidean_distance
					, "euclidean_l2_distance": euclidean_l2_distance
					, "normalization": normalization
				}
	return resp_obj

def represent(img_path, model_name = None, model = None, enforce_detection = True, detector_backend = 'retinaface', align = True, normalization = None):

	"""
	This function represents facial images as vectors.

	Parameters:
		img_path: exact image path, numpy array (BGR) or based64 encoded images could be passed.

		model_name (string):Facenet

		enforce_detection (boolean): If any face could not be detected in an image, then verify function will return exception. Set this to False not to have this exception. This might be convenient for low resolution images.

		detector_backend (string): retinaface

		normalization (string): normalize the input image before feeding to model

	Returns:
		Represent function returns a multidimensional vector. The number of dimensions is changing based on the reference model. E.g. FaceNet returns 128 dimensional vector; VGG-Face returns 2622 dimensional vector.
	"""

	if model is None:
		model = build_model(model_name)

	#---------------------------------

	#decide input shape
	input_shape_x, input_shape_y = functions.find_input_shape(model)

	#detect and align
	img = functions.preprocess_face(img = img_path
		, target_size=(input_shape_y, input_shape_x)
		, enforce_detection = enforce_detection
		, detector_backend = detector_backend
		, align = align)

	#---------------------------------

	img = functions.normalize_input(img = img, normalization = normalization)

	#---------------------------------

	#represent
	if "keras" in str(type(model)):
		embedding = model.predict(img, verbose=0)[0].tolist()
	else:
		embedding = model.predict(img)[0].tolist()

	return embedding

