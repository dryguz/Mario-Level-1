def prepare_network():
	# USAGE
	# python neural_style_transfer_video.py --models models

	# import the necessary packages
	from imutils.video import VideoStream
	from imutils import paths
	import itertools

	import imutils
	import time
	import cv2

	# construct the argument parser and parse the arguments
	args = {}
	args["models"] = "models"

	models = 'models'
	# grab the paths to all neural style transfer models in our 'models'
	# directory, provided all models end with the '.t7' file extension
	modelPaths = paths.list_files(args["models"], validExts=(".t7",))
	#modelPaths = paths.list_files(models, validExts=(".t7",))
	modelPaths = sorted(list(modelPaths))

	# generate unique IDs for each of the model paths, then combine the
	# two lists together
	models = list(zip(range(0, len(modelPaths)), (modelPaths)))

	# use the cycle function of itertools that can loop over all model
	# paths, and then when the end is reached, restart again
	modelIter = itertools.cycle(models)
	(modelID, modelPath) = next(modelIter)

	# load the neural style transfer model from disk
	print("[INFO] loading style transfer model...")
	net = cv2.dnn.readNetFromTorch(modelPath)

	# initialize the video stream, then allow the camera sensor to warm up
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)
	print("[INFO] {}. {}".format(modelID + 1, modelPath))
	return vs, net


def destroy_network():
	cv2.destroyAllWindows()
	vs.stop()