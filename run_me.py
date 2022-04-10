# from Detector import *
from charset_normalizer import detect
import cv2, time, os, tensorflow as tf
from cv2 import threshold
import numpy as np
from Detector import Detector

modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz"

imagePath = "test/2.png"
videoPath = "test/v1.mp4"
# videoPath = 0 #for webcam
threshold = 0.5
classFile = "coco.names"
detector = Detector()
detector.readClasses(classFile)

detector.downloadModel(modelURL)
detector.loadModel()
# detector.predictImage(imagePath, threshold)
detector.predictVideo(videoPath, threshold)