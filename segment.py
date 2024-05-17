# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 10:48:09 2019

@author: eduardo
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from os import listdir
from conv import Conv
from image import Image
from augment import Augment
from imutils import paths
image = Image()
augment = Augment()
conv = Conv(None, None, None, None)
batch_size = 1
stars = False
extended_objects = True
TEST_IMAGES = os.getcwd() + '\\test'
DIR_IMAGES = TEST_IMAGES + '/tif_pfcm'
MODEL_FILE = os.getcwd() + '/model'
DIR_RESULTS = os.getcwd() + '/results'
list_images = sorted(list(paths.list_images(DIR_IMAGES)))
num_images = len(list_images)
print("Number of test images: " + str(num_images))
list_results = sorted(list(listdir(DIR_RESULTS)))
for i, directory in enumerate(list_results):
    image.create_directory(DIR_RESULTS + "/" + directory + "/prediction")
    folder_model = DIR_RESULTS.replace(DIR_RESULTS, MODEL_FILE)
    list_parameters = directory.split("_")
    epochs = list_parameters[0]
    optimizer = list_parameters[7]
    if list_parameters[8] == "lecun" or list_parameters[8] == "glorot" or list_parameters[8] == "he":
        initializer = list_parameters[8] + "_" + list_parameters[9]
        dropout = list_parameters[10]
    else:
        initializer = list_parameters[8]
        dropout = list_parameters[9]

    model = conv.load_model(folder_model + "/" + directory, optimizer, dropout, initializer, False)
    test_gen = augment.test_generator(DIR_IMAGES, DIR_RESULTS + "/" + directory + "/prediction")
    predecir_generador = model.predict_generator(test_gen, num_images, verbose=2)
    conv.save_result(DIR_RESULTS + "/" + directory + "/prediction", predecir_generador, DIR_IMAGES, list_images)