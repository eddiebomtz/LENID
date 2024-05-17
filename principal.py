# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 10:48:09 2019

@author: eduardo
"""
import os
import argparse
from os import listdir
from conv import Conv
from image import Image
from augment import Augment
from preprocess import Preprocess
from imutils import paths
parser = argparse.ArgumentParser(description='Segmentation of extended objects.')
parser.add_argument("-d" , "--dir_images", action="store", dest="dir_images", help="Input directory")
parser.add_argument("-r", "--dir_results", action="store", dest="dir_result", help="Output directory")
parser.add_argument("-t", "--train", action="store_true", help="Train the model")
parser.add_argument("-s", "--segment", action="store_true", help="Segment with the previously CNN generated model")
parser.add_argument("-o", "--extended", action="store_true", help="Specify if you want to train or segment extended objects, must be used with -t or -s")
args = parser.parse_args()
#parser.print_help()
if args.train:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print("Traing...")
    list_epochs = [30]
    list_dropout = [0.2, 0.4]
    print("Extended objects...")
    list_optimizer = ['Adam']
    list_init_mode = ['he_normal']
    list_filter = [3]
    if args.resume:
        kf = int(args.kf)
        conv = Conv(None, None, None, None, None)
        conv.fit_generator_resume(kf)
    else:
        conv = Conv(list_epochs, list_optimizer, list_init_mode, list_filter, list_dropout)
        conv.fit_generator()
elif args.segment:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print("Segment...")
    image = Image()
    augment = Augment()
    conv = Conv(None, None, None, None, None)
    batch_size = 1
    print("Extended Objects...")
    TEST_IMAGES = os.getcwd() + '\\test'
    DIR_IMAGES = TEST_IMAGES + '/' + args.dir_images
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
            filter = list_parameters[10]
            dropout = list_parameters[11]
        else:
            initializer = list_parameters[8]
            filter = list_parameters[9]
            dropout = list_parameters[10]
        model = conv.load_model(folder_model + "/" + directory, optimizer, filter, dropout, initializer, False)
        test_gen = augment.test_generator(DIR_IMAGES, DIR_RESULTS + "/" + directory + "/prediction", args.extendidos)
        predecir_generador = model.predict_generator(test_gen, num_images, verbose=2)
        conv.save_result(DIR_RESULTS + "/" + directory + "/prediction", predecir_generador, DIR_IMAGES, list_images, args.stars)