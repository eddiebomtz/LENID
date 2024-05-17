# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 12:48:20 2020
@author: eduardo
"""
import os
import time
class Config:
    TRAINING = '\\training\\'
    TRAINING_STARS = '\\training_stars\\'
    #PATH_IMAGES = TRAINING + 'images_sin_fondo_512_threshold_2000'
    #PATH_IMAGES = TRAINING + 'images_sin_fondo_512_threshold_2000'
    PATH_IMAGES = TRAINING + 'cropped_images'
    PATH_IMAGES_STARS = TRAINING_STARS + 'fits_tif_512_pfcm'
    #PATH_MASKS = TRAINING + 'masks_512_threshold_2000'
    #PATH_MASKS = TRAINING + 'masks_512_threshold_2000'
    PATH_MASKS = TRAINING + 'cropped_masks'
    PATH_MASKS_STARS = TRAINING_STARS + 'fits_tif_masks_512'
    VALIDATION = '\\validate\\'
    VALIDATION_STARS = '\\validate_stars\\'
    #PATH_VALIDATE_IMG = VALIDATION + 'images_sin_fondo_512_threshold_2000'
    #PATH_VALIDATE_IMG = VALIDATION + 'images_sin_fondo_512_threshold_2000'
    PATH_VALIDATE_IMG = VALIDATION + 'cropped_images'
    PATH_VALIDATE_STARS_IMG = VALIDATION_STARS + 'fits_tif_512_pfcm'
    #PATH_VALIDATE_MASKS = VALIDATION + 'masks_512_threshold_2000'
    #PATH_VALIDATE_MASKS = VALIDATION + 'masks_512_threshold_2000'
    PATH_VALIDATE_MASKS = VALIDATION + 'cropped_masks'
    PATH_VALIDATE_STARS_MAS = VALIDATION_STARS + 'fits_tif_masks_512'
    PATH_MODEL = os.getcwd() + '\\model'
    PATH_MODEL_STARS = os.getcwd() + '\\model_stars'
    PATH_PLOTS = os.getcwd() + '\\plots'
    PATH_PLOTS_STARS = os.getcwd() + '\\plots_stars'
    PATH_RESULTS = os.getcwd() + '\\results'
    PATH_RESULTS_STARS = os.getcwd() + '\\results_stars'
    SIZE_IMAGES_X = 512
    SIZE_IMAGES_Y = 512
    __PATH_AUGMENTED = os.getcwd() + '\\augmented'
    __PATH_AUGMENTED_MAS = os.getcwd() + '\\augmented_masks'
    __PATH_AUGMENTED_VAL = os.getcwd() + '\\augmented_val'
    __PATH_AUGMENTED_VAL_MAS = os.getcwd() + '\\augmented_val_masks'
    def __init__(self, e, o, ini, f, d, type):
        if type == 1:
            self.config_directories(e, o, ini, f, d)
        elif type == 2:
            self.config_directories_stars(e, o, ini, d)
    def __create_directory__(self, path):
        try:
            os.stat(path)
        except:
            os.mkdir(path)
    def config_directories_resume(self, t, e, o, ini, d):
        self.__time = time.strftime(t)
        #STARS
        self.__create_directory__(self.PATH_MODEL_STARS)
        self.PATH_SAVE_MODEL_STARS = self.PATH_MODEL_STARS + "\\" + str(e) + "_" + self.__time + "_" + o + "_" + str(ini) + "_" + str(d) + "_2_layers"
        self.__create_directory__(self.PATH_SAVE_MODEL_STARS)
        self.__create_directory__(self.PATH_PLOTS_STARS)
        self.PATH_SAVE_PLOTS_STARS = self.PATH_PLOTS_STARS + "\\" + str(e) + "_" + self.__time + "_" + o + "_" + str(ini) + "_" + str(d) + "_2_layers"
        self.__create_directory__(self.PATH_SAVE_PLOTS_STARS)
        self.__create_directory__(self.PATH_RESULTS_STARS)
        self.PATH_SAVE_RESULTS_STARS = self.PATH_RESULTS_STARS + "\\" + str(e) + "_" + self.__time + "_" + o + "_" + str(ini) + "_" + str(d) + "_2_layers"
        self.__create_directory__(self.PATH_SAVE_RESULTS_STARS)
        #AUMENTADAS
        self.__create_directory__(self.__PATH_AUGMENTED)
        self.PATH_SAVE_AUGMENTED = self.__PATH_AUGMENTED + "\\" + str(e) + "_" + self.__time + "_" + o + "_" + str(ini) + "_" + str(d) + "_2_layers"
        self.__create_directory__(self.PATH_SAVE_AUGMENTED)
        self.__create_directory__(self.__PATH_AUGMENTED_MAS)
        self.PATH_SAVE_AUGMENTED_MAS = self.__PATH_AUGMENTED_MAS + "\\" + str(e) + "_" + self.__time + "_" + o + "_" + str(ini) + "_" + str(d) + "_2_layers"
        self.__create_directory__(self.PATH_SAVE_AUGMENTED_MAS)
        self.__create_directory__(self.__PATH_AUGMENTED_VAL)
        self.PATH_SAVE_AUGMENTED_VAL = self.__PATH_AUGMENTED_VAL + "\\" + str(e) + "_" + self.__time + "_" + o + "_" + str(ini) + "_" + str(d) + "_2_layers"
        self.__create_directory__(self.PATH_SAVE_AUGMENTED_VAL)
        self.__create_directory__(self.__PATH_AUGMENTED_VAL_MAS)
        self.PATH_SAVE_AUGMENTED_VAL_MAS = self.__PATH_AUGMENTED_VAL_MAS + "\\" + str(e) + "_" + self.__time + "_" + o + "_" + str(ini) + "_" + str(d) + "_2_layers"
        self.__create_directory__(self.PATH_SAVE_AUGMENTED_VAL_MAS)
    def config_directories(self, e, o, ini, f, d):
        self.__time = time.strftime("%d_%m_%y_%H_%M_%S")
        #NEBULOSAS
        self.__create_directory__(self.PATH_MODEL)
        self.PATH_SAVE_MODEL = self.PATH_MODEL + "\\" + str(e) + "_" + self.__time + "_" + o + "_" + str(ini) + "_" + str(f) + "_" + str(d) + "_3_layers"
        self.__create_directory__(self.PATH_SAVE_MODEL)
        self.__create_directory__(self.PATH_PLOTS)
        self.PATH_SAVE_PLOTS = self.PATH_PLOTS + "\\" + str(e) + "_" + self.__time + "_" + o + "_" + str(ini) + "_" + str(f) + "_" + str(d) + "_3_layers"
        self.__create_directory__(self.PATH_SAVE_PLOTS)
        self.__create_directory__(self.PATH_RESULTS)
        self.PATH_SAVE_RESULTS = self.PATH_RESULTS + "\\" + str(e) + "_" + self.__time + "_" + o + "_" + str(ini) + "_" + str(f) + "_" + str(d) + "_3_layers"
        self.__create_directory__(self.PATH_SAVE_RESULTS)
        #AUMENTADAS
        self.__create_directory__(self.__PATH_AUGMENTED)
        self.PATH_SAVE_AUGMENTED = self.__PATH_AUGMENTED + "\\" + str(e) + "_" + self.__time + "_" + o + "_" + str(ini) + "_" + str(f) + "_" +  str(d) + "_3_layers"
        self.__create_directory__(self.PATH_SAVE_AUGMENTED)
        self.__create_directory__(self.__PATH_AUGMENTED_MAS)
        self.PATH_SAVE_AUGMENTED_MAS = self.__PATH_AUGMENTED_MAS + "\\" + str(e) + "_" + self.__time + "_" + o + "_" + str(ini) + "_" + str(f) + "_" +  str(d) + "_3_layers"
        self.__create_directory__(self.PATH_SAVE_AUGMENTED_MAS)
        self.__create_directory__(self.__PATH_AUGMENTED_VAL)
        self.PATH_SAVE_AUGMENTED_VAL = self.__PATH_AUGMENTED_VAL + "\\" + str(e) + "_" + self.__time + "_" + o + "_" + str(ini) + "_" + str(f) + "_" +  str(d) + "_3_layers"
        self.__create_directory__(self.PATH_SAVE_AUGMENTED_VAL)
        self.__create_directory__(self.__PATH_AUGMENTED_VAL_MAS)
        self.PATH_SAVE_AUGMENTED_VAL_MAS = self.__PATH_AUGMENTED_VAL_MAS + "\\" + str(e) + "_" + self.__time + "_" + o + "_" + str(ini) + "_" + str(f) + "_" +  str(d) + "_3_layers"
        self.__create_directory__(self.PATH_SAVE_AUGMENTED_VAL_MAS)
    def config_directories_stars(self, e, o, ini, d):
        self.__time = time.strftime("%d_%m_%y_%H_%M_%S")
        #STARS
        self.__create_directory__(self.PATH_MODEL_STARS)
        self.PATH_SAVE_MODEL_STARS = self.PATH_MODEL_STARS + "\\" + str(e) + "_" + self.__time + "_" + o + "_" + str(ini) + "_" + str(d) + "_2_layers"
        self.__create_directory__(self.PATH_SAVE_MODEL_STARS)
        self.__create_directory__(self.PATH_PLOTS_STARS)
        self.PATH_SAVE_PLOTS_STARS = self.PATH_PLOTS_STARS + "\\" + str(e) + "_" + self.__time + "_" + o + "_" + str(ini) + "_" + str(d) + "_2_layers"
        self.__create_directory__(self.PATH_SAVE_PLOTS_STARS)
        self.__create_directory__(self.PATH_RESULTS_STARS)
        self.PATH_SAVE_RESULTS_STARS = self.PATH_RESULTS_STARS + "\\" + str(e) + "_" + self.__time + "_" + o + "_" + str(ini) + "_" + str(d) + "_2_layers"
        self.__create_directory__(self.PATH_SAVE_RESULTS_STARS)
        #AUMENTADAS
        self.__create_directory__(self.__PATH_AUGMENTED)
        self.PATH_SAVE_AUGMENTED = self.__PATH_AUGMENTED + "\\" + str(e) + "_" + self.__time + "_" + o + "_" + str(ini) + "_" + str(d) + "_2_layers"
        self.__create_directory__(self.PATH_SAVE_AUGMENTED)
        self.__create_directory__(self.__PATH_AUGMENTED_MAS)
        self.PATH_SAVE_AUGMENTED_MAS = self.__PATH_AUGMENTED_MAS + "\\" + str(e) + "_" + self.__time + "_" + o + "_" + str(ini) + "_" + str(d) + "_2_layers"
        self.__create_directory__(self.PATH_SAVE_AUGMENTED_MAS)
        self.__create_directory__(self.__PATH_AUGMENTED_VAL)
        self.PATH_SAVE_AUGMENTED_VAL = self.__PATH_AUGMENTED_VAL + "\\" + str(e) + "_" + self.__time + "_" + o + "_" + str(ini) + "_" + str(d) + "_2_layers"
        self.__create_directory__(self.PATH_SAVE_AUGMENTED_VAL)
        self.__create_directory__(self.__PATH_AUGMENTED_VAL_MAS)
        self.PATH_SAVE_AUGMENTED_VAL_MAS = self.__PATH_AUGMENTED_VAL_MAS + "\\" + str(e) + "_" + self.__time + "_" + o + "_" + str(ini) + "_" + str(d) + "_2_layers"
        self.__create_directory__(self.PATH_SAVE_AUGMENTED_VAL_MAS)
    def config_directories_k_fold(self, k, o):
        '''print("*****************************************")
        print("Training K-Fold " + str(k) + " " + o)
        print("*****************************************")
        print("Creating folder " + self.__time + " en " + self.TRAINING)'''
        #NEBULOSAS
        self.TRAINING_TIME = os.getcwd() + "\\" + self.TRAINING + "\\" + self.__time
        self.__create_directory__(self.TRAINING_TIME)
        self.KFOLD_TRAINING = self.TRAINING_TIME + "\\kfold_" + str(k)
        #NEBULOSAS
        #print("Creating folder kfold_" + str(k) + " en " + self.TRAINING_TIME)
        self.__create_directory__(self.KFOLD_TRAINING)
        #print("Creating folder for images (training)")
        self.__create_directory__(self.KFOLD_TRAINING + "\\" + self.TRAINING + "\\")
        self.__create_directory__(self.KFOLD_TRAINING + "\\" + self.PATH_IMAGES)
        #print("Creating folder for masks (training)")
        self.__create_directory__(self.KFOLD_TRAINING + "\\" + self.TRAINING + "\\")
        self.__create_directory__(self.KFOLD_TRAINING + "\\" + self.PATH_MASKS)
        #print("Creating folder for images (validación)")
        self.__create_directory__(self.KFOLD_TRAINING + "\\" + self.VALIDATION + "\\")
        self.__create_directory__(self.KFOLD_TRAINING + "\\" + self.PATH_VALIDATE_IMG)
        #print("Creating folder for masks (validación)")
        self.__create_directory__(self.KFOLD_TRAINING + "\\" + self.VALIDATION + "\\")
        self.__create_directory__(self.KFOLD_TRAINING + "\\" + self.PATH_VALIDATE_MASKS)
        self.__create_directory__(self.PATH_SAVE_RESULTS + "\\kfold_" + str(k) + "_" + o + "\\")
        #print("*****************************************")
    def config_directories_k_fold_stars(self, k, o):
        #STARS
        self.TRAINING_STARS_TIME = os.getcwd() + "\\" + self.TRAINING_STARS + "\\" + self.__time
        self.__create_directory__(self.TRAINING_STARS_TIME)
        self.TRAINING_STARS_KFOLD = self.TRAINING_STARS_TIME + "\\kfold_" + str(k)
        #STARS
        self.__create_directory__(self.TRAINING_STARS_KFOLD)
        self.__create_directory__(self.TRAINING_STARS_KFOLD + "\\" + self.TRAINING_STARS + "\\")
        self.__create_directory__(self.TRAINING_STARS_KFOLD + "\\" + self.PATH_IMAGES_STARS)
        self.__create_directory__(self.TRAINING_STARS_KFOLD + "\\" + self.TRAINING_STARS + "\\")
        self.__create_directory__(self.TRAINING_STARS_KFOLD + "\\" + self.PATH_MASKS_STARS)
        self.__create_directory__(self.TRAINING_STARS_KFOLD + "\\" + self.VALIDATION_STARS + "\\")
        self.__create_directory__(self.TRAINING_STARS_KFOLD + "\\" + self.PATH_VALIDATE_STARS_IMG)
        self.__create_directory__(self.TRAINING_STARS_KFOLD + "\\" + self.VALIDATION_STARS + "\\")
        self.__create_directory__(self.TRAINING_STARS_KFOLD + "\\" + self.PATH_VALIDATE_STARS_MAS)
        self.__create_directory__(self.PATH_SAVE_RESULTS_STARS + "\\kfold_" + str(k) + "_" + o + "\\")