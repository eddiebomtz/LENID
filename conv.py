# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 19:02:07 2020

@author: eduardo
"""
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import cv2
import numpy as np
from PIL import Image
from os import listdir
import skimage.io as io
from config import Config
from imutils import paths
from image import Image
from augment import Augment
from keras.models import Model
import matplotlib.pyplot as plt
from keras.regularizers import l2
from skimage.color import label2rgb
from contextlib import redirect_stdout
from preprocess import Preprocess
from scikitplot.metrics import plot_confusion_matrix, plot_roc
from sklearn.model_selection import KFold, RepeatedKFold, RepeatedStratifiedKFold
from keras.losses import binary_crossentropy
from keras.callbacks import Callback, TensorBoard, CSVLogger, ModelCheckpoint
from keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam
from keras.layers import Input, MaxPooling2D, Conv2D, Dropout, Conv2DTranspose, BatchNormalization, Activation, concatenate, LeakyReLU
class PerformanceVisualizationCallback(Callback):
    def __init__(self, model, validation_data, image_dir):
        super().__init__()
        self.model = model
        self.validation_data = validation_data
        os.makedirs(image_dir, exist_ok=True)
        self.image_dir = image_dir
    def on_epoch_end(self, epoch, logs={}):
        y_pred = np.asarray(self.model.predict(self.validation_data[0]))
        y_true = self.validation_data[1]
        y_pred_class = np.argmax(y_pred, axis=1)
        # plot and save confusion matrix
        fig, ax = plt.subplots(figsize=(16,12))
        plot_confusion_matrix(y_true, y_pred_class, ax=ax)
        fig.savefig(os.path.join(self.image_dir, 'confusion_matrix_epoch_' + str(epoch)))
       # plot and save roc curve
        fig, ax = plt.subplots(figsize=(16,12))
        plot_roc(y_true, y_pred, ax=ax)
        fig.savefig(os.path.join(self.image_dir, 'roc_curve_epoch_' + str(epoch)))
class Conv:
    def __init__(self, list_epochs, list_optimizer, list_init_mode, list_filter, list_dropout):
        self.list_epochs = list_epochs
        self.list_optimizer = list_optimizer
        self.list_init_mode = list_init_mode
        self.list_filter = list_filter
        self.list_dropout = list_dropout
    def __block_layers__(self, input, num_filters, tam_kernel, padding, strides, pool_size, kernel_init, activation, dropout, downsampling, layer_down=None):
        dropout = float(dropout)
        if downsampling:
            layer = Conv2D(num_filters, tam_kernel, padding=padding, kernel_regularizer=l2(0.005), bias_regularizer=l2(0.005), kernel_initializer = kernel_init)(input)
        else:
            conv2DTranspose = Conv2DTranspose(num_filters, pool_size, strides=strides, padding=padding)(input)
            layer = concatenate([conv2DTranspose, layer_down])
            layer = Conv2D(num_filters, tam_kernel, padding=padding, kernel_regularizer=l2(0.005), bias_regularizer=l2(0.005), kernel_initializer = kernel_init)(layer)
        layer = Activation(activation)(layer)
        layer = Conv2D(num_filters, tam_kernel, padding=padding, kernel_regularizer=l2(0.005), bias_regularizer=l2(0.005), kernel_initializer = kernel_init)(layer)
        layer = Activation(activation)(layer)
        layer = Conv2D(num_filters, tam_kernel, padding=padding, kernel_regularizer=l2(0.005), bias_regularizer=l2(0.005), kernel_initializer = kernel_init)(layer)
        layer = Activation(activation)(layer)
        if dropout > 0:
            layer = Dropout(dropout)(layer)
        if downsampling:
            maxpool = MaxPooling2D(pool_size, strides=strides)(layer) 
            return layer, maxpool
        else:
            if dropout > 0:
                layer = Dropout(dropout)(layer)
            return layer, conv2DTranspose
    def dice_coef(self, y_true, y_pred):
        import keras.backend as K
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())
    def dice_coef_loss(self, y_true, y_pred):
        return 1. - self.dice_coef(y_true, y_pred)
    def __create_model__(self, input_size, optimizer, filter, dropout_rate, init_mode, resumen):
        filter = int(filter)
        dropout_rate = float(dropout_rate)
        inputs = Input(input_size)
        down0, maxpool0 = self.__block_layers__(inputs, 32, (filter, filter), 'same', (2, 2), (2, 2), init_mode, 'relu', 0, True)
        down1, maxpool1 = self.__block_layers__(maxpool0, 64, (filter, filter), 'same', (2, 2), (2, 2), init_mode, 'relu', 0, True)
        down2, maxpool2 = self.__block_layers__(maxpool1, 128, (filter, filter), 'same', (2, 2), (2, 2), init_mode, 'relu', dropout_rate/2, True)
        #down3, maxpool3 = self.__block_layers__(maxpool2, 256, (filter, filter), 'same', (2, 2), (2, 2), init_mode, 'relu', 0, True)
        #down4, maxpool4 = self.__block_layers__(maxpool3, 512, (filter, filter), 'same', (2, 2), (2, 2), init_mode, 'relu', dropout_rate, True)
        center, maxpoolc = self.__block_layers__(maxpool2, 256, (filter, filter), 'same', (2, 2), (2, 2), init_mode, 'relu', dropout_rate, True)
        #up4, _ = self.__block_layers__(maxpool4, 512, (filter, filter), 'same', (2, 2), (2, 2), init_mode, 'relu', dropout_rate, False, down4)
        #up3, _ = self.__block_layers__(up4, 256, (filter, filter), 'same', (2, 2), (2, 2), init_mode, 'relu', 0, False, down3)
        up2, _ = self.__block_layers__(maxpool2, 128, (filter, filter), 'same', (2, 2), (2, 2), init_mode, 'relu', dropout_rate/2, False, down2)
        up1, _ = self.__block_layers__(up2, 64, (filter, filter), 'same', (2, 2), (2, 2), init_mode, 'relu', 0, False, down1)
        up0, _ = self.__block_layers__(up1, 32, (filter, filter), 'same', (2, 2), (2, 2), init_mode, 'relu', 0, False, down0)
        segment = Conv2D(1, 1, activation = 'sigmoid')(up0)
        model = Model(inputs = inputs, outputs = segment)
        if optimizer == "Adam":
            model.compile(optimizer = Adam(lr=2e-4), loss = self.bce_dice_loss, metrics = [self.dice_coef, 'accuracy'])
        elif optimizer == "SGD":
            model.compile(optimizer = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True), loss = 'binary_crossentropy', metrics = ['accuracy'])
        elif optimizer == "RMSprop":
            model.compile(optimizer = RMSprop(lr=0.001, rho=0.9), loss = 'binary_crossentropy', metrics = ['accuracy'])
        elif optimizer == "Adagrad":
            model.compile(optimizer = Adagrad(lr=0.01), loss = 'binary_crossentropy', metrics = ['accuracy'])
        elif optimizer == "Adadelta":
            model.compile(optimizer = Adadelta(lr=1.0, rho=0.95), loss = 'binary_crossentropy', metrics = ['accuracy'])
        elif optimizer == "Adamax":
            model.compile(optimizer = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999), loss = 'binary_crossentropy', metrics = ['accuracy'])
        elif optimizer == "Nadam":
            model.compile(optimizer = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999), loss = 'binary_crossentropy', metrics = ['accuracy'])
        if resumen:
            print("*** SAVING CONVOLUTIONAL NEURAL NETWORK RESUME ***")
            with open(self.config.PATH_SAVE_MODEL + '\\resume_model.txt', 'w') as f:
                with redirect_stdout(f):
                    model.summary()
        return model
    def load_model(self, path_model, optimizer, filter, dropout, initializer, resumen):
        model = self.__create_model__((None, None, 1), optimizer, filter, dropout, initializer, resumen)
        model.load_weights(path_model + '\\model_cnn.hdf5')
        return model
    def bce_dice_loss(self, y_true, y_pred):
        loss = binary_crossentropy(y_true, y_pred) + self.dice_coef_loss(y_true, y_pred)
        return loss
    def save_images_k_fold(self, i, training_images, masks_training, validate_images, masks_validate):
        print("*****************************************")
        print("Saving training images")
        print("Number of training images: " + str(len(training_images)))
        for path_img in training_images:
            img = self.imageobj.read_TIFF(path_img)
            name_img = path_img.replace(os.getcwd() + "\\" + self.config.PATH_IMAGES + '\\', "")
            io.imsave(self.config.KFOLD_TRAINING + self.config.PATH_IMAGES + "\\" + name_img, img)
        print("Number of mask (training): " + str(len(masks_training)))
        for path_img in masks_training:
            img = self.imageobj.read_TIFF(path_img)
            name_img = path_img.replace(os.getcwd() + "\\"  + self.config.PATH_MASKS + '\\', "")
            io.imsave(self.config.KFOLD_TRAINING + self.config.PATH_MASKS + "\\" + name_img, img)
        print("Saving images for validation")
        print("Number of images to validate: " + str(len(validate_images)))
        for path_img in validate_images:
            img =  self.imageobj.read_TIFF(path_img)
            name_img = path_img.replace(os.getcwd() + "\\" + self.config.PATH_IMAGES + '\\', "")
            io.imsave(self.config.KFOLD_TRAINING + self.config.PATH_VALIDATE_IMG + "\\" + name_img, img)
        print("Number of masks to validate: " + str(len(masks_validate)))
        for path_img in masks_validate:
            img =  self.imageobj.read_TIFF(path_img)
            name_img = path_img.replace(os.getcwd() + "\\" + self.config.PATH_MASKS + '\\', "")
            io.imsave(self.config.KFOLD_TRAINING + self.config.PATH_VALIDATE_MASKS + "\\" + name_img, img)
        print("*****************************************")
    def create_directory(self, path):
        try:
            os.stat(path)
        except:
            os.mkdir(path)
    def histogram(self, grayscale):
        counts, vals = np.histogram(grayscale, bins=range(2 ** 8))
        plt.plot(range(0, (2 ** 8) - 1), counts)
        plt.title("Grayscale image histogram")
        plt.xlabel("Pixel intensity")
        plt.ylabel("Count")
        plt.savefig(self.config.PATH_SAVE_PLOTS + "/plot_histogram.png")
        plt.show()
        plt.close()
    
    def prediction_to_img(self, prediction, percentage):
        img_bool = prediction.astype('float')
        img_bool[img_bool > percentage] = 1
        img_bool[img_bool <= percentage] = 0
        image = img_bool.astype('uint16')
        image = image * 65535
        image.shape = prediction.shape
        return image, img_bool
    def remove_stars(self, name, img_original, save_path, img_bool, percentage):
        img_without_stars = img_original * (1 - img_bool)
        img_without_stars.shape = img_original.shape
        img_without_stars = img_without_stars.astype('uint16')
        return save_path + "/" + name + "_sin_stars_" + str(percentage) + ".tif", img_without_stars
    def prediction_image_rgb(self, name, img_original, img_bool, save_path, color_rgba, percentage):
        h, w = img_bool.shape
        img_original = Image.new('RGBA', (h, w), (0,0,0,0))
        stars = Image.new('RGBA', (h, w), (0, 0, 0, 0))
        imgrgba = np.zeros((h,w,4))
        imguint = img_bool.astype('uint8')
        imguint = imguint.reshape(h, w)
        for k in range(0, imgrgba.shape[0]):
            for j in range(0, imgrgba.shape[1]):
                if imguint[k, j] == 1:
                    imgrgba[k,j,0] = color_rgba[0]
                    imgrgba[k,j,1] = color_rgba[1]
                    imgrgba[k,j,2] = color_rgba[2]
                    imgrgba[k,j,3] = color_rgba[3]
                else:
                    imgrgba[k,j,0] = 0
                    imgrgba[k,j,1] = 0
                    imgrgba[k,j,2] = 0
                    imgrgba[k,j,3] = 0
        stars = Image.fromarray(np.uint8(imgrgba))
        img_original = Image.fromarray(np.uint16(img_original))
        image_with_stars = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        image_with_stars.paste(img_original, box = (0, 0))
        image_with_stars.paste(stars, box = (0, 0), mask=stars)
        image_with_stars.save(save_path + "/" + name + "_predict_rgb_" + str(percentage) + ".tif")
        return stars
    def alphanumeric_order(self, list):
        import re
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
        return sorted(list, key=alphanum_key)
    def save_result(self, save_path,npimg, folder_images, list_images, stars):
        list_images = self.alphanumeric_order(list_images)
        for i,item in enumerate(npimg):
            img_predict = item[:,:,0]
            img = img_predict * 65535
            simg = img.astype('uint16')
            original_path = list_images[i]
            name = original_path.replace(folder_images + "\\", "")
            name = name.replace(".tif", "")
            print("Path: " + save_path)
            print("name: " + name)
            io.imsave(os.path.join(save_path,name + "_predict.tif"),simg)
            imgint20, img_bool20 = self.prediction_to_img(img_predict, 0.2)
            io.imsave(os.path.join(save_path,name + "_predict_20.tif"),imgint20)
            imgint30, img_bool30 = self.prediction_to_img(img_predict, 0.3)
            io.imsave(os.path.join(save_path,name + "_predict_30.tif"),imgint30)
            imgint50, img_bool = self.prediction_to_img(img_predict, 0.5)
            io.imsave(os.path.join(save_path,name + "_predict_50.tif"),imgint50)
            imgint70, img_bool70 = self.prediction_to_img(img_predict, 0.7)
            io.imsave(os.path.join(save_path,name + "_predict_70.tif"),imgint70)
            imgint80, img_bool80 = self.prediction_to_img(img_predict, 0.8)
            io.imsave(os.path.join(save_path,name + "_predict_80.tif"),imgint80)
            imgint90, img_bool90 = self.prediction_to_img(img_predict, 0.9)
            io.imsave(os.path.join(save_path,name + "_predict_90.tif"),imgint90)
            
            self.imageobj = Image()
            img_original = Image.open(original_path)
            img_original = np.array(img_original)
            img_original.shape = img_predict.shape
            img_original = img_original.astype('uint16')
            #20% probability
            stars20 = self.prediction_image_rgb(name, img_original, img_bool20, save_path, [141, 9, 118, 128], 20)
            #30% probability
            stars30 = self.prediction_image_rgb(name, img_original, img_bool30, save_path, [22, 124, 243, 128], 30)
            #50% probability
            stars50 = self.prediction_image_rgb(name, img_original, img_bool, save_path, [245, 138, 25, 128], 50)
            #70% probability
            stars70 = self.prediction_image_rgb(name, img_original, img_bool70, save_path, [234, 51, 11, 128], 70)
            #90% probability
            stars80 = self.prediction_image_rgb(name, img_original, img_bool80, save_path, [0, 128, 0, 128], 80)
            #90% probability
            stars90 = self.prediction_image_rgb(name, img_original, img_bool90, save_path, [255, 255, 0, 128], 90)
            h, w = img_original.shape
            img_original = Image.new('RGBA', (h, w), (0,0,0,0))
            img_original = Image.fromarray(np.uint16(img_original))
            image_with_stars = Image.new('RGBA', (w, h), (0, 0, 0, 0))
            image_with_stars.paste(img_original, box = (0, 0))
            image_with_stars.paste(stars20, box = (0, 0), mask=stars20)
            image_with_stars.paste(stars30, box = (0, 0), mask=stars30)
            image_with_stars.paste(stars50, box = (0, 0), mask=stars50)
            image_with_stars.paste(stars70, box = (0, 0), mask=stars70)
            image_with_stars.paste(stars80, box = (0, 0), mask=stars80)
            image_with_stars.paste(stars90, box = (0, 0), mask=stars90)
            image_with_stars.save(save_path + "/" + name + "_predict_rgb_todas.tif")
            
    def save_plots(self, i, hist, save_path):
        accuracy = hist.history['acc']
        val_accuracy = hist.history['val_acc']
        dice_coef = hist.history['dice_coef']
        val_dice_coef = hist.history['val_dice_coef']
        loss = hist.history['loss']
        val_loss = hist.history['val_loss']
        #dice_coef_loss = hist.history['dice_coef_loss']
        #val_dice_coef_loss = hist.history['val_dice_coef_loss']
        epochs_range = range(len(accuracy))
        plt.plot(epochs_range, accuracy, 'bo', label='Training accuracy')
        plt.plot(epochs_range, val_accuracy, 'b', label='Validation accuracy')
        plt.plot(epochs_range, dice_coef, 'ro', label='Training dice coef')
        plt.plot(epochs_range, val_dice_coef, 'r', label='Validation dice coef')
        plt.ylim(0, 1)
        plt.title('Training accuracy and dice coefficient')
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("accuracy and dice_coefficient")
        plt.savefig(save_path + "/plot_accuracy_dice_coefficient_kfold_" + str(i) + ".png")
        plt.show()
        
        plt.plot(epochs_range, accuracy, 'bo', label='Training accuracy')
        plt.plot(epochs_range, val_accuracy, 'b', label='Validation accuracy')
        plt.ylim(0, 1)
        plt.title('Training accuracy')
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy and Dice Coefficient")
        plt.savefig(save_path + "/plot_accuracy_kfold_" + str(i) + ".png")
        plt.show()
        
        plt.plot(epochs_range, dice_coef, 'ro', label='Training dice coef')
        plt.plot(epochs_range, val_dice_coef, 'r', label='Validation dice coef')
        plt.ylim(0, 1)
        plt.title('Training Dice Coefficient')
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Dice Coefficient")
        plt.savefig(save_path + "/plot_dice_coefficient_kfold_" + str(i) + ".png")
        plt.show()
        
        plt.plot(epochs_range, loss, 'bo', label='Training dice coef loss')
        plt.plot(epochs_range, val_loss, 'b', label='Validation dice coef loss')
        plt.plot(np.argmin(hist.history["val_loss"]), np.min(hist.history["val_loss"]), marker="x", color="r", label="Best model")
        plt.title("Learning curve")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.savefig(save_path + "/plot_dice_coef_loss_kfold_" + str(i) + ".png")
        plt.show()
        
    def fit_generator(self):
        batch_size = 6
        for e in self.list_epochs:
            for o in self.list_optimizer:
                for ini in self.list_init_mode:
                    for f in self.list_filter:
                        for d in self.list_dropout:
                            i = 0
                            self.config = Config(e, o, ini, f, d, 1)
                            self.imageobj = Image()
                            list_images = sorted(list(paths.list_images(os.getcwd() + "\\" + self.config.PATH_IMAGES)))
                            list_masks = sorted(list(paths.list_images(os.getcwd() + "\\" + self.config.PATH_MASKS)))
                            k_folds = 10
                            kf = KFold(n_splits = k_folds, random_state = 42, shuffle = True)
                            X = np.array(list_images)
                            y = np.array(list_masks)
                            print("Parameters: Epochs: " + str(e) + " Optimizer: " + str(o) + " Init mode: " + str(ini) + " filter " + str(f) + " dropout " + str(d))
                            if i > 0:
                                self.config.config_directories(e)
                            model = self.__create_model__((None, None, 1), o, f, d, ini, True)
                            k = 1
                            for training_x, validate_y in kf.split(X):
                                training_images = X[training_x]
                                validate_images = X[validate_y]
                                masks_training = y[training_x]
                                masks_validate = y[validate_y]
                                self.config.config_directories_k_fold(k, o)
                                self.save_images_k_fold(k, training_images, masks_training, validate_images, masks_validate)
                                try: 
                                    model.load_weights(self.config.MODEL_FILE + '\\model_cnn.hdf5')
                                except Exception as OSError:
                                    pass
                                csv = CSVLogger(self.config.PATH_SAVE_RESULTS + '\\training_kfold_' + str(k) + o + '.log')
                                tb = TensorBoard(log_dir=self.config.PATH_SAVE_RESULTS + '\\kfold_' + str(k) + '_' + o + '\\', histogram_freq=0, write_graph=True, write_images=True)
                                checkpoint = ModelCheckpoint(self.config.PATH_SAVE_MODEL + '\\model_cnn.hdf5', monitor='val_loss',verbose=2, save_best_only=True)
                                steps_per_epoch = len(training_images) // batch_size
                                val_steps = len(validate_images) // batch_size
                                augment = Augment()
                                images = sorted(list(paths.list_images(self.config.KFOLD_TRAINING + "\\" + self.config.PATH_IMAGES + "\\" )))
                                masks = sorted(list(paths.list_images(self.config.KFOLD_TRAINING + "\\" + self.config.PATH_MASKS + "\\")))
                                train_gen = augment.augment_generator(batch_size, images, masks, self.config.PATH_SAVE_AUGMENTED, self.config.PATH_SAVE_AUGMENTED_MAS, self.config.SIZE_IMAGES_X, self.config.SIZE_IMAGES_Y, augment = True)
                                images = sorted(list(paths.list_images(self.config.KFOLD_TRAINING + "\\" + self.config.PATH_VALIDATE_IMG + "\\")))
                                masks = sorted(list(paths.list_images(self.config.KFOLD_TRAINING + "\\" + self.config.PATH_VALIDATE_MASKS + "\\")))
                                validate_gen = augment.augment_generator(batch_size, images, masks, self.config.PATH_SAVE_AUGMENTED_VAL, self.config.PATH_SAVE_AUGMENTED_VAL_MAS, self.config.SIZE_IMAGES_X, self.config.SIZE_IMAGES_Y, augment = False)
                                #validation_data = list(validate_gen)
                                #performance_cbk = PerformanceVisualizationCallback(model=model, validation_data=validation_data, image_dir=self.config.PATH_PLOTS)
                                hist = model.fit_generator(train_gen, validation_data=validate_gen, validation_steps=val_steps, 
                                                            steps_per_epoch=steps_per_epoch, epochs=e, callbacks=[checkpoint, csv, tb])
                                self.save_plots(k, hist, self.config.PATH_SAVE_PLOTS)
                                k += 1
                            i += 1