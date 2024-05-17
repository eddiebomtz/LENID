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
from imagen import Imagen
from aumentar import Aumentar
from keras.models import Model
import matplotlib.pyplot as plt
from keras.regularizers import l2
from skimage.color import label2rgb
from contextlib import redirect_stdout
from preprocesamiento import preprocesamiento
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
    def __init__(self, lista_epochs, lista_optimizador, lista_init_mode, lista_filtro, lista_dropout):
        self.lista_epochs = lista_epochs
        self.lista_optimizador = lista_optimizador
        self.lista_init_mode = lista_init_mode
        self.lista_filtro = lista_filtro
        self.lista_dropout = lista_dropout
    def __bloque_capas__(self, entrada, num_filtros, tam_kernel, padding, strides, pool_size, kernel_init, activacion, dropout, downsampling, capa_down=None):
        dropout = float(dropout)
        if downsampling:
            capa = Conv2D(num_filtros, tam_kernel, padding=padding, kernel_regularizer=l2(0.005), bias_regularizer=l2(0.005), kernel_initializer = kernel_init)(entrada)
        else:
            conv2DTranspose = Conv2DTranspose(num_filtros, pool_size, strides=strides, padding=padding)(entrada)
            capa = concatenate([conv2DTranspose, capa_down])
            capa = Conv2D(num_filtros, tam_kernel, padding=padding, kernel_regularizer=l2(0.005), bias_regularizer=l2(0.005), kernel_initializer = kernel_init)(capa)
        capa = Activation(activacion)(capa)
        capa = Conv2D(num_filtros, tam_kernel, padding=padding, kernel_regularizer=l2(0.005), bias_regularizer=l2(0.005), kernel_initializer = kernel_init)(capa)
        capa = Activation(activacion)(capa)
        capa = Conv2D(num_filtros, tam_kernel, padding=padding, kernel_regularizer=l2(0.005), bias_regularizer=l2(0.005), kernel_initializer = kernel_init)(capa)
        capa = Activation(activacion)(capa)
        if dropout > 0:
            capa = Dropout(dropout)(capa)
        if downsampling:
            maxpool = MaxPooling2D(pool_size, strides=strides)(capa) 
            return capa, maxpool
        else:
            if dropout > 0:
                capa = Dropout(dropout)(capa)
            return capa, conv2DTranspose
    def dice_coef(self, y_true, y_pred):
        import keras.backend as K
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())
    def dice_coef_loss(self, y_true, y_pred):
        return 1. - self.dice_coef(y_true, y_pred)
    def __crea_modelo__(self, input_size, optimizer, filtro, dropout_rate, init_mode, resumen):
        filtro = int(filtro)
        dropout_rate = float(dropout_rate)
        inputs = Input(input_size)
        down0, maxpool0 = self.__bloque_capas__(inputs, 32, (filtro, filtro), 'same', (2, 2), (2, 2), init_mode, 'relu', 0, True)
        down1, maxpool1 = self.__bloque_capas__(maxpool0, 64, (filtro, filtro), 'same', (2, 2), (2, 2), init_mode, 'relu', 0, True)
        down2, maxpool2 = self.__bloque_capas__(maxpool1, 128, (filtro, filtro), 'same', (2, 2), (2, 2), init_mode, 'relu', dropout_rate/2, True)
        #down3, maxpool3 = self.__bloque_capas__(maxpool2, 256, (filtro, filtro), 'same', (2, 2), (2, 2), init_mode, 'relu', 0, True)
        #down4, maxpool4 = self.__bloque_capas__(maxpool3, 512, (filtro, filtro), 'same', (2, 2), (2, 2), init_mode, 'relu', dropout_rate, True)
        center, maxpoolc = self.__bloque_capas__(maxpool2, 256, (filtro, filtro), 'same', (2, 2), (2, 2), init_mode, 'relu', dropout_rate, True)
        #up4, _ = self.__bloque_capas__(maxpool4, 512, (filtro, filtro), 'same', (2, 2), (2, 2), init_mode, 'relu', dropout_rate, False, down4)
        #up3, _ = self.__bloque_capas__(up4, 256, (filtro, filtro), 'same', (2, 2), (2, 2), init_mode, 'relu', 0, False, down3)
        up2, _ = self.__bloque_capas__(maxpool2, 128, (filtro, filtro), 'same', (2, 2), (2, 2), init_mode, 'relu', dropout_rate/2, False, down2)
        up1, _ = self.__bloque_capas__(up2, 64, (filtro, filtro), 'same', (2, 2), (2, 2), init_mode, 'relu', 0, False, down1)
        up0, _ = self.__bloque_capas__(up1, 32, (filtro, filtro), 'same', (2, 2), (2, 2), init_mode, 'relu', 0, False, down0)
        segmentar = Conv2D(1, 1, activation = 'sigmoid')(up0)
        modelo = Model(inputs = inputs, outputs = segmentar)
        if optimizer == "Adam":
            modelo.compile(optimizer = Adam(lr=2e-4), loss = self.bce_dice_loss, metrics = [self.dice_coef, 'accuracy'])
        elif optimizer == "SGD":
            modelo.compile(optimizer = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True), loss = 'binary_crossentropy', metrics = ['accuracy'])
        elif optimizer == "RMSprop":
            modelo.compile(optimizer = RMSprop(lr=0.001, rho=0.9), loss = 'binary_crossentropy', metrics = ['accuracy'])
        elif optimizer == "Adagrad":
            modelo.compile(optimizer = Adagrad(lr=0.01), loss = 'binary_crossentropy', metrics = ['accuracy'])
        elif optimizer == "Adadelta":
            modelo.compile(optimizer = Adadelta(lr=1.0, rho=0.95), loss = 'binary_crossentropy', metrics = ['accuracy'])
        elif optimizer == "Adamax":
            modelo.compile(optimizer = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999), loss = 'binary_crossentropy', metrics = ['accuracy'])
        elif optimizer == "Nadam":
            modelo.compile(optimizer = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999), loss = 'binary_crossentropy', metrics = ['accuracy'])
        if resumen:
            print("*** GUARDANDO RESUMEN DE MODELO NEURONAL CONVOLUCIONAL ***")
            with open(self.config.RUTA_GUARDAR_MODELO + '\\resumen_modelo.txt', 'w') as f:
                with redirect_stdout(f):
                    modelo.summary()
        return modelo
    def cargar_modelo(self, ruta_modelo, optimizador, filtro, dropout, initializer, resumen):
        modelo = self.__crea_modelo__((None, None, 1), optimizador, filtro, dropout, initializer, resumen)
        modelo.load_weights(ruta_modelo + '\\modelo_cnn.hdf5')
        return modelo
    def bce_dice_loss(self, y_true, y_pred):
        loss = binary_crossentropy(y_true, y_pred) + self.dice_coef_loss(y_true, y_pred)
        return loss
    def guardar_imagenes_k_fold(self, i, imagenes_entrenamiento, mascaras_entrenamiento, imagenes_validar, mascaras_validar):
        print("*****************************************")
        print("Guardando imágenes de entrenamiento")
        print("Número de imágenes de entrenamiento: " + str(len(imagenes_entrenamiento)))
        for ruta_img in imagenes_entrenamiento:
            img = self.imagenobj.leer_TIFF(ruta_img)
            nombreimg = ruta_img.replace(os.getcwd() + "\\" + self.config.RUTA_IMAGENES + '\\', "")
            io.imsave(self.config.ENTRENAMIENTO_KFOLD + self.config.RUTA_IMAGENES + "\\" + nombreimg, img)
        print("Número de máscaras (entrenamiento): " + str(len(mascaras_entrenamiento)))
        for ruta_img in mascaras_entrenamiento:
            img = self.imagenobj.leer_TIFF(ruta_img)
            nombreimg = ruta_img.replace(os.getcwd() + "\\"  + self.config.RUTA_MASCARAS + '\\', "")
            io.imsave(self.config.ENTRENAMIENTO_KFOLD + self.config.RUTA_MASCARAS + "\\" + nombreimg, img)
        print("Guardando imágenes para validación")
        print("Número de imágenes de validacion: " + str(len(imagenes_validar)))
        for ruta_img in imagenes_validar:
            img =  self.imagenobj.leer_TIFF(ruta_img)
            nombreimg = ruta_img.replace(os.getcwd() + "\\" + self.config.RUTA_IMAGENES + '\\', "")
            io.imsave(self.config.ENTRENAMIENTO_KFOLD + self.config.RUTA_VALIDAR_IMG + "\\" + nombreimg, img)
        print("Número de máscaras (validación) : " + str(len(mascaras_validar)))
        for ruta_img in mascaras_validar:
            img =  self.imagenobj.leer_TIFF(ruta_img)
            nombreimg = ruta_img.replace(os.getcwd() + "\\" + self.config.RUTA_MASCARAS + '\\', "")
            io.imsave(self.config.ENTRENAMIENTO_KFOLD + self.config.RUTA_VALIDAR_MAS + "\\" + nombreimg, img)
        print("*****************************************")
    def crea_directorio(self, ruta):
        try:
            os.stat(ruta)
        except:
            os.mkdir(ruta)
    def histograma(self, grayscale):
        counts, vals = np.histogram(grayscale, bins=range(2 ** 8))
        plt.plot(range(0, (2 ** 8) - 1), counts)
        plt.title("Grayscale image histogram")
        plt.xlabel("Pixel intensity")
        plt.ylabel("Count")
        plt.savefig(self.config.RUTA_GUARDAR_GRAFICAS + "/plot_histograma.png")
        plt.show()
        plt.close()
    
    def prediccion_a_img(self, prediccion, porcentaje):
        imgbool = prediccion.astype('float')
        imgbool[imgbool > porcentaje] = 1
        imgbool[imgbool <= porcentaje] = 0
        imagen = imgbool.astype('uint16')
        imagen = imagen * 65535
        imagen.shape = prediccion.shape
        return imagen, imgbool
    def elimina_estrellas(self, nombre, imgoriginal, ruta_guardar, imgbool, porcentaje):
        imsinestrellas = imgoriginal * (1 - imgbool)
        imsinestrellas.shape = imgoriginal.shape
        imsinestrellas = imsinestrellas.astype('uint16')
        return ruta_guardar + "/" + nombre + "_sin_estrellas_" + str(porcentaje) + ".tif", imsinestrellas
    def prediccion_imagen_rgb(self, nombre, imgoriginal, imgbool, ruta_guardar, color_rgba, porcentaje):
        h, w = imgbool.shape
        imoriginal = Image.new('RGBA', (h, w), (0,0,0,0))
        estrellas = Image.new('RGBA', (h, w), (0, 0, 0, 0))
        imgrgba = np.zeros((h,w,4))
        imguint = imgbool.astype('uint8')
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
        estrellas = Image.fromarray(np.uint8(imgrgba))
        imoriginal = Image.fromarray(np.uint16(imgoriginal))
        imagenconestrellas = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        imagenconestrellas.paste(imoriginal, box = (0, 0))
        imagenconestrellas.paste(estrellas, box = (0, 0), mask=estrellas)
        imagenconestrellas.save(ruta_guardar + "/" + nombre + "_predict_rgb_" + str(porcentaje) + ".tif")
        return estrellas
    def ordenar_alfanumerico(self, lista):
        import re
        convertir = lambda texto: int(texto) if texto.isdigit() else texto.lower()
        alphanum_key = lambda key: [ convertir(c) for c in re.split('([0-9]+)', key) ] 
        return sorted(lista, key=alphanum_key)
    def guardar_resultado(self, ruta_guardar,npimg, folder_imagenes, lista_imagenes, estrellas):
        lista_imagenes = self.ordenar_alfanumerico(lista_imagenes)
        for i,item in enumerate(npimg):
            imgpredict = item[:,:,0]
            img = imgpredict * 65535
            simg = img.astype('uint16')
            rutaoriginal = lista_imagenes[i]
            nombre = rutaoriginal.replace(folder_imagenes + "\\", "")
            nombre = nombre.replace(".tif", "")
            print("Ruta: " + ruta_guardar)
            print("Nombre: " + nombre)
            io.imsave(os.path.join(ruta_guardar,nombre + "_predict.tif"),simg)
            imgint20, imgbool20 = self.prediccion_a_img(imgpredict, 0.2)
            io.imsave(os.path.join(ruta_guardar,nombre + "_predict_20.tif"),imgint20)
            imgint30, imgbool30 = self.prediccion_a_img(imgpredict, 0.3)
            io.imsave(os.path.join(ruta_guardar,nombre + "_predict_30.tif"),imgint30)
            imgint50, imgbool = self.prediccion_a_img(imgpredict, 0.5)
            io.imsave(os.path.join(ruta_guardar,nombre + "_predict_50.tif"),imgint50)
            imgint70, imgbool70 = self.prediccion_a_img(imgpredict, 0.7)
            io.imsave(os.path.join(ruta_guardar,nombre + "_predict_70.tif"),imgint70)
            imgint80, imgbool80 = self.prediccion_a_img(imgpredict, 0.8)
            io.imsave(os.path.join(ruta_guardar,nombre + "_predict_80.tif"),imgint80)
            imgint90, imgbool90 = self.prediccion_a_img(imgpredict, 0.9)
            io.imsave(os.path.join(ruta_guardar,nombre + "_predict_90.tif"),imgint90)
            
            self.imagenobj = Imagen()
            imoriginal = Image.open(rutaoriginal)
            imgoriginal = np.array(imoriginal)
            imgoriginal.shape = imgpredict.shape
            imgoriginal = imgoriginal.astype('uint16')
            #20% probabilidad
            estrellas20 = self.prediccion_imagen_rgb(nombre, imgoriginal, imgbool20, ruta_guardar, [141, 9, 118, 128], 20)
            #30% probabilidad
            estrellas30 = self.prediccion_imagen_rgb(nombre, imgoriginal, imgbool30, ruta_guardar, [22, 124, 243, 128], 30)
            #50% probabilidad
            estrellas50 = self.prediccion_imagen_rgb(nombre, imgoriginal, imgbool, ruta_guardar, [245, 138, 25, 128], 50)
            #70% probabilidad
            estrellas70 = self.prediccion_imagen_rgb(nombre, imgoriginal, imgbool70, ruta_guardar, [234, 51, 11, 128], 70)
            #90% probabilidad
            estrellas80 = self.prediccion_imagen_rgb(nombre, imgoriginal, imgbool80, ruta_guardar, [0, 128, 0, 128], 80)
            #90% probabilidad
            estrellas90 = self.prediccion_imagen_rgb(nombre, imgoriginal, imgbool90, ruta_guardar, [255, 255, 0, 128], 90)
            h, w = imgoriginal.shape
            imoriginal = Image.new('RGBA', (h, w), (0,0,0,0))
            imoriginal = Image.fromarray(np.uint16(imgoriginal))
            imagenconestrellas = Image.new('RGBA', (w, h), (0, 0, 0, 0))
            imagenconestrellas.paste(imoriginal, box = (0, 0))
            imagenconestrellas.paste(estrellas20, box = (0, 0), mask=estrellas20)
            imagenconestrellas.paste(estrellas30, box = (0, 0), mask=estrellas30)
            imagenconestrellas.paste(estrellas50, box = (0, 0), mask=estrellas50)
            imagenconestrellas.paste(estrellas70, box = (0, 0), mask=estrellas70)
            imagenconestrellas.paste(estrellas80, box = (0, 0), mask=estrellas80)
            imagenconestrellas.paste(estrellas90, box = (0, 0), mask=estrellas90)
            imagenconestrellas.save(ruta_guardar + "/" + nombre + "_predict_rgb_todas.tif")
            
    def guardar_graficas(self, i, hist, ruta_guardar):
        accuracy = hist.history['acc']
        val_accuracy = hist.history['val_acc']
        dice_coef = hist.history['dice_coef']
        val_dice_coef = hist.history['val_dice_coef']
        loss = hist.history['loss']
        val_loss = hist.history['val_loss']
        #dice_coef_loss = hist.history['dice_coef_loss']
        #val_dice_coef_loss = hist.history['val_dice_coef_loss']
        rango_epochs = range(len(accuracy))
        plt.plot(rango_epochs, accuracy, 'bo', label='Training accuracy')
        plt.plot(rango_epochs, val_accuracy, 'b', label='Validation accuracy')
        plt.plot(rango_epochs, dice_coef, 'ro', label='Training dice coef')
        plt.plot(rango_epochs, val_dice_coef, 'r', label='Validation dice coef')
        plt.ylim(0, 1)
        plt.title('Training accuracy and dice coefficient')
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("accuracy and dice_coefficient")
        plt.savefig(ruta_guardar + "/plot_accuracy_dice_coefficient_kfold_" + str(i) + ".png")
        plt.show()
        
        plt.plot(rango_epochs, accuracy, 'bo', label='Training accuracy')
        plt.plot(rango_epochs, val_accuracy, 'b', label='Validation accuracy')
        plt.ylim(0, 1)
        plt.title('Training accuracy')
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("accuracy and dice_coefficient")
        plt.savefig(ruta_guardar + "/plot_accuracy_kfold_" + str(i) + ".png")
        plt.show()
        
        plt.plot(rango_epochs, dice_coef, 'ro', label='Training dice coef')
        plt.plot(rango_epochs, val_dice_coef, 'r', label='Validation dice coef')
        plt.ylim(0, 1)
        plt.title('Training dice coefficient')
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Dice_coefficient")
        plt.savefig(ruta_guardar + "/plot_dice_coefficient_kfold_" + str(i) + ".png")
        plt.show()
        
        plt.plot(rango_epochs, loss, 'bo', label='Training dice coef loss')
        plt.plot(rango_epochs, val_loss, 'b', label='Validation dice coef loss')
        plt.plot(np.argmin(hist.history["val_loss"]), np.min(hist.history["val_loss"]), marker="x", color="r", label="Best model")
        plt.title("Learning curve")
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.savefig(ruta_guardar + "/plot_dice_coef_loss_kfold_" + str(i) + ".png")
        plt.show()
        
    def fit_generador(self):
        batch_size = 6
        for e in self.lista_epochs:
            for o in self.lista_optimizador:
                for ini in self.lista_init_mode:
                    for f in self.lista_filtro:
                        for d in self.lista_dropout:
                            i = 0
                            self.config = Config(e, o, ini, f, d, 1)
                            self.imagenobj = Imagen()
                            lista_imagenes = sorted(list(paths.list_images(os.getcwd() + "\\" + self.config.RUTA_IMAGENES)))
                            lista_mascaras = sorted(list(paths.list_images(os.getcwd() + "\\" + self.config.RUTA_MASCARAS)))
                            k_folds = 10
                            kf = KFold(n_splits = k_folds, random_state = 42, shuffle = True)
                            X = np.array(lista_imagenes)
                            y = np.array(lista_mascaras)
                            print("Parametros: Epochs: " + str(e) + " Optimizador: " + str(o) + " Init mode: " + str(ini) + " filtro " + str(f) + " dropout " + str(d))
                            if i > 0:
                                self.config.config_directorios(e)
                            modelo = self.__crea_modelo__((None, None, 1), o, f, d, ini, True)
                            k = 1
                            for entrenamiento_x, validar_y in kf.split(X):
                                imagenes_entrenamiento = X[entrenamiento_x]
                                imagenes_validar = X[validar_y]
                                mascaras_entrenamiento = y[entrenamiento_x]
                                mascaras_validar = y[validar_y]
                                self.config.config_directorios_k_fold(k, o)
                                self.guardar_imagenes_k_fold(k, imagenes_entrenamiento, mascaras_entrenamiento, imagenes_validar, mascaras_validar)
                                try: 
                                    modelo.load_weights(self.config.FOLDER_MODELO + '\\modelo_cnn.hdf5')
                                except Exception as OSError:
                                    pass
                                csv = CSVLogger(self.config.RUTA_GUARDAR_RESULTADOS + '\\entrenamiento_kfold_' + str(k) + o + '.log')
                                tb = TensorBoard(log_dir=self.config.RUTA_GUARDAR_RESULTADOS + '\\kfold_' + str(k) + '_' + o + '\\', histogram_freq=0, write_graph=True, write_images=True)
                                checkpoint = ModelCheckpoint(self.config.RUTA_GUARDAR_MODELO + '\\modelo_cnn.hdf5', monitor='val_loss',verbose=2, save_best_only=True)
                                steps_per_epoch = len(imagenes_entrenamiento) // batch_size
                                val_steps = len(imagenes_validar) // batch_size
                                aumentar = Aumentar()
                                imagenes = sorted(list(paths.list_images(self.config.ENTRENAMIENTO_KFOLD + "\\" + self.config.RUTA_IMAGENES + "\\" )))
                                mascaras = sorted(list(paths.list_images(self.config.ENTRENAMIENTO_KFOLD + "\\" + self.config.RUTA_MASCARAS + "\\")))
                                entrenar_gen = aumentar.generador_aumentar(batch_size, imagenes, mascaras, self.config.RUTA_GUARDAR_AUMENTADAS, self.config.RUTA_GUARDAR_AUMENTADAS_MAS, self.config.TAMANIO_IMAGENES_X, self.config.TAMANIO_IMAGENES_Y, aumentar = True)
                                imagenes = sorted(list(paths.list_images(self.config.ENTRENAMIENTO_KFOLD + "\\" + self.config.RUTA_VALIDAR_IMG + "\\")))
                                mascaras = sorted(list(paths.list_images(self.config.ENTRENAMIENTO_KFOLD + "\\" + self.config.RUTA_VALIDAR_MAS + "\\")))
                                validar_gen = aumentar.generador_aumentar(batch_size, imagenes, mascaras, self.config.RUTA_GUARDAR_AUMENTADAS_VAL, self.config.RUTA_GUARDAR_AUMENTADAS_VAL_MAS, self.config.TAMANIO_IMAGENES_X, self.config.TAMANIO_IMAGENES_Y, aumentar = False)
                                #validation_data = list(validar_gen)
                                #performance_cbk = PerformanceVisualizationCallback(model=modelo, validation_data=validation_data, image_dir=self.config.RUTA_GRAFICAS)
                                hist = modelo.fit_generator(entrenar_gen, validation_data=validar_gen, validation_steps=val_steps, 
                                                            steps_per_epoch=steps_per_epoch, epochs=e, callbacks=[checkpoint, csv, tb])
                                self.guardar_graficas(k, hist, self.config.RUTA_GUARDAR_GRAFICAS)
                                k += 1
                            i += 1