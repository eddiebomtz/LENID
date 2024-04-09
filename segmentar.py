# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 10:48:09 2019

@author: eduardo
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from os import listdir
from conv import Conv
from imagen import Imagen
from aumentar import Aumentar
from imutils import paths
imagen = Imagen()
aumentar = Aumentar()
conv = Conv(None, None, None, None)
batch_size = 1
estrellas = False
objetos_extendidos = True
IMAGENES_PRUEBAS = os.getcwd() + '\\pruebas'
FOLDER_IMAGENES = IMAGENES_PRUEBAS + '/tif_pfcm'
FOLDER_MODELO = os.getcwd() + '/modelo'
FOLDER_RESULTADOS = os.getcwd() + '/resultados'
lista_imagenes = sorted(list(paths.list_images(FOLDER_IMAGENES)))
num_imagenes = len(lista_imagenes)
print("Número de imágenes de prueba: " + str(num_imagenes))
lista_resultados = sorted(list(listdir(FOLDER_RESULTADOS)))
for i, carpeta in enumerate(lista_resultados):
    imagen.crea_directorio(FOLDER_RESULTADOS + "/" + carpeta + "/prediccion")
    folder_modelo = FOLDER_RESULTADOS.replace(FOLDER_RESULTADOS, FOLDER_MODELO)
    lista_parametros = carpeta.split("_")
    epochs = lista_parametros[0]
    optimizador = lista_parametros[7]
    if lista_parametros[8] == "lecun" or lista_parametros[8] == "glorot" or lista_parametros[8] == "he":
        initializer = lista_parametros[8] + "_" + lista_parametros[9]
        dropout = lista_parametros[10]
    else:
        initializer = lista_parametros[8]
        dropout = lista_parametros[9]

    modelo = conv.cargar_modelo(folder_modelo + "/" + carpeta, optimizador, dropout, initializer, False)
    test_gen = aumentar.generador_pruebas(FOLDER_IMAGENES, FOLDER_RESULTADOS + "/" + carpeta + "/prediccion")
    predecir_generador = modelo.predict_generator(test_gen, num_imagenes, verbose=2)
    conv.guardar_resultado(FOLDER_RESULTADOS + "/" + carpeta + "/prediccion", predecir_generador, FOLDER_IMAGENES, lista_imagenes)