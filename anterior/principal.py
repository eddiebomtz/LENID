# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 10:48:09 2019

@author: eduardo
"""
import os
import argparse
from os import listdir
from conv import Conv
from imagen import Imagen
from aumentar import Aumentar
from preprocesamiento import preprocesamiento
from imutils import paths
parser = argparse.ArgumentParser(description='Segmentación de objetos extendidos.')
parser.add_argument("-d" , "--dir_imagenes", action="store", dest="dir_imagenes", help="directorio de entrada")
parser.add_argument("-r", "--dir_results", action="store", dest="dir_resultado", help="directorio de salida")
parser.add_argument("-t", "--train", action="store_true", help="Especifica si está en modo para entrenar el modelo")
#parser.add_argument("-re", "--reanudar", action="store_true", help="Especifica si está en modo de reanudar el entrenamiento del modelo")
#parser.add_argument("-k", "--kfold", action="store", dest="kf", help="Especifica un numero entero para el número de k fold en el que se quedó el entrenamiento.")
parser.add_argument("-s", "--segment", action="store_true", help="Especifica si está en modo para segmentar las imágenes de prueba, tomando como base el modelo previamente creado")
parser.add_argument("-o", "--extended", action="store_true", help="Especifica si está utilizando el programa para segmentación de objetos extendidos, debe utilizarse junto con -t o -s")
args = parser.parse_args()
#parser.print_help()
if args.entrenar:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print("Entrenar...")
    lista_epochs = [30]
    lista_dropout = [0.2, 0.4]
    print("Objetos extendidos...")
    lista_optimizador = ['Adam']
    lista_init_mode = ['he_normal']
    lista_filtro = [3]
    if args.reanudar:
        kf = int(args.kf)
        conv = Conv(None, None, None, None, None)
        conv.fit_generador_reanudar(kf)
    else:
        conv = Conv(lista_epochs, lista_optimizador, lista_init_mode, lista_filtro, lista_dropout)
        conv.fit_generador()
elif args.segmentar:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print("Segmentar...")
    imagen = Imagen()
    aumentar = Aumentar()
    conv = Conv(None, None, None, None, None)
    batch_size = 1
    print("Objetos extendidos...")
    IMAGENES_PRUEBAS = os.getcwd() + '\\pruebas'
    FOLDER_IMAGENES = IMAGENES_PRUEBAS + '/' + args.dir_imagenes
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
            filtro = lista_parametros[10]
            dropout = lista_parametros[11]
        else:
            initializer = lista_parametros[8]
            filtro = lista_parametros[9]
            dropout = lista_parametros[10]
        modelo = conv.cargar_modelo(folder_modelo + "/" + carpeta, optimizador, filtro, dropout, initializer, False)
        test_gen = aumentar.generador_pruebas(FOLDER_IMAGENES, FOLDER_RESULTADOS + "/" + carpeta + "/prediccion", args.extendidos)
        predecir_generador = modelo.predict_generator(test_gen, num_imagenes, verbose=2)
        conv.guardar_resultado(FOLDER_RESULTADOS + "/" + carpeta + "/prediccion", predecir_generador, FOLDER_IMAGENES, lista_imagenes, args.estrellas)