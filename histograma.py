# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 10:14:37 2021

@author: eduardo
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
dir_imagenes = "pruebas/imagenes_tif"
dir_resultado = "pruebas/imagenes_tif_histograma"
imagenes = os.listdir(dir_imagenes)
for img in imagenes:
    imagen = cv2.imread(dir_imagenes + "/" + img, -1)
    nombre = img.replace('_', " ")
    nombre = nombre.replace('original.png', 'con ajuste de contraste')
    nombre = nombre.replace(".png", "")
    nombre = nombre.replace("Ha 2", "")
    nombre = nombre.replace("Ha 4 1", "")
    plt.hist(imagen.ravel(), bins = 50, range = [0, 65535], fc='k', ec='k')
    plt.title("Histograma de " + nombre)
    plt.xlabel("Intensidad de pixel")
    plt.ylabel("Número de pixeles")
    plt.savefig(dir_resultado + "/" + img + "_histograma.png")
    plt.show()
    plt.close()
    