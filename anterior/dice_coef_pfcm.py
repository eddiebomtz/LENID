# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 16:51:12 2021

@author: eduardo
"""

import os
import numpy as np
from PIL import Image
def dice(pred, true):
    overlap = np.logical_and(true, pred)
    dice = np.sum(overlap)*2 / (np.sum(true)+ np.sum(pred))
    return dice
dir_imagenes = "pruebas/imagenes_prueba_pfcm"
dir_esperado = "pruebas/imagenes_prueba_pfcm_esperado"
f = open (dir_esperado + '/evaluacion_dice_pfcm.txt','wt')
imagenes = os.listdir(dir_imagenes)
for img in imagenes:
    imgpred = Image.open(dir_imagenes + "/" + img)
    imgpred = np.array(imgpred)
    imgpred = imgpred / 65533
    nombre = img.replace("_prediccion.png", "_mascara.png")
    imagentrue = Image.open(dir_esperado + "/" + nombre)
    imagentrue = np.array(imagentrue)
    imagentrue = imagentrue / 65533
    resultado = 'Imagen ' + img + ' ' + str(dice(imgpred, imagentrue))
    print(resultado)
    f.write(resultado + "\n")
f.close()