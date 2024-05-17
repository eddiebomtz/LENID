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
dir_images = "test/images_prueba_pfcm"
dir_expected = "test/images_prueba_pfcm_esperado"
f = open (dir_expected + '/evaluation_dice_pfcm.txt', 'wt')
images = os.listdir(dir_images)
for img in images:
    img_pred = Image.open(dir_images + "/" + img)
    img_pred = np.array(img_pred)
    img_pred = img_pred / 65533
    name = img.replace("_prediction.png", "_mask.png")
    image_true = Image.open(dir_expected + "/" + name)
    image_true = np.array(image_true)
    image_true = image_true / 65533
    result = 'image ' + img + ' ' + str(dice(img_pred, image_true))
    print(result)
    f.write(result + "\n")
f.close()