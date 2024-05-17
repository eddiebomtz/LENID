# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 10:14:37 2021

@author: eduardo
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
dir_images = "test/images_tif"
dir_result = "test/images_tif_histogram"
images = os.listdir(dir_images)
for img in images:
    image = cv2.imread(dir_images + "/" + img, -1)
    name = img.replace('_', " ")
    name = name.replace('original.png', 'with contrast adjustment')
    name = name.replace(".png", "")
    name = name.replace("Ha 2", "")
    name = name.replace("Ha 4 1", "")
    plt.hist(image.ravel(), bins = 50, range = [0, 65535], fc='k', ec='k')
    plt.title("Histogram of " + name)
    plt.xlabel("Píxel intensity")
    plt.ylabel("Number of pixels")
    plt.savefig(dir_result + "/" + img + "_histogram.png")
    plt.show()
    plt.close()
    