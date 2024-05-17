# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 21:00:13 2020

@author: eduardo
"""

import os
from image import Image
imageobj = Image()
for count, path_name in enumerate(os.listdir("training/fits")): 
    image, header = imageobj.read_fits("training/fits/" + path_name)
    run = header["run"]
    imageid = header["imageid"]
    if not path_name.startswith("r"):
        print("Renombrando: " + str(run) + "-" + str(imageid) + "_" + path_name)
        nueva_path_name = "r" + str(run) + "-" + str(imageid) + "_" + path_name
        os.rename("training/fits/" + path_name, "training/fits/" + nueva_path_name) 
    else:
        continue
    #dst ="Hostel" + str(count) + ".jpg"
    #src ='xyz'+ filename 
    #dst ='xyz'+ dst 
    #os.rename(src, dst) 