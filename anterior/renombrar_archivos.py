# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 21:00:13 2020

@author: eduardo
"""

import os
from imagen import Imagen
imagenobj = Imagen()
for count, ruta_nombre in enumerate(os.listdir("entrenamiento/fits")): 
    imagen, header = imagenobj.leer_fits("entrenamiento/fits/" + ruta_nombre)
    run = header["run"]
    imageid = header["imageid"]
    if not ruta_nombre.startswith("r"):
        print("Renombrando: " + str(run) + "-" + str(imageid) + "_" + ruta_nombre)
        nueva_ruta_nombre = "r" + str(run) + "-" + str(imageid) + "_" + ruta_nombre
        os.rename("entrenamiento/fits/" + ruta_nombre, "entrenamiento/fits/" + nueva_ruta_nombre) 
    else:
        continue
    #dst ="Hostel" + str(count) + ".jpg"
    #src ='xyz'+ filename 
    #dst ='xyz'+ dst 
    #os.rename(src, dst) 