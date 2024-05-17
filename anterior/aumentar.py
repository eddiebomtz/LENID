# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 19:02:07 2020

@author: eduardo
"""
import os
import cv2
import random
import numpy as np
import skimage.io as io
from imutils import paths
from imagen import Imagen
from preprocesamiento import preprocesamiento
class Aumentar:
    def __init__(self):
        self.num_imagenes = 1
    def __shift__(self, imagen, tipo, ratio = 0.0):
        if ratio > 1 or ratio < 0:
            print('El valor del ratio debe de estar entre 0 y 1')
            return imagen
        ratio = random.uniform(-ratio, ratio)
        h, w = imagen.shape[:2]
        #Horizontal
        if tipo == 1:
            to_shift = w * ratio
            imagen = imagen[:, :int(w-to_shift)]
            imagen = imagen[:, int(-1*to_shift):]
        #Vertical
        elif tipo == 2:
            to_shift = h * ratio
            imagen = imagen[:int(h-to_shift), :]
            imagen = imagen[int(-1*to_shift):, :]
        imagen = cv2.resize(imagen, (h, w), cv2.INTER_CUBIC)
        return imagen
    def __brillo_contraste__(self, imagen, b, c):
        nueva_imagen = imagen.copy()
        for y in range(imagen.shape[0]):
            for x in range(imagen.shape[1]):
                nueva_imagen[y,x] = np.clip(c * imagen[y,x] + b, 0, 65535)
        return nueva_imagen
    def __zoom_in__(self, imagen, mascara):
        height, width = imagen.shape
        zoom_pix = random.randint(0, 10)
        zoom_factor = 1 + (2 * zoom_pix) / height
        imagen = cv2.resize(imagen, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_CUBIC)
        top_crop = (imagen.shape[0] - height) // 2
        left_crop = (imagen.shape[1] - width) // 2
        imagen = imagen[top_crop: top_crop + height, left_crop: left_crop + width]
        mascara = cv2.resize(mascara, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_CUBIC)
        top_crop = (mascara.shape[0] - height) // 2
        left_crop = (mascara.shape[1] - width) // 2
        mascara = mascara[top_crop: top_crop + height, left_crop: left_crop + width]
        return imagen, mascara
    def __flip_vertical__(self, imagen):
        imagen = cv2.flip(imagen, 0)
        return imagen
    def __flip_horizontal__(self, imagen):
        imagen = cv2.flip(imagen, 1)
        return imagen
    def __flip_ambos__(self, imagen):
        imagen = cv2.flip(imagen, -1)
        return imagen
    def get_num_imagenes(self):
        return self.num_imagenes
    def obtener_imagen(self, indice, ruta_imagen, ruta_mascaras, ruta_guardar, ruta_guardar_mas, aumentar):
        carpetas = ruta_imagen[indice].split('\\')
        nombre = carpetas[len(carpetas)-1]
        nombre = nombre.replace(".tif", "")
        ruta_img = ruta_imagen[indice]
        ruta_mas = ruta_mascaras[indice]
        imagen = cv2.imread(ruta_img, -1)
        self.pp = preprocesamiento(ruta_imagen[indice], True)
        '''sigma = 2.5
        if aumentar:
            imagen = self.pp.anisodiff(self.pp.imagenprocesada,100,80,0.075,(1,1),sigma,1)'''
        x, y = imagen.shape
        mascara = cv2.imread(ruta_mas, -1)
        texto = ""
        if aumentar:
            self.flip_vertical = random.randint(0, 1)
            self.flip_horizontal = random.randint(0, 1)
            self.flip_ambos = random.randint(0, 1)
            self.zoom_in = random.randint(0, 1)
            #self.shift_horizontal = random.randint(0, 1)
            #self.shift_vertical = random.randint(0, 1)
            #self.brillo_contraste = random.randint(0, 1)
            if self.flip_vertical == 1:
                imagen = self.__flip_vertical__(imagen)
                mascara = self.__flip_vertical__(mascara)
                texto += "_flip_vertical"
                #imagen, mascara = self.__zoom_in__(imagen, mascara)
                #texto += "_zoom_in"
            if self.flip_horizontal == 1:
                imagen = self.__flip_horizontal__(imagen)
                mascara = self.__flip_horizontal__(mascara)
                texto += "_flip_horizontal"
                #imagen, mascara = self.__zoom_in__(imagen, mascara)
                #texto += "_zoom_in"
            if self.flip_ambos == 1:
                imagen = self.__flip_ambos__(imagen)
                mascara = self.__flip_ambos__(mascara)
                texto += "_flip_ambos"
                #imagen, mascara = self.__zoom_in__(imagen, mascara)
                #texto += "_zoom_in"
            if self.zoom_in == 1:
                imagen, mascara = self.__zoom_in__(imagen, mascara)
                texto += "_zoom_in"
            '''if self.shift_horizontal == 1:
                imagen = self.__shift__(imagen, 1, ratio = 0.2)
                mascara = self.__shift__(mascara, 1, ratio = 0.2)
            if self.shift_vertical == 1:
                imagen = self.__shift__(imagen, 2, ratio = 0.2)
                mascara = self.__shift__(mascara, 2, ratio = 0.2)
            if self.brillo_contraste == 1:
                #brillo de 0 a 100
                #contraste de 1 al 3
                imagen = self.__brillo_contraste__(imagen, b = 20, c = 1.5)
                mascara = self.__brillo_contraste__(mascara, b = 20, c = 1.5)'''
            if texto == "":
                imagen, mascara = self.__zoom_in__(imagen, mascara)
                texto += "_zoom_in"
                self.flip_horizontal = random.randint(0, 1)
                if self.flip_horizontal == 1:
                    imagen = self.__flip_horizontal__(imagen)
                    mascara = self.__flip_horizontal__(mascara)
                    texto += "_flip_horizontal"
                else:
                    imagen = self.__flip_vertical__(imagen)
                    mascara = self.__flip_vertical__(mascara)
                    texto += "_flip_vertical"
            #cv2.imwrite(ruta_guardar + '/' + nombre + "_" + str(self.num) + "_" + texto + '.tif', imagen)
            #cv2.imwrite(ruta_guardar_mas + '/' + nombre + "_" + str(self.num) + "_" + texto + '.tif', mascara)
            self.num_imagenes += 1
        '''else:
            cv2.imwrite(ruta_guardar + '/' + nombre + "_" + str(self.num) + "_" + texto + '.tif', imagen)
            if ruta_mascaras != "":
                cv2.imwrite(ruta_guardar_mas + '/' + nombre + "_" + str(self.num) + "_" + texto + '.tif', mascara)'''
        imagen = imagen.reshape(x, y, 1)
        mascara = mascara.reshape(x, y, 1)
        return imagen, mascara
    def generador_aumentar(self, tam_lote, ruta_originales, ruta_mascaras, ruta_guardar, ruta_guardar_mas, tam_img_x, tam_img_y, aumentar = True):
        while True:
            indices = np.random.permutation(len(ruta_originales))
            for lote in range(0, len(indices), tam_lote):
                lote_actual = indices[lote:(lote + tam_lote)]
                imagenes = np.empty([0, tam_img_y, tam_img_x, 1], dtype=np.float32)
                mascaras = np.empty([0, tam_img_y, tam_img_x, 1], dtype=np.float32)
                for i in lote_actual:
                    imagen, mascara = self.obtener_imagen(i, ruta_originales, ruta_mascaras, ruta_guardar, ruta_guardar_mas, aumentar = aumentar)
                    imagen = (imagen - imagen.min()) / (imagen.max() - imagen.min())
                    mascara = mascara / mascara.max()
                    mascara[mascara > 0.5] = 1
                    mascara[mascara <= 0.5] = 0
                    '''print(imagen.shape)
                    print(mascara.shape)
                    print(imagenes.shape)
                    print(mascaras.shape)'''
                    imagenes = np.append(imagenes, [imagen], axis=0)
                    mascaras = np.append(mascaras, [mascara], axis=0)
                yield (imagenes, mascaras)
    def crea_directorio(self, ruta):
        try:
            os.stat(ruta)
        except:
            os.mkdir(ruta)
    def ordenar_alfanumerico(self, lista):
        import re
        convertir = lambda texto: int(texto) if texto.isdigit() else texto.lower()
        alphanum_key = lambda key: [ convertir(c) for c in re.split('([0-9]+)', key) ] 
        return sorted(lista, key=alphanum_key)
    def generador_pruebas(self, ruta, ruta_guardar, objetos_extendidos):
        imagenobj = Imagen()
        lista_imagenes = self.ordenar_alfanumerico(list(paths.list_images(ruta)))
        for i,item in enumerate(lista_imagenes):
            #imagen = cv2.imread(item, -1)
            imagen = imagenobj.leer_TIFF(item)
            nombre = item.replace(ruta + "\\", "")
            nombre = nombre.replace(".tif", "")
            #self.crea_directorio(ruta_guardar + "/original/")
            io.imsave(os.path.join(ruta_guardar, nombre + ".tif"), imagen)
            pp = preprocesamiento(item, True)
            pp.autocontraste(2, True)
            pp.guardar_imagen_tif(ruta_guardar, nombre + "_percentile_range.tif", True)
            #if objetos_extendidos:
                #pp.pfcm_3()
                #pp.guardar_imagen_tif(ruta_guardar, nombre + "_percentile_range_pfcm.tif", True)
            #imagen = pp.imagenprocesada
            imagen = (imagen - imagen.min()) / (imagen.max() - imagen.min())
            imagen = np.reshape(imagen,imagen.shape+(1,))
            imagen = np.reshape(imagen,(1,)+imagen.shape)
            yield imagen