# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 17:24:37 2020

@author: Eduardo
"""
import os
import numpy as np
from io import BytesIO
import skimage.io as io
from astropy.io import fits
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
class Imagen:
    def crea_directorio(self, ruta):
        try:
            os.stat(ruta)
        except:
            os.mkdir(ruta)
    def leer_TIFF(self, ruta):
        im = Image.open(ruta)
        out = np.array(im)
        img = out.astype('uint16')
        self.imagen = img
        return img
    def leer_fits(self, ruta):
        im = fits.open(ruta)
        header = im[1].header
        im = im[1].data
        out = np.array(im)
        out = out.astype('uint16')
        self.imagen = out
        return out, header
    def histograma(self, ruta_guardar, nombre, imagen, mascara):
        import cv2
        NBINS = 256
        #histogram, bin_edges = np.histogram(imagen, bins=NBINS, range=(0, 2000))
        imagen = imagen / imagen.max()
        mascara = mascara / mascara.max()
        mascara = mascara.astype("uint8")
        histogram = cv2.calcHist([imagen],[0],None,[NBINS],[0,1])
        hist_mascara = cv2.calcHist([imagen],[0],mascara,[NBINS],[0,1])
        plt.figure()
        plt.title("Grayscale Histogram of " + nombre)
        plt.xlabel("Grayscale value")
        plt.ylabel("Number of pixels")
        plt.xlim([0.0, 256.0])
        plt.plot(histogram)
        plt.plot(hist_mascara)
        plt.savefig(ruta_guardar + "/" + nombre + "_histograma.tif")
        plt.close()
    def recortar_imagenes(self, ruta_img_completa, ruta_guardar, tamcorte):
        import os
        from PIL import Image
        import skimage.io as io
        os.chdir(os.getcwd())
        imagenes = os.listdir(ruta_img_completa)
        for img in imagenes:
            imagenha = Image.open(ruta_img_completa + "/" + img)
            ancho, alto = imagenha.size
            contador = 0
            for i in range(0, alto, tamcorte):
                for j in range(0, ancho, tamcorte):
                    caja = (j, i, j + tamcorte, i + tamcorte)
                    cortar = imagenha.crop(caja)
                    cortar.save(ruta_guardar + "/" + img + "_corte_" + str(contador) + ".tif")
                    imagenrecortada = Image.open(ruta_guardar + "/" + img + "_corte_" + str(contador) + ".tif")
                    imagenrecortada = np.array(imagenrecortada)
                    imgint = imagenrecortada.astype('uint16')
                    io.imsave(ruta_guardar + "/" + img + "_corte_" + str(contador) + ".tif", imgint)
                    contador += 1
    def ordenar_alfanumerico(self, lista):
        import re
        convertir = lambda texto: int(texto) if texto.isdigit() else texto.lower()
        alphanum_key = lambda key: [ convertir(c) for c in re.split('([0-9]+)', key) ] 
        return sorted(lista, key=alphanum_key)
    def pega_imagenes(self, dir_imagenes, dir_guardar, ancho, alto, tam):
        import skimage.io as io
        self.crea_directorio(dir_guardar)
        lista_imagenes = self.ordenar_alfanumerico(os.listdir(dir_imagenes))
        i = 1
        ruta_imagenes_x = []
        imagenes_y = []
        for img in lista_imagenes:
            ruta_imagenes_x += [img]
            if i % (ancho / tam) == 0:
                concatenar_imagenes = Image.fromarray(
                  np.concatenate(
                    [np.array(Image.open(dir_imagenes + "/" + x)) for x in ruta_imagenes_x],
                    axis=1 #Concatena las imagenes en horizontal
                  )
                )
                imagenes_y += [concatenar_imagenes]
                ruta_imagenes_x = []
            if i % ((alto / tam) * (ancho / tam)) == 0:
                concatenar_imagenes = Image.fromarray(
                  np.concatenate(
                    [np.array(imagenes) for imagenes in imagenes_y],
                    axis=0 #Concatena las imagenes en vertical
                  )
                )
                concatenar_imagenes = np.array(concatenar_imagenes)
                concatenar_imagenes = concatenar_imagenes.astype('uint16')
                imagenes_y = []
                img = img.replace(".tif", "")
                io.imsave(os.path.join(dir_guardar, img + "_imagen_completa.tif"), concatenar_imagenes)
            i += 1
    def prediccion_a_imagen(self, ruta, ruta_guardar, porcentaje):
        from imutils import paths
        lista_imagenes = self.ordenar_alfanumerico(list(paths.list_images(ruta)))
        for i, item in enumerate(lista_imagenes):
            prediccion = self.leer_TIFF(item)
            prediccion = prediccion / 65535
            imgbool = prediccion.astype('float')
            imgbool[imgbool > porcentaje] = 1
            imgbool[imgbool <= porcentaje] = 0
            imagen = imgbool.astype('uint16')
            imagen = imagen * 65535
            imagen.shape = prediccion.shape
            io.imsave(os.path.join(ruta_guardar,  str(i) + "_estrellas_" + str(porcentaje) + ".tif"), imagen)
    def imagen_mascara(self, rutaoriginal, rutamascara, ruta_guardar):
        from imutils import paths
        lista_imagenes = self.ordenar_alfanumerico(list(paths.list_images(rutaoriginal)))
        for i, item in enumerate(lista_imagenes):
            imagen = self.leer_TIFF(item)
            nombremascara = item.replace(rutaoriginal + "\\", "")
            nombremascara = nombremascara.replace("interpolacion_1000.tif", "interpolacion_1000_predict_50.tif")
            mascara = self.leer_TIFF(os.path.join(rutamascara, nombremascara))
            imagen = imagen / 65535 * 255
            mascara = mascara / 65535
            sinfondo = imagen * mascara
            sinfondo = sinfondo.astype("uint8")
            io.imsave(os.path.join(ruta_guardar, nombremascara), sinfondo)
    def elimina_estrellas(self, rutaoriginal, rutapredict, ruta_guardar, porcentaje):
        from imutils import paths
        #from preprocesamiento import preprocesamiento
        lista_imagenes = self.ordenar_alfanumerico(list(paths.list_images(rutaoriginal)))
        for i, item in enumerate(lista_imagenes):
            imgoriginal = self.leer_TIFF(item)
            print(str(i) + item)
            img = item.replace(rutaoriginal + "\\", "")
            img = img.replace(".tif", "_predict.tif")
            print(str(i) + " " + rutapredict + "/" + img)
            imgbool = self.leer_TIFF(os.path.join(rutapredict, img))
            imgbool = imgbool / 65535
            #porcentaje_1_0 = porcentaje / 100
            imgbool[imgbool > porcentaje] = 1
            imgbool[imgbool <= porcentaje] = 0
            h, w = imgoriginal.shape 
            imsinestrellas = imgoriginal * (1 - imgbool)
            imsinestrellas.shape = imgoriginal.shape
            imsinestrellas = imsinestrellas.astype('uint16')
            img = item.replace(rutaoriginal + "\\", "") 
            img = img.replace(".tif.tif", "_predict.tif")
            print(str(i) + " " + ruta_guardar + "/" + img)
            io.imsave(os.path.join(ruta_guardar, img), imsinestrellas)
            '''pp = preprocesamiento(os.path.join(ruta_guardar,  str(i) + "_sin_estrellas_" + str(porcentaje) + ".tif"), True)
            pp.autocontraste(2, True)
            pp.guardar_imagen_tif(os.getcwd() + "/" + ruta_guardar, str(i) + "_sin_estrellas_" + str(porcentaje) + "_percentile_range.tif", True)'''
    def pegar_imagenes(self, dirimgs, dirguar, num_imgs, width):
        sumar = width
        self.crea_directorio(dirguar)
        imagenes = self.ordenar_alfanumerico(os.listdir(dirimgs))
        img = Image.new("RGB", (2048,4096))
        xv = 0
        yv = 0
        k = 1
        guardar = (num_imgs / ((2048 / width) * (4096 / width)) + 1)
        for i in range(1, num_imgs + 1, 1):
            im = self.leer_TIFF(dirimgs + "/" + imagenes[i - 1])
            if im.max() == 0:
                im = abs(im)
            else:
                im = im / im.max()
            for x in range(width):
                for y in range(width):
                    v = abs(im[x, y])
                    v = round(v * 255)
                    v = v.astype(int)
                    img.putpixel((xv + y, yv + x), (v, v, v))
            if i % (2048 / width) == 0:
                xv = 0
                yv = yv + sumar
            else:
                xv = xv + sumar
            if i % guardar == 0:
                xv = 0
                yv = 0
                img.save(dirguar + "/" + str(k) + "_completa.tif")
                k = k + 1
'''
import os
iobj = Imagen()
rutafits = "entrenamiento/mascaras_recortadas"
rutaimagenres = "entrenamiento/imagenes_recortadas_respaldo"
rutaimagen = "entrenamiento/imagenes_recortadas"
imagenes = os.listdir(rutafits)
for img in imagenes:
    imagen = Image.open(rutaimagenres + "/" + img)
    io.imsave(rutaimagen + "/" + img, imagen)

import os
iobj = Imagen()
rutafits = "entrenamiento/mascaras_recortadas"
rutaimagen = "entrenamiento/imagenes_recortadas"
rutaimgmask = "entrenamiento/imagenes_mascaras"
imagenes = os.listdir(rutafits)
#data = []
#imagesdata = []
for img in imagenes:
    #imagen = iobj.leer_TIFF(rutafits + "/" + img)
    imagen = Image.open(rutafits + "/" + img)
    #imagesdata.append(np.array(imagen))
    extrema = imagen.convert("L").getextrema()
    #data.append(extrema)
    print(img)
    if extrema == (0, 0): 
        print("Todos los pixeles son negros")
        imagen.close()
        os.remove(rutafits + "/" + img)
        os.remove(rutaimgmask + "/" + img)
        img2 = img.replace("_predict_50", "")
        os.remove(rutaimagen + "/" + img2)
    else:
        print("No todos los pixeles son negros")

iobj = Imagen()
ruta_imagen = "fits_g139_tif"
imagen_ha = iobj.leer_TIFF(ruta_imagen + "/r431413-1_PN_G139.0+03.2_Ha_2_600_x_600.tif")
imagen_i = iobj.leer_TIFF(ruta_imagen + "/r431415-1_PN_G139.0+03.2_i_2_600_x_600.tif")
imagen_ha_i = imagen_ha - imagen_i
io.imsave(ruta_imagen + "/r431414_5-1_ha_i.fits.tif", imagen_ha_i)

imagen_u = iobj.leer_TIFF(ruta_imagen + "/r763738-1_u.fits.tif")
imagen_g = iobj.leer_TIFF(ruta_imagen + "/r763739-1_g.fits.tif")
imagen_u_g = imagen_u - imagen_g
io.imsave(ruta_imagen + "/r763738_9-1_u_gfits.tif", imagen_u_g)

iobj = Imagen()
ruta_imagen = "fits_g139"
ruta_resultado = "fits_g139_tif"
imagenes = os.listdir(ruta_imagen)
for img in imagenes:
    imagen, header = iobj.leer_fits(ruta_imagen + "/" + img)
    io.imsave(ruta_resultado + "/" + img + ".tif", imagen)
iobj = Imagen()
ruta_entrenamiento = "entrenamiento/imagenes_sin_fondo_512_threshold_2000"
ruta_mascaras = "entrenamiento/mascaras_512_threshold_2000"
ruta_histograma = "entrenamiento/imagenes_sin_fondo_512_threshold_2000_histograma"
imagenes = os.listdir(ruta_entrenamiento)
for img in imagenes:
    imagen = iobj.leer_TIFF(ruta_entrenamiento + "/" + img)
    mascara = iobj.leer_TIFF(ruta_mascaras + "/" + img)
    print(imagen.max())
    iobj.histograma(ruta_histograma, img, imagen, mascara)

import matplotlib.pyplot as plt
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
i = Imagen()
img, header = i.leer_fits("fits/r750456-1_PN_NAME_LDU_1_PN_K_3-83_Ha.fits")
img2 = img.copy().astype('uint16')
print("El máximo es: " + str(img.max()))
io.imsave("fits/r750456-1_PN_NAME_LDU_1_PN_K_3-83_Ha_original.tif", img2)
img = img / 65535
threshold = 25000 / 65535
mascara = img.copy().astype(float)
mascara[mascara >= threshold] = np.nan
mascara2 = mascara.copy().astype(float)
mascara2 = mascara2 * 65535
mascara2 = mascara2.astype('uint16')
nans = np.count_nonzero(np.isnan(mascara))
print("Numero de nans: " + str(nans))
print("El máximo es: " + str(mascara2.max()))
io.imsave("fits/r750456-1_PN_NAME_LDU_1_PN_K_3-83_Ha_mascara.tif", mascara2)
#mascara = mascara.astype('uint16')
kernel = Gaussian2DKernel(1)
reconstructed_image = interpolate_replace_nans(mascara, kernel)
nans = np.count_nonzero(np.isnan(reconstructed_image))
print("Numero de nans: " + str(nans))
reconstructed_image = reconstructed_image * 65535
reconstructed_image = reconstructed_image.astype('uint16')
io.imsave("fits/r750456-1_PN_NAME_LDU_1_PN_K_3-83_Ha.tif", reconstructed_image)

rutaoriginal = "entrenamiento/imagenes_recortadas"
rutamascara = "entrenamiento/mascaras_recortadas"
ruta_guardar = "entrenamiento/imagenes_mascaras"
iobj = Imagen()
iobj.imagen_mascara(rutaoriginal, rutamascara, ruta_guardar)

dir_imagenes = "entrenamiento/imagenes"
dir_imagenes_512 = "entrenamiento/imagenes_recortadas"
iobj = Imagen()
iobj.recortar_imagenes(dir_imagenes, dir_imagenes_512, 512)

dir_imagenes = "entrenamiento/mascaras"
dir_imagenes_512 = "entrenamiento/mascaras_recortadas"
iobj = Imagen()
iobj.recortar_imagenes(dir_imagenes, dir_imagenes_512, 512)

dir_imagenes = "entrenamiento/mascaras_HII"
dir_imagenes_512 = "entrenamiento/mascaras_HII_512_ok"
iobj = Imagen()
iobj.recortar_imagenes(dir_imagenes, dir_imagenes_512, 512)

iobj = Imagen()
rutafits = "entrenamiento/mascaras_512_ok"
rutatif = "entrenamiento/mascaras_512_100_img"
imagenes = os.listdir(rutafits)
for img in imagenes:
    imagen = iobj.leer_TIFF(rutafits + "/" + img)
    io.imsave(os.path.join(rutatif, img + ".tif"), imagen)

dir_imagenes = "entrenamiento_estrellas//tif_original_en_dr2"
dir_imagenes_512 = "entrenamiento_estrellas//tif_original_recortadas_en_dr2"
iobj = Imagen()
iobj.recortar_imagenes(dir_imagenes, dir_imagenes_512, 512)

dir_imagenes = "entrenamiento_estrellas//tif_en_dr2_mascara"
dir_imagenes_512 = "entrenamiento_estrellas//tif_recortadas_en_dr2_mascara"
iobj.recortar_imagenes(dir_imagenes, dir_imagenes_512, 512)

dir_imagenes = "entrenamiento_estrellas//tif_original_no_en_dr2"
dir_imagenes_512 = "entrenamiento_estrellas//tif_original_recortadas_no_en_dr2"
iobj = Imagen()
iobj.recortar_imagenes(dir_imagenes, dir_imagenes_512, 512)
iobj = Imagen()
dir_imagenes = "entrenamiento_estrellas//tif_en_dr2_mascara"
dir_imagenes_512 = "entrenamiento_estrellas//tif_recortadas_en_dr2_mascara"
iobj.recortar_imagenes(dir_imagenes, dir_imagenes_512, 512)
dir_imagenes = "entrenamiento_estrellas//tif_no_en_dr2_mascara"
dir_imagenes_512 = "entrenamiento_estrellas//tif_recortadas_no_en_dr2_mascara"
iobj.recortar_imagenes(dir_imagenes, dir_imagenes_512, 512)
dir_imagenes = "entrenamiento//imagenes_pr"
#dir_imagenes_1024 = "entrenamiento//imagenes_min_max_1024"
dir_imagenes_512 = "entrenamiento//imagenes_pr_512"
dir_mascaras = "entrenamiento//mascaras_pr"
#dir_mascaras_1024 = "entrenamiento//mascaras_min_max_1024"
dir_mascaras_512 = "entrenamiento//mascaras_pr_512"
iobj = Imagen()
#iobj.recortar_imagenes(dir_imagenes, dir_imagenes_1024, 1024)
iobj.recortar_imagenes(dir_imagenes, dir_imagenes_512, 512)
#iobj.recortar_imagenes(dir_mascaras, dir_mascaras_1024, 1024)
iobj.recortar_imagenes(dir_mascaras, dir_mascaras_512, 512)
dir_imagenes = "pruebas//imagenes_completas"
dir_imagenes_512 = "pruebas//imagenes_recortadas"
iobj = Imagen()
iobj.recortar_imagenes(dir_imagenes, dir_imagenes_512, 512)'''