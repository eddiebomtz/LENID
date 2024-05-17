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
class image:
    def create_directory(self, path):
        try:
            os.stat(path)
        except:
            os.mkdir(path)
    def read_TIFF(self, path):
        im = Image.open(path)
        out = np.array(im)
        img = out.astype('uint16')
        self.image = img
        return img
    def read_fits(self, path):
        im = fits.open(path)
        header = im[1].header
        im = im[1].data
        out = np.array(im)
        out = out.astype('uint16')
        self.image = out
        return out, header
    def histogram(self, save_path, name, image, mask):
        import cv2
        NBINS = 256
        #histogram, bin_edges = np.histogram(image, bins=NBINS, range=(0, 2000))
        image = image / image.max()
        mask = mask / mask.max()
        mask = mask.astype("uint8")
        histogram = cv2.calcHist([image],[0],None,[NBINS],[0,1])
        hist_mask = cv2.calcHist([image],[0],mask,[NBINS],[0,1])
        plt.figure()
        plt.title("Grayscale Histogram of " + name)
        plt.xlabel("Grayscale value")
        plt.ylabel("Number of pixels")
        plt.xlim([0.0, 256.0])
        plt.plot(histogram)
        plt.plot(hist_mask)
        plt.savefig(save_path + "/" + name + "_histogram.tif")
        plt.close()
    def cut_images(self, path_img_completa, save_path, tamcorte):
        import os
        from PIL import Image
        import skimage.io as io
        os.chdir(os.getcwd())
        images = os.listdir(path_img_completa)
        for img in images:
            imageha = Image.open(path_img_completa + "/" + img)
            ancho, alto = imageha.size
            contador = 0
            for i in range(0, alto, tamcorte):
                for j in range(0, ancho, tamcorte):
                    caja = (j, i, j + tamcorte, i + tamcorte)
                    cortar = imageha.crop(caja)
                    cortar.save(save_path + "/" + img + "_corte_" + str(contador) + ".tif")
                    imagerecortada = Image.open(save_path + "/" + img + "_corte_" + str(contador) + ".tif")
                    imagerecortada = np.array(imagerecortada)
                    imgint = imagerecortada.astype('uint16')
                    io.imsave(save_path + "/" + img + "_corte_" + str(contador) + ".tif", imgint)
                    contador += 1
    def alphanumeric_order(self, list):
        import re
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
        return sorted(list, key=alphanum_key)
    def paste_images(self, dir_images, dir_save, ancho, alto, tam):
        import skimage.io as io
        self.create_directory(dir_save)
        list_images = self.alphanumeric_order(os.listdir(dir_images))
        i = 1
        path_images_x = []
        images_y = []
        for img in list_images:
            path_images_x += [img]
            if i % (ancho / tam) == 0:
                concatenate_images = Image.fromarray(
                  np.concatenate(
                    [np.array(Image.open(dir_images + "/" + x)) for x in path_images_x],
                    axis=1 #Concatenate images in horizontal axis
                  )
                )
                images_y += [concatenate_images]
                path_images_x = []
            if i % ((alto / tam) * (ancho / tam)) == 0:
                concatenate_images = Image.fromarray(
                  np.concatenate(
                    [np.array(images) for images in images_y],
                    axis=0 #Concatenate images in vertical axis
                  )
                )
                concatenate_images = np.array(concatenate_images)
                concatenate_images = concatenate_images.astype('uint16')
                images_y = []
                img = img.replace(".tif", "")
                io.imsave(os.path.join(dir_save, img + "_image_completa.tif"), concatenate_images)
            i += 1
    def prediction_to_image(self, path, save_path, percentage):
        from imutils import paths
        list_images = self.alphanumeric_order(list(paths.list_images(path)))
        for i, item in enumerate(list_images):
            prediction = self.read_TIFF(item)
            prediction = prediction / 65535
            img_bool = prediction.astype('float')
            img_bool[img_bool > percentage] = 1
            img_bool[img_bool <= percentage] = 0
            image = img_bool.astype('uint16')
            image = image * 65535
            image.shape = prediction.shape
            io.imsave(os.path.join(save_path,  str(i) + "_stars_" + str(percentage) + ".tif"), image)
    def image_mask(self, original_path, pathmask, save_path):
        from imutils import paths
        list_images = self.alphanumeric_order(list(paths.list_images(original_path)))
        for i, item in enumerate(list_images):
            image = self.read_TIFF(item)
            name_mask = item.replace(original_path + "\\", "")
            name_mask = name_mask.replace("interpolation_1000.tif", "interpolation_1000_predict_50.tif")
            mask = self.read_TIFF(os.path.join(pathmask, name_mask))
            image = image / 65535 * 255
            mask = mask / 65535
            without_background = image * mask
            without_background = without_background.astype("uint8")
            io.imsave(os.path.join(save_path, name_mask), without_background)
    def remove_stars(self, original_path, path_predict, save_path, percentage):
        from imutils import paths
        #from Preprocess import Preprocess
        list_images = self.alphanumeric_order(list(paths.list_images(original_path)))
        for i, item in enumerate(list_images):
            img_original = self.read_TIFF(item)
            print(str(i) + item)
            img = item.replace(original_path + "\\", "")
            img = img.replace(".tif", "_predict.tif")
            print(str(i) + " " + path_predict + "/" + img)
            img_bool = self.read_TIFF(os.path.join(path_predict, img))
            img_bool = img_bool / 65535
            #percentage_1_0 = percentage / 100
            img_bool[img_bool > percentage] = 1
            img_bool[img_bool <= percentage] = 0
            h, w = img_original.shape 
            img_without_stars = img_original * (1 - img_bool)
            img_without_stars.shape = img_original.shape
            img_without_stars = img_without_stars.astype('uint16')
            img = item.replace(original_path + "\\", "") 
            img = img.replace(".tif.tif", "_predict.tif")
            print(str(i) + " " + save_path + "/" + img)
            io.imsave(os.path.join(save_path, img), img_without_stars)
            '''pp = Preprocess(os.path.join(save_path,  str(i) + "_sin_stars_" + str(percentage) + ".tif"), True)
            pp.autocontrast(2, True)
            pp.save_image_tif(os.getcwd() + "/" + save_path, str(i) + "_sin_stars_" + str(percentage) + "_percentile_range.tif", True)'''
    def paste_images(self, imgs_dir, save_dir, num_imgs, width):
        sum = width
        self.create_directory(save_dir)
        images = self.alphanumeric_order(os.listdir(imgs_dir))
        img = Image.new("RGB", (2048,4096))
        xv = 0
        yv = 0
        k = 1
        save = (num_imgs / ((2048 / width) * (4096 / width)) + 1)
        for i in range(1, num_imgs + 1, 1):
            im = self.read_TIFF(imgs_dir + "/" + images[i - 1])
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
                yv = yv + sum
            else:
                xv = xv + sum
            if i % save == 0:
                xv = 0
                yv = 0
                img.save(save_dir + "/" + str(k) + "_completa.tif")
                k = k + 1
'''
import os
iobj = Image()
pathfits = "training/cropped_masks"
pathimageres = "training/cropped_images_respaldo"
pathimage = "training/cropped_images"
images = os.listdir(pathfits)
for img in images:
    image = Image.open(pathimageres + "/" + img)
    io.imsave(pathimage + "/" + img, image)

import os
iobj = Image()
pathfits = "training/cropped_masks"
pathimage = "training/cropped_images"
pathimgmask = "training/images_masks"
images = os.listdir(pathfits)
#data = []
#imagesdata = []
for img in images:
    #image = iobj.leer_TIFF(pathfits + "/" + img)
    image = Image.open(pathfits + "/" + img)
    #imagesdata.append(np.array(image))
    extrema = image.convert("L").getextrema()
    #data.append(extrema)
    print(img)
    if extrema == (0, 0): 
        print("All the pixels are black")
        image.close()
        os.remove(pathfits + "/" + img)
        os.remove(pathimgmask + "/" + img)
        img2 = img.replace("_predict_50", "")
        os.remove(pathimage + "/" + img2)
    else:
        print("Not all the pixels are black")

iobj = Image()
path_image = "fits_g139_tif"
image_ha = iobj.leer_TIFF(path_image + "/r431413-1_PN_G139.0+03.2_Ha_2_600_x_600.tif")
image_i = iobj.leer_TIFF(path_image + "/r431415-1_PN_G139.0+03.2_i_2_600_x_600.tif")
image_ha_i = image_ha - image_i
io.imsave(path_image + "/r431414_5-1_ha_i.fits.tif", image_ha_i)

image_u = iobj.leer_TIFF(path_image + "/r763738-1_u.fits.tif")
image_g = iobj.leer_TIFF(path_image + "/r763739-1_g.fits.tif")
image_u_g = image_u - image_g
io.imsave(path_image + "/r763738_9-1_u_gfits.tif", image_u_g)

iobj = Image()
path_image = "fits_g139"
path_result = "fits_g139_tif"
images = os.listdir(path_image)
for img in images:
    image, header = iobj.leer_fits(path_image + "/" + img)
    io.imsave(path_result + "/" + img + ".tif", image)
iobj = Image()
path_training = "training/images_sin_fondo_512_threshold_2000"
path_masks = "training/masks_512_threshold_2000"
path_histogram = "training/images_sin_fondo_512_threshold_2000_histogram"
images = os.listdir(path_training)
for img in images:
    image = iobj.leer_TIFF(path_training + "/" + img)
    mask = iobj.leer_TIFF(path_masks + "/" + img)
    print(image.max())
    iobj.histogram(path_histogram, img, image, mask)

import matplotlib.pyplot as plt
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
i = Image()
img, header = i.leer_fits("fits/r750456-1_PN_NAME_LDU_1_PN_K_3-83_Ha.fits")
img2 = img.copy().astype('uint16')
print("El máximo es: " + str(img.max()))
io.imsave("fits/r750456-1_PN_NAME_LDU_1_PN_K_3-83_Ha_original.tif", img2)
img = img / 65535
threshold = 25000 / 65535
mask = img.copy().astype(float)
mask[mask >= threshold] = np.nan
mask2 = mask.copy().astype(float)
mask2 = mask2 * 65535
mask2 = mask2.astype('uint16')
nans = np.count_nonzero(np.isnan(mask))
print("Numero de nans: " + str(nans))
print("El máximo es: " + str(mask2.max()))
io.imsave("fits/r750456-1_PN_NAME_LDU_1_PN_K_3-83_Ha_mask.tif", mask2)
#mask = mask.astype('uint16')
kernel = Gaussian2DKernel(1)
reconstructed_image = interpolate_replace_nans(mask, kernel)
nans = np.count_nonzero(np.isnan(reconstructed_image))
print("Numero de nans: " + str(nans))
reconstructed_image = reconstructed_image * 65535
reconstructed_image = reconstructed_image.astype('uint16')
io.imsave("fits/r750456-1_PN_NAME_LDU_1_PN_K_3-83_Ha.tif", reconstructed_image)

original_path = "training/cropped_images"
pathmask = "training/cropped_masks"
save_path = "training/images_masks"
iobj = Image()
iobj.image_mask(original_path, pathmask, save_path)

dir_images = "training/images"
dir_images_512 = "training/cropped_images"
iobj = Image()
iobj.recortar_images(dir_images, dir_images_512, 512)

dir_images = "training/masks"
dir_images_512 = "training/cropped_masks"
iobj = Image()
iobj.recortar_images(dir_images, dir_images_512, 512)

dir_images = "training/masks_HII"
dir_images_512 = "training/masks_HII_512_ok"
iobj = Image()
iobj.recortar_images(dir_images, dir_images_512, 512)

iobj = Image()
pathfits = "training/masks_512_ok"
pathtif = "training/masks_512_100_img"
images = os.listdir(pathfits)
for img in images:
    image = iobj.leer_TIFF(pathfits + "/" + img)
    io.imsave(os.path.join(pathtif, img + ".tif"), image)

dir_images = "training_stars//tif_original_en_dr2"
dir_images_512 = "training_stars//tif_original_recortadas_en_dr2"
iobj = Image()
iobj.recortar_images(dir_images, dir_images_512, 512)

dir_images = "training_stars//tif_en_dr2_mask"
dir_images_512 = "training_stars//tif_recortadas_en_dr2_mask"
iobj.recortar_images(dir_images, dir_images_512, 512)

dir_images = "training_stars//tif_original_no_en_dr2"
dir_images_512 = "training_stars//tif_original_recortadas_no_en_dr2"
iobj = Image()
iobj.recortar_images(dir_images, dir_images_512, 512)
iobj = Image()
dir_images = "training_stars//tif_en_dr2_mask"
dir_images_512 = "training_stars//tif_recortadas_en_dr2_mask"
iobj.recortar_images(dir_images, dir_images_512, 512)
dir_images = "training_stars//tif_no_en_dr2_mask"
dir_images_512 = "training_stars//tif_recortadas_no_en_dr2_mask"
iobj.recortar_images(dir_images, dir_images_512, 512)
dir_images = "training//images_pr"
#dir_images_1024 = "training//images_min_max_1024"
dir_images_512 = "training//images_pr_512"
dir_masks = "training//masks_pr"
#dir_masks_1024 = "training//masks_min_max_1024"
dir_masks_512 = "training//masks_pr_512"
iobj = Image()
#iobj.recortar_images(dir_images, dir_images_1024, 1024)
iobj.recortar_images(dir_images, dir_images_512, 512)
#iobj.recortar_images(dir_masks, dir_masks_1024, 1024)
iobj.recortar_images(dir_masks, dir_masks_512, 512)
dir_images = "test//images_completas"
dir_images_512 = "test//cropped_images"
iobj = Image()
iobj.recortar_images(dir_images, dir_images_512, 512)'''