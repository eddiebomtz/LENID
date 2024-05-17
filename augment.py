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
from image import Image
from preprocess import Preprocess
class Augment:
    def __init__(self):
        self.num_images = 1
    def __shift__(self, image, type, ratio = 0.0):
        if ratio > 1 or ratio < 0:
            print('The value of ratio must be between 0 y 1')
            return image
        ratio = random.uniform(-ratio, ratio)
        h, w = image.shape[:2]
        #Horizontal
        if type == 1:
            to_shift = w * ratio
            image = image[:, :int(w-to_shift)]
            image = image[:, int(-1*to_shift):]
        #Vertical
        elif type == 2:
            to_shift = h * ratio
            image = image[:int(h-to_shift), :]
            image = image[int(-1*to_shift):, :]
        image = cv2.resize(image, (h, w), cv2.INTER_CUBIC)
        return image
    def __bright_contrast__(self, image, b, c):
        new_image = image.copy()
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                new_image[y,x] = np.clip(c * image[y,x] + b, 0, 65535)
        return new_image
    def __zoom_in__(self, image, mask):
        height, width = image.shape
        zoom_pix = random.randint(0, 10)
        zoom_factor = 1 + (2 * zoom_pix) / height
        image = cv2.resize(image, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_CUBIC)
        top_crop = (image.shape[0] - height) // 2
        left_crop = (image.shape[1] - width) // 2
        image = image[top_crop: top_crop + height, left_crop: left_crop + width]
        mask = cv2.resize(mask, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_CUBIC)
        top_crop = (mask.shape[0] - height) // 2
        left_crop = (mask.shape[1] - width) // 2
        mask = mask[top_crop: top_crop + height, left_crop: left_crop + width]
        return image, mask
    def __flip_vertical__(self, image):
        image = cv2.flip(image, 0)
        return image
    def __flip_horizontal__(self, image):
        image = cv2.flip(image, 1)
        return image
    def __flip_both__(self, image):
        image = cv2.flip(image, -1)
        return image
    def get_num_images(self):
        return self.num_images
    def get_image(self, indice, path_image, path_masks, save_path, save_path_mas, augment):
        folders = path_image[indice].split('\\')
        name = folders[len(folders)-1]
        name = name.replace(".tif", "")
        path_img = path_image[indice]
        path_mas = path_masks[indice]
        image = cv2.imread(path_img, -1)
        self.pp = Preprocess(path_image[indice], True)
        '''sigma = 2.5
        if augment:
            image = self.pp.anisodiff(self.pp.processed_image,100,80,0.075,(1,1),sigma,1)'''
        x, y = image.shape
        mask = cv2.imread(path_mas, -1)
        text = ""
        if augment:
            self.flip_vertical = random.randint(0, 1)
            self.flip_horizontal = random.randint(0, 1)
            self.flip_both = random.randint(0, 1)
            self.zoom_in = random.randint(0, 1)
            #self.shift_horizontal = random.randint(0, 1)
            #self.shift_vertical = random.randint(0, 1)
            #self.bright_contrast = random.randint(0, 1)
            if self.flip_vertical == 1:
                image = self.__flip_vertical__(image)
                mask = self.__flip_vertical__(mask)
                text += "_flip_vertical"
                #image, mask = self.__zoom_in__(image, mask)
                #text += "_zoom_in"
            if self.flip_horizontal == 1:
                image = self.__flip_horizontal__(image)
                mask = self.__flip_horizontal__(mask)
                text += "_flip_horizontal"
                #image, mask = self.__zoom_in__(image, mask)
                #text += "_zoom_in"
            if self.flip_both == 1:
                image = self.__flip_both__(image)
                mask = self.__flip_both__(mask)
                text += "_flip_both"
                #image, mask = self.__zoom_in__(image, mask)
                #text += "_zoom_in"
            if self.zoom_in == 1:
                image, mask = self.__zoom_in__(image, mask)
                text += "_zoom_in"
            '''if self.shift_horizontal == 1:
                image = self.__shift__(image, 1, ratio = 0.2)
                mask = self.__shift__(mask, 1, ratio = 0.2)
            if self.shift_vertical == 1:
                image = self.__shift__(image, 2, ratio = 0.2)
                mask = self.__shift__(mask, 2, ratio = 0.2)
            if self.bright_contrast == 1:
                #bright de 0 a 100
                #contrast de 1 al 3
                image = self.__bright_contrast__(image, b = 20, c = 1.5)
                mask = self.__bright_contrast__(mask, b = 20, c = 1.5)'''
            if text == "":
                image, mask = self.__zoom_in__(image, mask)
                text += "_zoom_in"
                self.flip_horizontal = random.randint(0, 1)
                if self.flip_horizontal == 1:
                    image = self.__flip_horizontal__(image)
                    mask = self.__flip_horizontal__(mask)
                    text += "_flip_horizontal"
                else:
                    image = self.__flip_vertical__(image)
                    mask = self.__flip_vertical__(mask)
                    text += "_flip_vertical"
            #cv2.imwrite(save_path + '/' + name + "_" + str(self.num) + "_" + text + '.tif', image)
            #cv2.imwrite(save_path_mas + '/' + name + "_" + str(self.num) + "_" + text + '.tif', mask)
            self.num_images += 1
        '''else:
            cv2.imwrite(save_path + '/' + name + "_" + str(self.num) + "_" + text + '.tif', image)
            if path_masks != "":
                cv2.imwrite(save_path_mas + '/' + name + "_" + str(self.num) + "_" + text + '.tif', mask)'''
        image = image.reshape(x, y, 1)
        mask = mask.reshape(x, y, 1)
        return image, mask
    def augment_generator(self, size_batch, path_originals, path_masks, save_path, save_path_mas, size_img_x, size_img_y, augment = True):
        while True:
            indexes = np.random.permutation(len(path_originals))
            for batch in range(0, len(indexes), size_batch):
                actual_batch = indexes[batch:(batch + size_batch)]
                images = np.empty([0, size_img_y, size_img_x, 1], dtype=np.float32)
                masks = np.empty([0, size_img_y, size_img_x, 1], dtype=np.float32)
                for i in actual_batch:
                    image, mask = self.get_image(i, path_originals, path_masks, save_path, save_path_mas, augment = augment)
                    image = (image - image.min()) / (image.max() - image.min())
                    mask = mask / mask.max()
                    mask[mask > 0.5] = 1
                    mask[mask <= 0.5] = 0
                    '''print(image.shape)
                    print(mask.shape)
                    print(images.shape)
                    print(masks.shape)'''
                    images = np.append(images, [image], axis=0)
                    masks = np.append(masks, [mask], axis=0)
                yield (images, masks)
    def create_directory(self, path):
        try:
            os.stat(path)
        except:
            os.mkdir(path)
    def alphanumeric_order(self, list):
        import re
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
        return sorted(list, key=alphanum_key)
    def test_generator(self, path, save_path, extended_objects):
        imageobj = Image()
        list_images = self.alphanumeric_order(list(paths.list_images(path)))
        for i,item in enumerate(list_images):
            #image = cv2.imread(item, -1)
            image = imageobj.read_TIFF(item)
            name = item.replace(path + "\\", "")
            name = name.replace(".tif", "")
            #self.create_directory(save_path + "/original/")
            io.imsave(os.path.join(save_path, name + ".tif"), image)
            pp = Preprocess(item, True)
            pp.autocontrast(2, True)
            pp.save_image_tif(save_path, name + "_percentile_range.tif", True)
            #if extended_objects:
                #pp.pfcm_3()
                #pp.save_image_tif(save_path, name + "_percentile_range_pfcm.tif", True)
            #image = pp.processed_image
            image = (image - image.min()) / (image.max() - image.min())
            image = np.reshape(image,image.shape+(1,))
            image = np.reshape(image,(1,)+image.shape)
            yield image