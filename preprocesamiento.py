# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 10:56:59 2019

@author: eduardo
"""
import os
import cv2
import cmath
import numba
import warnings
import pointarray
import numpy as np
from numba import cuda
import skfuzzy as fuzz
import skimage.io as io
from PFCM import PFCM
from imagen import Imagen
from scipy import ndimage
from skimage import exposure
import skimage.exposure as skie
import skimage.morphology as morph
from skimage.filters import gaussian
from astropy.stats import mad_std
import numpy as np
import matplotlib.pyplot as plt
class preprocesamiento:
    def __init__(self, ruta, tiff):
        if ruta != None:
            self.imagenobj = Imagen()
            if tiff:
                self.imagen = self.imagenobj.leer_TIFF(ruta)
            else:
                self.imagen, self.header = self.imagenobj.leer_fits(ruta)
            self.imagenprocesada = self.imagen
    def imagen_a_procesar(self, imagen):
        self.imagen = imagen
        self.imagenprocesada = self.imagen
    def normalizar_img(self, tipo):
        img = self.imagenprocesada
        if tipo == 1:
            img = 2 * ((img - img.min()) / (img.max() - img.min())) - 1
        elif tipo == 2:
            img = (img - img.min()) / (img.max() - img.min()) * 65535
            img = img.astype("uint16")
            self.imagenprocesada = img
        return img
    def elimina_fondo_mascara(self, veces_sigma):
        from astropy.io import fits
        fits_table = fits.open('iphas-images.fits')
        self.run = self.header["run"]
        self.imageid = self.header["imageid"]
        data = fits_table[1].data
        runs = np.array(data["run"])
        valids = np.array(np.where(runs == self.run))
        idrun = valids.flat[self.imageid - 1]
        original = self.imagenprocesada
        sigma = self.sigma()
        mascara = original.copy().astype(float)
        skylevel = data["skylevel"][idrun]
        skynoise = data["skynoise"][idrun]
        threshold = skylevel + skynoise + (sigma * veces_sigma)
        mascara[mascara <= threshold] = 0
        mascara[mascara > 0] = 65535
        mascara = mascara.astype('uint16')
        self.imagenprocesada = mascara
    def elimina_fondo(self):
        from astropy.io import fits
        fits_table = fits.open('iphas-images.fits')
        self.run = self.header["run"]
        self.imageid = self.header["imageid"]
        data = fits_table[1].data
        runs = np.array(data["run"])
        valids = np.array(np.where(runs == self.run))
        idrun = valids.flat[self.imageid - 1]
        original = self.imagenprocesada
        sigma = self.sigma()
        mascara = original.copy().astype(float)
        skylevel = data["skylevel"][idrun]
        skynoise = data["skynoise"][idrun]
        threshold = skylevel + skynoise + (sigma * 3)
        mascara[mascara <= threshold] = 0
        mascara[mascara > 0] = 1
        mascara = mascara.astype('uint16')
        original = self.imagenprocesada * mascara
        original = original.astype('uint16')
        self.imagenprocesada = original
    def elimina_fondo_tif(self):
        threshold = 2000
        original = self.imagenprocesada
        mascara = original.copy().astype(float)
        mascara[mascara >= threshold] = threshold
        mascara = mascara.astype('uint16')
        self.imagenprocesada = mascara
    def interpolacion_saturadas(self, threshold):
        from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
        img = self.imagenprocesada
        img = img / 65535
        threshold = threshold / 65535
        mascara = img.copy().astype(float)
        mascara[mascara >= threshold] = np.nan
        mascara2 = mascara.copy().astype(float)
        mascara2 = mascara2 * 65535
        mascara2 = mascara2.astype('uint16')
        nans = np.count_nonzero(np.isnan(mascara))
        kernel = Gaussian2DKernel(1)
        imagenreconstruida = interpolate_replace_nans(mascara, kernel)
        while nans > 0:
            imagenreconstruida = interpolate_replace_nans(imagenreconstruida, kernel)
            nans = np.count_nonzero(np.isnan(imagenreconstruida))
        imagenreconstruida = imagenreconstruida * 65535
        imagenreconstruida = imagenreconstruida.astype('uint16')
        self.imagenprocesada = imagenreconstruida
    def elimina_blooming(self):
        original = self.imagen
        mascara = original.copy().astype(float)
        mascara[mascara >= 32500.0] = 0
        mascara[mascara > 0] = 1
        mascara = mascara.astype('uint16')
        original = self.imagen * mascara
        self.imagenprocesada = original
    def sigma(self):
        sigma = mad_std(self.imagenprocesada)
        return sigma
    def anisodiff(self,img,niter=1,kappa=50,gamma=0.1,step=(1.,1.),sigma=0,option=1,ploton=False):
        """
        Anisotropic diffusion.
        Usage:
        imgout = anisodiff(im, niter, kappa, gamma, option)
        Arguments:
                img    - input image
                niter  - number of iterations
                kappa  - conduction coefficient 20-100 ?
                gamma  - max value of .25 for stability
                step   - tuple, the distance between adjacent pixels in (y,x)
                option - 1 Perona Malik diffusion equation No 1
                         2 Perona Malik diffusion equation No 2
                ploton - if True, the image will be plotted on every iteration
        Returns:
                imgout   - diffused image.
        kappa controls conduction as a function of gradient.  If kappa is low
        small intensity gradients are able to block conduction and hence diffusion
        across step edges.  A large value reduces the influence of intensity
        gradients on conduction.
        gamma controls speed of diffusion (you usually want it at a maximum of
        0.25)
        step is used to scale the gradients in case the spacing between adjacent
        pixels differs in the x and y axes
        Diffusion equation 1 favours high contrast edges over low contrast ones.
        Diffusion equation 2 favours wide regions over smaller ones.
        Reference: 
        P. Perona and J. Malik. 
        Scale-space and edge detection using ansotropic diffusion.
        IEEE Transactions on Pattern Analysis and Machine Intelligence, 
        12(7):629-639, July 1990.
        """
        if img.ndim == 3:
            warnings.warn("Only grayscale images allowed, converting to 2D matrix")
            img = img.mean(2)
        img = img.astype('float64')
        imgout = img.copy()
        deltaS = np.zeros_like(imgout)
        deltaE = deltaS.copy()
        NS = deltaS.copy()
        EW = deltaS.copy()
        gS = np.ones_like(imgout)
        gE = gS.copy()
        if ploton:
            import pylab as pl
            fig = pl.figure(figsize=(20,5.5),num="Anisotropic diffusion")
            ax1,ax2 = fig.add_subplot(1,2,1),fig.add_subplot(1,2,2)
            ax1.imshow(img,interpolation='nearest')
            ih = ax2.imshow(imgout,interpolation='nearest',animated=True)
            ax1.set_title("Original image")
            ax2.set_title("Iteration 0")
            fig.canvas.draw()
        for ii in np.arange(1,niter):
            deltaS[:-1,: ] = np.diff(imgout,axis=0)
            deltaE[: ,:-1] = np.diff(imgout,axis=1)
            if 0<sigma:
                deltaSf=gaussian(deltaS,sigma);
                deltaEf=gaussian(deltaE,sigma);
            else: 
                deltaSf=deltaS;
                deltaEf=deltaE;	
            if option == 1:
                gS = np.exp(-(deltaSf/kappa)**2.)/step[0]
                gE = np.exp(-(deltaEf/kappa)**2.)/step[1]
            elif option == 2:
                gS = 1./(1.+(deltaSf/kappa)**2.)/step[0]
                gE = 1./(1.+(deltaEf/kappa)**2.)/step[1]
            E = gE*deltaE
            S = gS*deltaS
            NS[:] = S
            EW[:] = E
            NS[1:,:] -= S[:-1,:]
            EW[:,1:] -= E[:,:-1]
            imgout += gamma*(NS+EW)
            if ploton:
                iterstring = "Iteration %i" %(ii+1)
                ih.set_data(imgout)
                ax2.set_title(iterstring)
                fig.canvas.draw()
        self.imagenprocesada = imgout
        return imgout
    def zscale_range(self, contrast=0.25, num_points=600, num_per_row=120):
        #print("::: Calculando valor minimo y maximo con zscale range :::")
        if len(self.imagen.shape) != 2:
            raise ValueError("input data is not an image")
        if contrast <= 0.0:
            contrast = 1.0
        if num_points > np.size(self.imagen) or num_points < 0:
            num_points = 0.5 * np.size(self.imagen)
        num_per_col = int(float(num_points) / float(num_per_row) + 0.5)
        xsize, ysize = self.imagen.shape
        row_skip = float(xsize - 1) / float(num_per_row - 1)
        col_skip = float(ysize - 1) / float(num_per_col - 1)
        data = []
        for i in range(num_per_row):
            x = int(i * row_skip + 0.5)
            for j in range(num_per_col):
                y = int(j * col_skip + 0.5)
                data.append(self.imagen[x, y])
        num_pixels = len(data)
        data.sort()
        data_min = min(data)
        data_max = max(data)
        center_pixel = (num_pixels + 1) / 2
        if data_min == data_max:
            return data_min, data_max
        if num_pixels % 2 == 0:
            center_pixel = round(center_pixel)
            median = data[center_pixel - 1]
        else:
            median = 0.5 * (data[center_pixel - 1] + data[center_pixel])
        pixel_indeces = map(float, range(num_pixels))
        points = pointarray.PointArray(pixel_indeces, data, min_err=1.0e-4)
        fit = points.sigmaIterate()
        num_allowed = 0
        for pt in points.allowedPoints():
            num_allowed += 1
        if num_allowed < int(num_pixels / 2.0):
            return data_min, data_max
        z1 = median - (center_pixel - 1) * (fit.slope / contrast)
        z2 = median + (num_pixels - center_pixel) * (fit.slope / contrast)
        if z1 > data_min:
            zmin = z1
        else:
            zmin = data_min
        if z2 < data_max:
            zmax = z2
        else:
            zmax = data_max
        if zmin >= zmax:
            zmin = data_min
            zmax = data_max
        return zmin, zmax
    def arcsin_percentile(self, min_percent=3.0, max_percent=99.0):
        img = self.imagenprocesada
        limg = np.arcsinh(img)
        limg = limg / limg.max()
        low = np.percentile(limg, min_percent)
        high = np.percentile(limg, max_percent)
        return limg, low, high
    def percentile_range(self, min_percent=3.0, max_percent=99.0, num_points=5000, num_per_row=250):
        #print("::: Calculando valor minimo y maximo con percentile :::")
        if not 0 <= min_percent <= 100:
            raise ValueError("invalid value for min percent '%s'" % min_percent)
        elif not 0 <= max_percent <= 100:
            raise ValueError("invalid value for max percent '%s'" % max_percent)
        min_percent = float(min_percent) / 100.0
        max_percent = float(max_percent) / 100.0
        if len(self.imagen.shape) != 2:
            raise ValueError("input data is not an image")
        if num_points > np.size(self.imagen) or num_points < 0:
            num_points = 0.5 * np.size(self.imagen)
        num_per_col = int(float(num_points) / float(num_per_row) + 0.5)
        xsize, ysize = self.imagen.shape
        row_skip = float(xsize - 1) / float(num_per_row - 1)
        col_skip = float(ysize - 1) / float(num_per_col - 1)
        data = []
        for i in range(num_per_row):
            x = int(i * row_skip + 0.5)
            for j in range(num_per_col):
                y = int(j * col_skip + 0.5)
                data.append(self.imagen[x, y])
        data.sort()
        zmin = data[int(min_percent * len(data))]
        zmax = data[int(max_percent * len(data))]
        return zmin, zmax
    def autocontraste(self, tipo, original):
        #print("::: Procesando imagen :::")
        zmin = 0
        zmax = 0
        limg = 0
        if tipo == 1:
            zmin, zmax = self.zscale_range()
            if original:
                self.imagenprocesada = np.where(self.imagen > zmin, self.imagen, zmin)
                self.imagenprocesada = np.where(self.imagenprocesada < zmax, self.imagenprocesada, zmax)
            else:
                self.imagenprocesada = np.where(self.imagenprocesada > zmin, self.imagenprocesada, zmin)
                self.imagenprocesada = np.where(self.imagenprocesada < zmax, self.imagenprocesada, zmax)
            nonlinearity = 3.0
            nonlinearity = max(nonlinearity, 0.001)
            max_asinh = cmath.asinh(nonlinearity).real
            self.imagenprocesada = (self.imagen.max() / max_asinh) * (np.arcsinh((self.imagenprocesada - zmin) * (nonlinearity / (zmax - zmin))))
        elif tipo == 2:
            zmin, zmax = self.percentile_range(min_percent=3.0, max_percent=99.5, num_points=6000, num_per_row=350)
            if original:
                self.imagenprocesada = np.where(self.imagen > zmin, self.imagen, zmin)
                self.imagenprocesada = np.where(self.imagenprocesada < zmax, self.imagenprocesada, zmax)
            else:
                self.imagenprocesada = np.where(self.imagenprocesada > zmin, self.imagenprocesada, zmin)
                self.imagenprocesada = np.where(self.imagenprocesada < zmax, self.imagenprocesada, zmax)
            self.imagenprocesada = (self.imagenprocesada - zmin) * (self.imagen.max() / (zmax - zmin))
        elif tipo == 3:
            limg, zmin, zmax = self.arcsin_percentile(min_percent=3.0, max_percent=99.5)
            self.imagenprocesada = skie.exposure.rescale_intensity(limg, in_range=(zmin, zmax))
            self.imagenprocesada = self.imagenprocesada * self.imagen.max()
        elif tipo == 4:
            zmin, zmax = self.percentile_range(min_percent=3.0, max_percent=99.5, num_points=6000, num_per_row=350)
            if original:
                self.imagenprocesada = np.where(self.imagen > zmin, self.imagen, zmin)
                self.imagenprocesada = np.where(self.imagenprocesada < zmax, self.imagenprocesada, zmax)
            else:
                self.imagenprocesada = np.where(self.imagenprocesada > zmin, self.imagenprocesada, zmin)
                self.imagenprocesada = np.where(self.imagenprocesada < zmax, self.imagenprocesada, zmax)
            nonlinearity = 3.0
            nonlinearity = max(nonlinearity, 0.001)
            max_asinh = cmath.asinh(nonlinearity).real
            self.imagenprocesada = (self.imagen.max() / max_asinh) * (np.arcsinh((self.imagenprocesada - zmin) * (nonlinearity / (zmax - zmin))))
        elif tipo == 5:
            nonlinearity = 3.0
            nonlinearity = max(nonlinearity, 0.001)
            max_asinh = cmath.asinh(nonlinearity).real
            self.imagenprocesada = (self.imagen.max() / max_asinh) * (np.arcsinh((self.imagen - self.imagen.min()) * (nonlinearity / (self.imagen.max() - self.imagen.min()))))
        elif tipo == 6:
            nonlinearity = 3.0
            nonlinearity = max(nonlinearity, 0.001)
            max_asinh = cmath.asinh(nonlinearity).real
            self.imagenprocesada = (self.imagen.max() / max_asinh) * (np.arcsinh((self.imagenprocesada - zmin) * (nonlinearity / (zmax - zmin))))
        self.imagenprocesada = self.imagenprocesada.astype('uint16')
        return limg, zmin, zmax
    def fcm_3(self, minimo, anisodiff = False, median = False, gaussian = False):
        im = self.imagenprocesada
        if anisodiff:
            I = 2 * ((im - im.min()) / (im.max() - im.min())) - 1
            sigma = mad_std(I)
            I = self.anisodiff(I,100,80,0.075,(1,1),sigma,2)
            I = (I - I.min()) / (I.max() - I.min())
        else:
            I = (im - im.min()) / (im.max() - im.min()) 
            if median:
                I = ndimage.median_filter(I, size=3)
            if gaussian:
                I = ndimage.gaussian_filter(I, 2)
        x, y = I.shape
        I = I.reshape(1, x * y)
        fuzziness_degree = 3
        error = 0.001
        maxiter = 100
        centers, u, u0, d, jm, n_iters, fpc = fuzz.cluster.cmeans(I, c=3, m=fuzziness_degree, error=error, maxiter=maxiter, init=None)
        img_clustered = np.argmax(u, axis=0).astype(float)
        img_clustered.shape = I.shape
        label0 = I[img_clustered == 0]
        label1 = I[img_clustered == 1]
        label2 = I[img_clustered == 2]
        maxlabel0 = np.max(label0)
        maxlabel1 = np.max(label1)
        maxlabel2 = np.max(label2)
        img_clustered[img_clustered == 0] = 3
        img_clustered[img_clustered == 1] = 4
        img_clustered[img_clustered == 2] = 5
        if maxlabel0 < maxlabel1 and maxlabel0 < maxlabel2:
            img_clustered[img_clustered == 3] = 0
            if maxlabel1 > maxlabel0 and maxlabel1 < maxlabel2:
                img_clustered[img_clustered == 4] = 1
                img_clustered[img_clustered == 5] = 2
            else:
                img_clustered[img_clustered == 4] = 2
                img_clustered[img_clustered == 5] = 1
        if maxlabel1 < maxlabel0 and maxlabel1 < maxlabel2:
            img_clustered[img_clustered == 4] = 0
            if maxlabel2 > maxlabel0 and maxlabel2 < maxlabel1:
                img_clustered[img_clustered == 3] = 2
                img_clustered[img_clustered == 5] = 1
            else:
                img_clustered[img_clustered == 3] = 1
                img_clustered[img_clustered == 5] = 2
        if maxlabel2 < maxlabel0 and maxlabel2 < maxlabel1:
            img_clustered[img_clustered == 5] = 0
            if maxlabel1 > maxlabel0 and maxlabel1 < maxlabel0:
                img_clustered[img_clustered == 4] = 1
                img_clustered[img_clustered == 3] = 2
            else:
                img_clustered[img_clustered == 4] = 2
                img_clustered[img_clustered == 3] = 1
        label0 = I[img_clustered == 0]
        label1 = I[img_clustered == 1]
        label2 = I[img_clustered == 2]
        threshold = 0
        if minimo:
            maxlabel0 = np.max(label0)
            minlabel1 = np.min(label1)
            threshold = (maxlabel0 + minlabel1) / 2
        else:
            minlabel2 = np.min(label2)
            maxlabel2 = np.max(label2)
            threshold = minlabel2
        imagensinfondo = np.where(img_clustered > threshold, 1, 0)
        imagensinfondo.shape = im.shape
        imagensinfondo = imagensinfondo * im
        imagensinfondo = np.int16(imagensinfondo)
        self.imagenprocesada = imagensinfondo
    def histograma(self):
        import cv2
        h = cv2.calcHist([self.imagen.ravel()], [0], None, [65536], [0,65536]) 
        return h
    def pfcm_2(self, ruta, nombre, anisodiff=True, median=False, gaussian=False):
        im = self.imagenprocesada
        if anisodiff:
            I = 2 * ((im - im.min()) / (im.max() - im.min())) - 1
            sigma = mad_std(I)
            I = self.anisodiff(I,100,80,0.075,(1,1),sigma,2)
            I = (I - I.min()) / (I.max() - I.min())
            self.imagenprocesada = self.imagenprocesada.astype("uint16")
            self.guardar_imagen_tif(ruta, nombre + "_anisodiff.tif", True)
        else:
            I = (im - im.min()) / (im.max() - im.min()) 
            if median:
                I = ndimage.median_filter(I, size=3)
            if gaussian:
                I = ndimage.gaussian_filter(I, 2)
        x, y = I.shape
        I = I.reshape(x * y, 1)
        pfcm = PFCM()
        centers, U, T, obj_fcn = pfcm.pfcm(I, 2, a = 1, b = 2, nc = 2)
        
        colores = []
        grupos_colores = {0:np.array([255,0,0]),1:np.array([0,255,0])}
        for n in range(I.shape[0]):
            color = np.zeros([2])
            for c in range(U.shape[0]):
                color += grupos_colores[c]*U[c,n]
            colores.append(color)
        
        labels = np.argmax(U, axis=0).reshape(im.shape[0], im.shape[1])
        I = I.reshape(im.shape[0], im.shape[1])
        label0 = I[labels == 0]
        label1 = I[labels == 1]
        maxlabel0 = np.max(label0)
        maxlabel1 = np.max(label1)
        labels[labels == 0] = 3
        labels[labels == 1] = 4
        if maxlabel0 < maxlabel1:
            labels[labels == 3] = 0
            labels[labels == 4] = 1
        else:
            labels[labels == 3] = 1 
            labels[labels == 4] = 0
        imglabel = labels.astype("uint16")
        imglabel16 = imglabel * 65535
        self.imagenprocesada = imglabel16
        self.guardar_imagen_tif(ruta, nombre + "_pfcm_binaria_", True)
        imglabel.shape = im.shape
        imagenbinaria = imglabel * self.imagen
        imagenbinaria = imagenbinaria.astype("uint16")
        self.imagenprocesada = imagenbinaria
        self.guardar_imagen_tif(ruta, nombre + "_pfcm", True)
    def dice(self, pred, true, k = 1):
        intersection = np.sum(pred[true==k]) * 2.0
        dice = intersection / (np.sum(pred) + np.sum(true))
        return dice
    def top_hat(self, ruta, nombre):
        import cv2
    def guardar_imagen_png(self, ruta, nombre):
        self.autocontraste(2, True)
        self.imagenprocesada = 255 * (self.imagenprocesada - self.imagenprocesada.min()) / (self.imagenprocesada.max() - self.imagenprocesada.min())
        self.imagenprocesada = self.imagenprocesada.astype('uint8')
        io.imsave(ruta + "/" + nombre + ".png", self.imagenprocesada)
        return self.imagenprocesada
    def guardar_imagen_tif(self, ruta, nombre, guardar_procesada = False):
        if guardar_procesada:
            io.imsave(ruta + "/" + nombre + ".tif", self.imagenprocesada)
            return self.imagenprocesada
        else:
            io.imsave(ruta + "/" + nombre + ".tif", self.imagen)
            return self.imagen
dir_imagenes = "busqueda_nebulosa"
dir_resultado = "busqueda_nebulosa_autocontraste"
imagenes = os.listdir(dir_imagenes)
for img in imagenes:
    print(img)
    pp = preprocesamiento(dir_imagenes + "/" + img, True)
    #pp.guardar_imagen_tif(dir_resultado, img + "_original.tif", True)
    #pp.autocontraste(3, True)
    pp.guardar_imagen_png(dir_resultado, img)