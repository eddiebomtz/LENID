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
from image import Image
from scipy import ndimage
from skimage import exposure
import skimage.exposure as skie
import skimage.morphology as morph
from skimage.filters import gaussian
from astropy.stats import mad_std
import numpy as np
import matplotlib.pyplot as plt
class Preprocess:
    def __init__(self, path, tiff):
        if path != None:
            self.imageobj = Image()
            if tiff:
                self.image = self.imageobj.read_TIFF(path)
            else:
                self.image, self.header = self.imageobj.read_fits(path)
            self.processed_image = self.image
    def image_to_process(self, image):
        self.image = image
        self.processed_image = self.image
    def normalize_img(self, type):
        img = self.processed_image
        if type == 1:
            img = 2 * ((img - img.min()) / (img.max() - img.min())) - 1
        elif type == 2:
            img = (img - img.min()) / (img.max() - img.min()) * 65535
            img = img.astype("uint16")
            self.processed_image = img
        return img
    def remove_background_mask(self, veces_sigma):
        from astropy.io import fits
        fits_table = fits.open('iphas-images.fits')
        self.run = self.header["run"]
        self.imageid = self.header["imageid"]
        data = fits_table[1].data
        runs = np.array(data["run"])
        valids = np.array(np.where(runs == self.run))
        idrun = valids.flat[self.imageid - 1]
        original = self.processed_image
        sigma = self.sigma()
        mask = original.copy().astype(float)
        skylevel = data["skylevel"][idrun]
        skynoise = data["skynoise"][idrun]
        threshold = skylevel + skynoise + (sigma * veces_sigma)
        mask[mask <= threshold] = 0
        mask[mask > 0] = 65535
        mask = mask.astype('uint16')
        self.processed_image = mask
    def remove_background(self):
        from astropy.io import fits
        fits_table = fits.open('iphas-images.fits')
        self.run = self.header["run"]
        self.imageid = self.header["imageid"]
        data = fits_table[1].data
        runs = np.array(data["run"])
        valids = np.array(np.where(runs == self.run))
        idrun = valids.flat[self.imageid - 1]
        original = self.processed_image
        sigma = self.sigma()
        mask = original.copy().astype(float)
        skylevel = data["skylevel"][idrun]
        skynoise = data["skynoise"][idrun]
        threshold = skylevel + skynoise + (sigma * 3)
        mask[mask <= threshold] = 0
        mask[mask > 0] = 1
        mask = mask.astype('uint16')
        original = self.processed_image * mask
        original = original.astype('uint16')
        self.processed_image = original
    def remove_background_tif(self):
        threshold = 2000
        original = self.processed_image
        mask = original.copy().astype(float)
        mask[mask >= threshold] = threshold
        mask = mask.astype('uint16')
        self.processed_image = mask
    def interpolate_saturated(self, threshold):
        from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
        img = self.processed_image
        img = img / 65535
        threshold = threshold / 65535
        mask = img.copy().astype(float)
        mask[mask >= threshold] = np.nan
        mask2 = mask.copy().astype(float)
        mask2 = mask2 * 65535
        mask2 = mask2.astype('uint16')
        nans = np.count_nonzero(np.isnan(mask))
        kernel = Gaussian2DKernel(1)
        reconstructed_image = interpolate_replace_nans(mask, kernel)
        while nans > 0:
            reconstructed_image = interpolate_replace_nans(reconstructed_image, kernel)
            nans = np.count_nonzero(np.isnan(reconstructed_image))
        reconstructed_image = reconstructed_image * 65535
        reconstructed_image = reconstructed_image.astype('uint16')
        self.processed_image = reconstructed_image
    def remove_blooming(self):
        original = self.image
        mask = original.copy().astype(float)
        mask[mask >= 32500.0] = 0
        mask[mask > 0] = 1
        mask = mask.astype('uint16')
        original = self.image * mask
        self.processed_image = original
    def sigma(self):
        sigma = mad_std(self.processed_image)
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
        self.processed_image = imgout
        return imgout
    def zscale_range(self, contrast=0.25, num_points=600, num_per_row=120):
        #print("::: Calculando valor min y maximo con zscale range :::")
        if len(self.image.shape) != 2:
            raise ValueError("input data is not an image")
        if contrast <= 0.0:
            contrast = 1.0
        if num_points > np.size(self.image) or num_points < 0:
            num_points = 0.5 * np.size(self.image)
        num_per_col = int(float(num_points) / float(num_per_row) + 0.5)
        xsize, ysize = self.image.shape
        row_skip = float(xsize - 1) / float(num_per_row - 1)
        col_skip = float(ysize - 1) / float(num_per_col - 1)
        data = []
        for i in range(num_per_row):
            x = int(i * row_skip + 0.5)
            for j in range(num_per_col):
                y = int(j * col_skip + 0.5)
                data.append(self.image[x, y])
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
        img = self.processed_image
        limg = np.arcsinh(img)
        limg = limg / limg.max()
        low = np.percentile(limg, min_percent)
        high = np.percentile(limg, max_percent)
        return limg, low, high
    def percentile_range(self, min_percent=3.0, max_percent=99.0, num_points=5000, num_per_row=250):
        #print("::: Calculate min and max value with percentile :::")
        if not 0 <= min_percent <= 100:
            raise ValueError("invalid value for min percent '%s'" % min_percent)
        elif not 0 <= max_percent <= 100:
            raise ValueError("invalid value for max percent '%s'" % max_percent)
        min_percent = float(min_percent) / 100.0
        max_percent = float(max_percent) / 100.0
        if len(self.image.shape) != 2:
            raise ValueError("input data is not an image")
        if num_points > np.size(self.image) or num_points < 0:
            num_points = 0.5 * np.size(self.image)
        num_per_col = int(float(num_points) / float(num_per_row) + 0.5)
        xsize, ysize = self.image.shape
        row_skip = float(xsize - 1) / float(num_per_row - 1)
        col_skip = float(ysize - 1) / float(num_per_col - 1)
        data = []
        for i in range(num_per_row):
            x = int(i * row_skip + 0.5)
            for j in range(num_per_col):
                y = int(j * col_skip + 0.5)
                data.append(self.image[x, y])
        data.sort()
        zmin = data[int(min_percent * len(data))]
        zmax = data[int(max_percent * len(data))]
        return zmin, zmax
    def autocontrast(self, type, original):
        #print("::: Procesando image :::")
        zmin = 0
        zmax = 0
        limg = 0
        if type == 1:
            zmin, zmax = self.zscale_range()
            if original:
                self.processed_image = np.where(self.image > zmin, self.image, zmin)
                self.processed_image = np.where(self.processed_image < zmax, self.processed_image, zmax)
            else:
                self.processed_image = np.where(self.processed_image > zmin, self.processed_image, zmin)
                self.processed_image = np.where(self.processed_image < zmax, self.processed_image, zmax)
            nonlinearity = 3.0
            nonlinearity = max(nonlinearity, 0.001)
            max_asinh = cmath.asinh(nonlinearity).real
            self.processed_image = (self.image.max() / max_asinh) * (np.arcsinh((self.processed_image - zmin) * (nonlinearity / (zmax - zmin))))
        elif type == 2:
            zmin, zmax = self.percentile_range(min_percent=3.0, max_percent=99.5, num_points=6000, num_per_row=350)
            if original:
                self.processed_image = np.where(self.image > zmin, self.image, zmin)
                self.processed_image = np.where(self.processed_image < zmax, self.processed_image, zmax)
            else:
                self.processed_image = np.where(self.processed_image > zmin, self.processed_image, zmin)
                self.processed_image = np.where(self.processed_image < zmax, self.processed_image, zmax)
            self.processed_image = (self.processed_image - zmin) * (self.image.max() / (zmax - zmin))
        elif type == 3:
            limg, zmin, zmax = self.arcsin_percentile(min_percent=3.0, max_percent=99.5)
            self.processed_image = skie.exposure.rescale_intensity(limg, in_range=(zmin, zmax))
            self.processed_image = self.processed_image * self.image.max()
        elif type == 4:
            zmin, zmax = self.percentile_range(min_percent=3.0, max_percent=99.5, num_points=6000, num_per_row=350)
            if original:
                self.processed_image = np.where(self.image > zmin, self.image, zmin)
                self.processed_image = np.where(self.processed_image < zmax, self.processed_image, zmax)
            else:
                self.processed_image = np.where(self.processed_image > zmin, self.processed_image, zmin)
                self.processed_image = np.where(self.processed_image < zmax, self.processed_image, zmax)
            nonlinearity = 3.0
            nonlinearity = max(nonlinearity, 0.001)
            max_asinh = cmath.asinh(nonlinearity).real
            self.processed_image = (self.image.max() / max_asinh) * (np.arcsinh((self.processed_image - zmin) * (nonlinearity / (zmax - zmin))))
        elif type == 5:
            nonlinearity = 3.0
            nonlinearity = max(nonlinearity, 0.001)
            max_asinh = cmath.asinh(nonlinearity).real
            self.processed_image = (self.image.max() / max_asinh) * (np.arcsinh((self.image - self.image.min()) * (nonlinearity / (self.image.max() - self.image.min()))))
        elif type == 6:
            nonlinearity = 3.0
            nonlinearity = max(nonlinearity, 0.001)
            max_asinh = cmath.asinh(nonlinearity).real
            self.processed_image = (self.image.max() / max_asinh) * (np.arcsinh((self.processed_image - zmin) * (nonlinearity / (zmax - zmin))))
        self.processed_image = self.processed_image.astype('uint16')
        return limg, zmin, zmax
    def fcm_3(self, min, anisodiff = False, median = False, gaussian = False):
        im = self.processed_image
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
        if min:
            maxlabel0 = np.max(label0)
            minlabel1 = np.min(label1)
            threshold = (maxlabel0 + minlabel1) / 2
        else:
            minlabel2 = np.min(label2)
            maxlabel2 = np.max(label2)
            threshold = minlabel2
        image_without_background = np.where(img_clustered > threshold, 1, 0)
        image_without_background.shape = im.shape
        image_without_background = image_without_background * im
        image_without_background = np.int16(image_without_background)
        self.processed_image = image_without_background
    def histogram(self):
        import cv2
        h = cv2.calcHist([self.image.ravel()], [0], None, [65536], [0,65536]) 
        return h
    def pfcm_2(self, path, name, anisodiff=True, median=False, gaussian=False):
        im = self.processed_image
        if anisodiff:
            I = 2 * ((im - im.min()) / (im.max() - im.min())) - 1
            sigma = mad_std(I)
            I = self.anisodiff(I,100,80,0.075,(1,1),sigma,2)
            I = (I - I.min()) / (I.max() - I.min())
            self.processed_image = self.processed_image.astype("uint16")
            self.save_image_tif(path, name + "_anisodiff.tif", True)
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
        
        colors = []
        color_groups = {0:np.array([255,0,0]),1:np.array([0,255,0])}
        for n in range(I.shape[0]):
            color = np.zeros([2])
            for c in range(U.shape[0]):
                color += color_groups[c]*U[c,n]
            colors.append(color)
        
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
        label_img = labels.astype("uint16")
        label_img16 = label_img * 65535
        self.processed_image = label_img16
        self.save_image_tif(path, name + "_pfcm_binary_", True)
        label_img.shape = im.shape
        binary_image = label_img * self.image
        binary_image = binary_image.astype("uint16")
        self.processed_image = binary_image
        self.save_image_tif(path, name + "_pfcm", True)
    def dice(self, pred, true, k = 1):
        intersection = np.sum(pred[true==k]) * 2.0
        dice = intersection / (np.sum(pred) + np.sum(true))
        return dice
    def top_hat(self, path, name):
        import cv2
    def save_image_png(self, path, name):
        self.autocontrast(2, True)
        self.processed_image = 255 * (self.processed_image - self.processed_image.min()) / (self.processed_image.max() - self.processed_image.min())
        self.processed_image = self.processed_image.astype('uint8')
        io.imsave(path + "/" + name + ".png", self.processed_image)
        return self.processed_image
    def save_image_tif(self, path, name, save_processed = False):
        if save_processed:
            io.imsave(path + "/" + name + ".tif", self.processed_image)
            return self.processed_image
        else:
            io.imsave(path + "/" + name + ".tif", self.image)
            return self.image
dir_images = "search_nebulae"
dir_result = "search_nebulae_autocontrast"
images = os.listdir(dir_images)
for img in images:
    print(img)
    pp = Preprocess(dir_images + "/" + img, True)
    #pp.save_image_tif(dir_result, img + "_original.tif", True)
    #pp.autocontrast(3, True)
    pp.save_image_png(dir_result, img)