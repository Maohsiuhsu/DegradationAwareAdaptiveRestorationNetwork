import os
import cv2
import numpy as np

def grayscale_mean(img, background_img):
    pixels = img.ravel()
    mask_pixels = background_img.ravel()
    mask_loc = np.array(np.where(mask_pixels==255))[0]
    mask_pixels = np.delete(pixels, mask_loc)
    gray_mean = np.mean(mask_pixels)
    return gray_mean

def grayscale_std(img, background_img):
    pixels = img.ravel()
    mask_pixels = background_img.ravel()
    mask_loc = np.array(np.where(mask_pixels==255))[0]
    mask_pixels = np.delete(pixels, mask_loc)
    gray_std = np.std(mask_pixels)
    return gray_std

def black_pixel_ratio(img, background_img):
    pixels = img.ravel()
    mask_pixels = background_img.ravel()
    mask_loc = np.array(np.where(mask_pixels==255))[0]
    mask_pixels = np.delete(pixels, mask_loc)

   # Cale Histogram    
    # black_value = len(np.array(np.where(mask_pixels==0))[0])
    black_value = len(np.array(np.where(mask_pixels>=235))[0])
    black_area_ratio = black_value / mask_pixels.size
    # print(black_value, mask_pixels.size)
    return black_area_ratio

def otsu_black_ratio(img, large_background_area):
    gray_area = img.size - large_background_area
    # Otsu Thershold
    ret, th = cv2.threshold(img ,0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # Cale Histogram
    hist = cv2.calcHist([th], [0], None, [256], [0, 256])
    hist = hist.reshape((256))
    
    black_value_ratio = int(hist[0]) / gray_area
    white_value_ratio = (int(hist[255]) - large_background_area) / gray_area
    diff_value_ratio = (int(hist[0]) - (int(hist[255]) - large_background_area)) / gray_area
    
    return black_value_ratio, white_value_ratio, diff_value_ratio

def sobel_gradient_magnitude(img):
    gX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    gY = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    magnitude = np.sqrt((gX ** 2) + (gY ** 2))
    grad_mean = np.mean(magnitude)
    return grad_mean

def lap_spectrum(img):
    # compute 2-th gradient (only leave postive number)
    lap_pos = cv2.Laplacian(img, -1, 3, 1)
    # fft
    H = np.fft.fft2(lap_pos)
    # shift (center is zero frequency)
    H_shift = np.fft.fftshift(H)
    # compute 2d spectrum (a+bi =>  sqrt(a^2+b^2))
    power_spectrum_2d = np.abs(H_shift)

    # shape
    Heigh, Weight = power_spectrum_2d.shape
    # normalize 2d spectrum
    power_spectrum_2d_norm = power_spectrum_2d / (Heigh * Weight)
    power_spectrum_2d_norm = power_spectrum_2d_norm

    return lap_pos, power_spectrum_2d_norm

def block_info(img):
    H, W = img.shape
    rms_list = []
    lap_spec_list = []
    lap_list = []
    # print(H, W) #(172, 32)
    # num_block_h = 5
    # num_block_w = 3
    num_block_h = 4
    num_block_w = 4
    block_h = H // num_block_h
    block_w = W // num_block_w
    # print(block_h, block_w) #(17,16)
    
    for i in range(num_block_h):
        for j in range(num_block_w):
            block = img[i*block_h: (i+1)*block_h, j*block_w: (j + 1)*block_w]
            # rms = np.sqrt(np.sum(block))/(block_h*block_w)
            # rms_list.append(rms)
            lap_pos, power_spectrum_2d_norm = lap_spectrum(block)
            lap_spec_list.append(np.sum(power_spectrum_2d_norm))
            lap_list.append(np.mean(lap_pos))

    for i in range(num_block_h):
        for j in range(num_block_w):
            block = img[H-(i+1)*block_h: H-i*block_h, W-(j+1)*block_w: W-j*block_w]
            # rms = np.sqrt(np.sum(block))/(block_h*block_w)
            # rms_list.append(rms)
            lap_pos, power_spectrum_2d_norm = lap_spectrum(block)
            lap_spec_list.append(np.sum(power_spectrum_2d_norm))
            lap_list.append(np.mean(lap_pos))

    lap_spec_list = np.asarray(lap_spec_list)
    lap_list = np.asarray(lap_list)
    block_lap_spec = np.mean(lap_spec_list)
    block_lap_spec = block_lap_spec / 1.625 * 15
    block_lap = np.mean(lap_list)
    block_lap = block_lap / 21.1 * 15

    return block_lap_spec, block_lap