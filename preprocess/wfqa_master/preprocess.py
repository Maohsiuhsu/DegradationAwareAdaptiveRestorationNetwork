import numpy as np
import glob
import os
import cv2
from matplotlib import pyplot as plt

contour_area_limit = 200

img_path = fr"F:\Dataset\88x88\bad\bad_v4\058_L0_Unknown_13.bmp"

def find_contour(img):
    ret, thresh = cv2.threshold(img, 235, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def find_black_contour(img):
    """
    偵測影像中的黑色區域輪廓

    參數:
    - img: 灰階影像

    回傳:
    - contours: 偵測到的黑色區域輪廓
    - thresh: 進行二值化處理後的影像 (供除錯使用)
    """
    # 先做模糊處理，降低雜訊影響
    blurred = cv2.GaussianBlur(img, (7, 7), 0)

    # 反轉二值化，偵測黑色區域
    ret, thresh = cv2.threshold(blurred, 4, 255, cv2.THRESH_BINARY_INV)
    # ret, thresh = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY_INV)

    # 找出輪廓
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours, thresh


def find_contour_2(img, use_canny=False):
    """
    偵測影像中的輪廓，並回傳找到的輪廓。
    
    參數:
    - img: 灰階影像
    - use_canny: 是否使用 Canny 邊緣偵測 (預設 False)

    回傳:
    - contours: 偵測到的輪廓
    - thresh: 進行二值化處理後的影像 (供除錯使用)
    """
    # 先做模糊處理，降低雜訊影響
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    if use_canny:
        # 使用 Canny 邊緣檢測
        edges = cv2.Canny(blurred, 50, 150)
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours, edges
    else:
        # 使用 Otsu’s 門檻來自動選擇最佳閥值
        ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours, thresh


def find_large_background_area(img):
    # find large background area contour
    # contours, _ = find_contour_2(img,True)
    # contours = find_contour(img)
    contours, _ = find_black_contour(img)

    mask_img = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    for c in contours:
        contour_area = cv2.contourArea(c)
        if(contour_area > contour_area_limit):
            cv2.drawContours(mask_img, [c], -1, (255),
                             cv2.FILLED)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    erosion_img = cv2.erode(mask_img, kernel)
    # background_img = cv2.bitwise_and(img, erosion_img)

    # contour location
    y, x = np.where(erosion_img == 255)
    large_background_loc = np.column_stack((y, x))
    
    # contour area
    hist = cv2.calcHist([erosion_img], [0], None, [256], [0, 256])
    hist = hist.reshape((256))
    white_value = int(hist[255])

    return erosion_img, large_background_loc, white_value

if __name__ == "__main__":
    origin_img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)
    erosion_img, large_background_loc, white_value = find_large_background_area(gray_img)
    
    # Display the images
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    
    plt.subplot(1, 2, 2)
    plt.imshow(erosion_img, cmap='gray')
    plt.title(f'Erosion Image\nWhite Value: {white_value}')
    
    plt.show()
    
    # Print the large background locations
    print("Large Background Locations:")
    print(large_background_loc)