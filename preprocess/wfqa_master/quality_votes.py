import numpy as np
import glob
import os
import cv2
import shutil
from matplotlib import pyplot as plt
from tqdm import tqdm
from preprocess import find_large_background_area
from wet_quality_metrices import grayscale_mean, grayscale_std, black_pixel_ratio, otsu_black_ratio, sobel_gradient_magnitude, block_info

def bad_quality_vote(score_arr, bad_vote_thresh = 2):
    noise_vote = 0

    # gray_mean
    if(score_arr[0] in [5]):
        noise_vote += 1
    # gray_std
    if(score_arr[1] in [0, 5]):
        noise_vote += 1
    # black_pixel_r
    if(score_arr[2] in [0,  5]):
        noise_vote += 1
    # otsu_black_r
    if(score_arr[3] in [0, 5]):
        noise_vote += 1
    # gradient_m
    if(score_arr[4] in [0]):
        noise_vote += 1
    # power_spectrum_2d_norm
    if(score_arr[5] in [0, 5]):
        noise_vote += 1

    if (noise_vote <= 1) and ((score_arr[4]==0) or (score_arr[2] in [0, 5]) ):
        noise_vote = len(score_arr) + 1

    if noise_vote >= bad_vote_thresh:
        return True, noise_vote
    else:
        return False, noise_vote
    
def wet_quality_votes(score_arr):
    wet_score = -1
    dry_vote_result, dry_vote = dry_votes(score_arr)
    medium1_vote_result, medium1_vote = medium1_votes(score_arr)
    medium2_vote_result, medium2_vote = medium2_votes(score_arr)
    wet_vote_result, wet_vote = wet_votes(score_arr)
    if dry_vote_result:
        wet_score = 0
    elif wet_vote_result:
        wet_score = 2
    elif medium1_vote_result:
        wet_score = 1
    elif medium2_vote_result:
        wet_score = 1
    # else:
    #     votes = [dry_vote, medium1_vote, medium2_vote, wet_vote]
    #     wet_score = np.argmax(votes)
    return wet_score
    

def dry_votes(score_arr, dry_vote_thresh=2):
    dry_vote = 0
    # gray_std
    if(score_arr[1] in [0]):
        dry_vote += 1
    # otsu_black_r
    if(score_arr[3] in [0, 1, 2]):
        dry_vote += 1
    # gradient_m
    if(score_arr[4] in [8, 9]):
        dry_vote += 1
    # power_spectrum_2d_norm
    if(score_arr[5] in [8, 9]):
        dry_vote += 1

    if dry_vote >= dry_vote_thresh:
        return True, dry_vote
    else:
        return False, dry_vote
    
def medium1_votes(score_arr, vote_thresh=3):
    medium_vote = 0
    # gray_std
    if(score_arr[1] in [1, 2, 3]):
        medium_vote += 1
    # otsu_black_r
    if(score_arr[3] in [3, 4, 5, 6]):
        medium_vote += 1
    # gradient_m
    if(score_arr[4] in [5, 6, 7]):
        medium_vote += 1
     # power_spectrum_2d_norm
    if(score_arr[5] in [5, 6, 7]):
        medium_vote += 1

    if medium_vote >= vote_thresh:
        return True, medium_vote
    else:
        return False, medium_vote
    
def medium2_votes(score_arr, vote_thresh=3):
    medium_vote = 0
    # gray_std
    if(score_arr[1] in [4, 5, 6]):
        medium_vote += 1
    # otsu_black_r
    # if(score_arr[3] in [3, 4, 5, 6]):
    if(score_arr[3] in [3, 4, 5]):
        medium_vote += 1
    # gradient_m
    if(score_arr[4] in [3, 4]):
        medium_vote += 1
     # power_spectrum_2d_norm
    if(score_arr[5] in [2, 3, 4]):
        medium_vote += 1

    if medium_vote >= vote_thresh:
        return True, medium_vote
    else:
        return False, medium_vote
    
def wet_votes(score_arr, wet_vote_thresh=3):
    wet_vote = 0
    # gray_std
    if(score_arr[1] in [7, 8, 9]):
        wet_vote += 1
    # otsu_black_r
    if(score_arr[3] in [6, 7, 8]):
        wet_vote += 1
    # gradient_m
    if(score_arr[4] in [0, 1, 2]):
        wet_vote += 1
     # power_spectrum_2d_norm
    if(score_arr[5] in [0, 1]):
        wet_vote += 1

    if wet_vote >= wet_vote_thresh:
        return True, wet_vote
    else:
        return False, wet_vote

def find_standard_score_interval(metrices_array):
    mean = np.mean(metrices_array)
    std = np.std(metrices_array)
    std_score_arr = [0, mean-2*std, mean-std, mean, mean+std, mean+2*std, np.inf]
    return std_score_arr

if __name__ == "__main__":
    img_folder = fr'C:\Users\NMSOC\all_dataset\noise'
    result_folder = fr'C:\Users\NMSOC\all_dataset\quality_level\bad_quality'
    quality_metrices_num = 6

    gray_mean_arr = []
    gray_std_arr = []
    black_pixel_ratio_arr = []
    otsu_black_ratio_arr = []
    gradient_magnitude_arr = []
    lap_spectrum_arr = []

    for img_fname in tqdm(os.listdir(img_folder)):
        img_path = os.path.join(img_folder, img_fname)
        gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        gray_slide_img = gray_img[:,2:-2]

        background_img, large_background_loc, white_value = find_large_background_area(gray_slide_img)

        gray_mean = grayscale_mean(gray_slide_img, background_img)
        gray_std = grayscale_std(gray_slide_img, background_img)
        black_pixel_r = black_pixel_ratio(gray_img, background_img)
        otsu_black_r, otsu_white_r, _ = otsu_black_ratio(gray_slide_img, white_value)
        gradient_m = sobel_gradient_magnitude(gray_slide_img)
        lap_pos, power_spectrum_2d_norm = block_info(gray_slide_img)

        gray_mean_arr.append(gray_mean)
        gray_std_arr.append(gray_std)
        black_pixel_ratio_arr.append(black_pixel_r)
        otsu_black_ratio_arr.append(otsu_black_r)
        gradient_magnitude_arr.append(gradient_m)
        lap_spectrum_arr.append(power_spectrum_2d_norm)

    gray_mean_score_arr = find_standard_score_interval(gray_mean_arr)
    gray_std_score_arr = find_standard_score_interval(gray_std_arr)
    black_pixel_ratio_score_arr = find_standard_score_interval(black_pixel_ratio_arr)
    otsu_black_ratio_score_arr = find_standard_score_interval(otsu_black_ratio_arr)
    gradient_magnitude_score_arr = find_standard_score_interval(gradient_magnitude_arr)
    lap_spectrum_score_arr = find_standard_score_interval(lap_spectrum_arr)
    wet_q_scores = [gray_mean_score_arr, gray_std_score_arr, black_pixel_ratio_score_arr, otsu_black_ratio_score_arr, gradient_magnitude_score_arr, lap_spectrum_score_arr]

    score_counts_arrs = [0, 0, 0, 0, 0, 0, 0]
    # for m in quality_metrices_num:
    # score_counts_arrs.append([0, 0, 0, 0, 0, 0])
    
    for i in range(quality_metrices_num):
        if os.path.exists(os.path.join(result_folder, str(i))):
            shutil.rmtree(os.path.join(result_folder, str(i)))
        os.makedirs(os.path.join(result_folder, str(i)), exist_ok=True)

    if os.path.exists(os.path.join(result_folder, 'too_bad')):
        shutil.rmtree(os.path.join(result_folder, 'too_bad'))
    os.makedirs(os.path.join(result_folder, 'too_bad'), exist_ok=True)
    
    
    for id, img_fname in tqdm(enumerate(os.listdir(img_folder))):
        img_path = os.path.join(img_folder, img_fname)
        origin_img = cv2.imread(img_path)
        
        gray_mean = gray_mean_arr[id]
        gray_std = gray_std_arr[id]
        black_pixel_r = black_pixel_ratio_arr[id]
        otsu_black_r = otsu_black_ratio_arr[id]
        gradient_m = gradient_magnitude_arr[id]
        power_spectrum_2d_norm = lap_spectrum_arr[id]

        wet_q_values = [gray_mean, gray_std, black_pixel_r, otsu_black_r, gradient_m, power_spectrum_2d_norm]
        score_arr = [0, 0, 0, 0, 0, 0]

        for m_id in range(quality_metrices_num):
            wet_q = wet_q_values[m_id]
            wet_q_score = wet_q_scores[m_id]
            for score_id in range(len(wet_q_score) - 1):
                if (wet_q>=wet_q_score[score_id]) and (wet_q<wet_q_score[score_id+1]):
                    # cv2.imwrite(os.path.join(result_folder, metrices_path[m_id], str(score_id), img_fname), origin_img)
                    score_arr[m_id] = score_id
                    # score_counts_arrs[m_id][score_id] += 1

        noise_vote = 0

        # gray_mean
        if(score_arr[0] in [5]):
            noise_vote += 1
        # gray_std
        if(score_arr[1] in [0, 4, 5]):
            noise_vote += 1
        # black_pixel_r
        if(score_arr[2] in [0, 1, 4, 5]):
            noise_vote += 1
        # otsu_black_r
        if(score_arr[3] in [0, 5]):
            noise_vote += 1
        # gradient_m
        if(score_arr[4] in [0]):
            noise_vote += 1
        # power_spectrum_2d_norm
        if(score_arr[5] in [0, 5]):
            noise_vote += 1

        if (noise_vote <= 1) and ((score_arr[4]==0) or (score_arr[2] in [0, 4, 5]) ):
            cv2.imwrite(os.path.join(result_folder, 'too_bad', img_fname), origin_img)
            score_counts_arrs[-1]+=1
        else:
            score_counts_arrs[noise_vote]+=1
            cv2.imwrite(os.path.join(result_folder, str(noise_vote), img_fname), origin_img)
        
    print(score_counts_arrs)
    # for m_id in range(len(metrices_path)):
    #     print('wet_metrices: {name}'.format(name=metrices_path[m_id]))
    #     print(wet_q_scores[m_id][1:-1])
    #     print(score_counts_arrs[m_id])
