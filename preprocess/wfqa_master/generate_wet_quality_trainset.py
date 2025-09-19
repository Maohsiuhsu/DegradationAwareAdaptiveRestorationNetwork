import numpy as np
import glob
import os
import cv2
import shutil
from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd

from preprocess import find_large_background_area
from wet_quality_metrices import grayscale_mean, grayscale_std, black_pixel_ratio, otsu_black_ratio, sobel_gradient_magnitude, block_info
from quality_votes import wet_quality_votes, bad_quality_vote

# set the path you need to build as humidity quality dataset
img_folder = fr"F:\Dataset\88x88\cyclegan\v1\wet"
# img_folder = fr"F:\Dataset\88x88\88x88_wet_v5_knn_v2\0"
# set the result path
result_folder = fr"F:\桌面\code\wfqa-master\88x88_cyclegan_v1_trainset_v2"
metrices_path = ['gray_mean', 'gray_std', 'black_pixel_ratio', 'otsu_black_ratio', 'gradient_magnitude', 'lap_spectrum']
quality_metrices_num = 6
quality_levels = 3

def compute_wet_quality(img_root, img_name_list):
    gray_mean_arr = []
    gray_std_arr = []
    black_pixel_ratio_arr = []
    otsu_black_ratio_arr = []
    gradient_magnitude_arr = []
    lap_spectrum_arr = []

    for img_fname in tqdm(img_name_list):
        img_path = os.path.join(img_root, img_fname)
        gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # gray_slide_img = gray_img[:,2:-2]
        gray_slide_img = gray_img
        background_img, large_background_loc, white_value = find_large_background_area(gray_slide_img)

        gray_mean = grayscale_mean(gray_slide_img, background_img)
        gray_std = grayscale_std(gray_slide_img, background_img)
        black_pixel_r = black_pixel_ratio(gray_img, background_img)
        # otsu_black_r, _, _ = otsu_black_ratio(gray_slide_img, white_value)
        _, otsu_black_r, _ = otsu_black_ratio(gray_slide_img, white_value)       # change to while value
        gradient_m = sobel_gradient_magnitude(gray_slide_img)
        lap_pos, power_spectrum_2d_norm = block_info(gray_slide_img)

        gray_mean_arr.append(gray_mean)
        gray_std_arr.append(gray_std)
        black_pixel_ratio_arr.append(black_pixel_r)
        otsu_black_ratio_arr.append(otsu_black_r)
        gradient_magnitude_arr.append(gradient_m)
        lap_spectrum_arr.append(power_spectrum_2d_norm)

    return gray_mean_arr, gray_std_arr, black_pixel_ratio_arr, otsu_black_ratio_arr, gradient_magnitude_arr, lap_spectrum_arr

def find_standard_score_interval(metrices_array, std_mode = True):
    mean = np.mean(metrices_array)
    std = np.std(metrices_array)
    std_score_arr = [0, mean-2*std, mean-std, mean, mean+std, mean+2*std, np.inf]

    percent_score_arr = [np.percentile(metrices_array, i*10) for i in range(11)]
    if std_mode:
        return std_score_arr
    else:
        return percent_score_arr

def create_folders(folder_num):
    for m in metrices_path:
        for i in range(folder_num):
            if os.path.exists(os.path.join(result_folder, m, str(i))):
                shutil.rmtree(os.path.join(result_folder, m, str(i)))
            os.makedirs(os.path.join(result_folder, m, str(i)), exist_ok=True)
    
def create_bad_quality_folders(bad_levels):
    for i in range(bad_levels):
        if os.path.exists(os.path.join(result_folder, 'bad', str(i))):
            shutil.rmtree(os.path.join(result_folder, 'bad', str(i)))
        os.makedirs(os.path.join(result_folder, 'bad', str(i)), exist_ok=True)
    
def create_wet_level_folders(quality_levels):
    for i in range(-1, quality_levels):
        if os.path.exists(os.path.join(result_folder, str(i))):
            shutil.rmtree(os.path.join(result_folder, str(i)))
        os.makedirs(os.path.join(result_folder, str(i)), exist_ok=True)

def quality_score_evaluate(wet_q_values, wet_q_scores):
    wet_level_arr = [0 for i in range(quality_metrices_num)]
    for m_id in range(quality_metrices_num):
            wet_q = wet_q_values[m_id]
            wet_q_score = wet_q_scores[m_id]
            for score_id in range(len(wet_q_score) - 1):
                if (wet_q>=wet_q_score[score_id]) and (wet_q<wet_q_score[score_id+1]):
                    wet_level_arr[m_id] = score_id
    return wet_level_arr

if __name__ == "__main__":
    ## Compute Wet Quality Metrices ##
    print('Compute Wet Quality Metrices')
    print('-'*80)
    all_img_name_list = os.listdir(img_folder)
    
    # for each image, compute 6 wet quality metrices
    gray_mean_arr, gray_std_arr, black_pixel_ratio_arr, otsu_black_ratio_arr, gradient_magnitude_arr, lap_spectrum_arr = compute_wet_quality(img_folder, all_img_name_list)

    # get the stancard scores of each quality metrix
    gray_mean_score_arr = find_standard_score_interval(gray_mean_arr)
    gray_std_score_arr = find_standard_score_interval(gray_std_arr)
    black_pixel_ratio_score_arr = find_standard_score_interval(black_pixel_ratio_arr)
    otsu_black_ratio_score_arr = find_standard_score_interval(otsu_black_ratio_arr)
    gradient_magnitude_score_arr = find_standard_score_interval(gradient_magnitude_arr)
    lap_spectrum_score_arr = find_standard_score_interval(lap_spectrum_arr)
    wet_q_scores = [gray_mean_score_arr, gray_std_score_arr, black_pixel_ratio_score_arr, otsu_black_ratio_score_arr, gradient_magnitude_score_arr, lap_spectrum_score_arr]
    print('-'*80)
    ## ##
    
    ## Data Cleaning: remove bad quality fingerprints & Save to CSV ##
    print('Remove Bad Quality Images(Outlier)')
    print('-'*80)
    wet_level_df = pd.DataFrame(columns=['img_fname', 'gray_mean', 'gray_std', 'black_pixel_ratio', 'otsu_black_ratio', 'gradient_magnitude', 'lap_spectrum', 'gray_mean_std_score', 'gray_std_std_score', 'black_pixel_ratio_std_score', 'otsu_black_ratio_std_score', 'gradient_magnitude_std_score', 'lap_spectrum_std_score', 'gray_mean_percent_score', 'gray_std_percent_score', 'black_pixel_ratio_percent_score', 'otsu_black_ratio_percent_score', 'gradient_magnitude_percent_score', 'lap_spectrum_percent_score', 'bad_votes', 'wet_votes'])
    bad_vote_counts_arrs = [0 for i in range(quality_metrices_num+2)]
    bad_img_name_list = []
    for id, img_fname in tqdm(enumerate(all_img_name_list)):
        # get the wet quality metrices values of the image
        gray_mean = gray_mean_arr[id]
        gray_std = gray_std_arr[id]
        black_pixel_r = black_pixel_ratio_arr[id]
        otsu_black_r = otsu_black_ratio_arr[id]
        gradient_m = gradient_magnitude_arr[id]
        power_spectrum_2d_norm = lap_spectrum_arr[id]
        # put the metrices values into a list
        wet_q_values = [gray_mean, gray_std, black_pixel_r, otsu_black_r, gradient_m, power_spectrum_2d_norm]

        # classify the score of each wet quality metrix
        wet_level_arr = quality_score_evaluate(wet_q_values, wet_q_scores)
        # compute the bad quality degree
        bad_result, bad_vote = bad_quality_vote(wet_level_arr, bad_vote_thresh = 1)
        bad_vote_counts_arrs[bad_vote] += 1

        # save the image quality info in csv
        new_row = pd.DataFrame({'img_fname': img_fname, 'gray_mean': gray_mean, 'gray_std': gray_std, 'black_pixel_ratio': black_pixel_r, 
                                'otsu_black_ratio': otsu_black_r, 'gradient_magnitude': gradient_m, 'lap_spectrum': power_spectrum_2d_norm, 
                                'gray_mean_std_score': wet_level_arr[0], 'gray_std_std_score': wet_level_arr[1], 'black_pixel_ratio_std_score': wet_level_arr[2], 
                                'otsu_black_ratio_std_score': wet_level_arr[3], 'gradient_magnitude_std_score': wet_level_arr[4], 'lap_spectrum_std_score': wet_level_arr[5],
                                'gray_mean_percent_score': -1, 'gray_std_percent_score': -1, 'black_pixel_ratio_percent_score': -1,
                                'otsu_black_ratio_percent_score': -1, 'gradient_magnitude_percent_score': -1, 'lap_spectrum_percent_score': -1, 'bad_votes': bad_vote}, index=[0])
        wet_level_df = pd.concat([new_row, wet_level_df.loc[:]]).reset_index(drop=True)
        
        if bad_result:
            bad_img_name_list.append(img_fname)       
    
    # create the bad quality folder
    create_bad_quality_folders(quality_metrices_num+2)
    # put the bad quality images to the folder
    for index, row in tqdm(wet_level_df.iterrows()):
        img_fname = row['img_fname']
        if row['bad_votes'] != 0:
            shutil.copyfile(os.path.join(img_folder, img_fname), os.path.join(result_folder, 'bad', str(row['bad_votes']), img_fname))
           
    bad_img_name_list = set(bad_img_name_list)
    # print('bad_votes_nums: {num}'.format(bad_vote_counts_arrs))
    # print('bad_quality_image_num: {num}'.format(num=len(bad_img_name_list)))
    print('bad_votes_nums:', bad_vote_counts_arrs)
    print('bad_quality_image_num:', len(bad_img_name_list))
    print('-'*80)
    ## ##
    
    ## Recompute Wet Quality Metrices for Good Quality Images ##
    print('Recompute Wet Quality Metrices for Good Quality Images')
    print('-'*80)
    good_img_name_list = [i for i in all_img_name_list if i not in bad_img_name_list]
    gray_mean_arr, gray_std_arr, black_pixel_ratio_arr, otsu_black_ratio_arr, gradient_magnitude_arr, lap_spectrum_arr = compute_wet_quality(img_folder, good_img_name_list)

    gray_mean_score_arr = find_standard_score_interval(gray_mean_arr, std_mode=False)
    gray_std_score_arr = find_standard_score_interval(gray_std_arr, std_mode=False)
    black_pixel_ratio_score_arr = find_standard_score_interval(black_pixel_ratio_arr, std_mode=False)
    otsu_black_ratio_score_arr = find_standard_score_interval(otsu_black_ratio_arr, std_mode=False)
    gradient_magnitude_score_arr = find_standard_score_interval(gradient_magnitude_arr, std_mode=False)
    lap_spectrum_score_arr = find_standard_score_interval(lap_spectrum_arr, std_mode=False)
    wet_q_scores = [gray_mean_score_arr, gray_std_score_arr, black_pixel_ratio_score_arr, otsu_black_ratio_score_arr, gradient_magnitude_score_arr, lap_spectrum_score_arr]
    print('-'*80)

    score_counts_arrs = []
    for m in metrices_path:
        score_counts_arrs.append([0 for _ in range(len(gray_mean_score_arr)-1)])
        
    for id, img_fname in tqdm(enumerate(good_img_name_list)):
        gray_mean = gray_mean_arr[id]
        gray_std = gray_std_arr[id]
        black_pixel_r = black_pixel_ratio_arr[id]
        otsu_black_r = otsu_black_ratio_arr[id]
        gradient_m = gradient_magnitude_arr[id]
        power_spectrum_2d_norm = lap_spectrum_arr[id]

        wet_q_values = [gray_mean, gray_std, black_pixel_r, otsu_black_r, gradient_m, power_spectrum_2d_norm]

        wet_level_arr = quality_score_evaluate(wet_q_values, wet_q_scores)
        
        wet_level_df.loc[wet_level_df['img_fname'] == img_fname, 'gray_mean_percent_score'] = wet_level_arr[0]
        wet_level_df.loc[wet_level_df['img_fname'] == img_fname, 'gray_std_percent_score'] = wet_level_arr[1]
        wet_level_df.loc[wet_level_df['img_fname'] == img_fname, 'black_pixel_ratio_percent_score'] = wet_level_arr[2]
        wet_level_df.loc[wet_level_df['img_fname'] == img_fname, 'otsu_black_ratio_percent_score'] = wet_level_arr[3]
        wet_level_df.loc[wet_level_df['img_fname'] == img_fname, 'gradient_magnitude_percent_score'] = wet_level_arr[4]
        wet_level_df.loc[wet_level_df['img_fname'] == img_fname, 'lap_spectrum_percent_score'] = wet_level_arr[5]
        
        for m_id in range(quality_metrices_num):
            score_id = wet_level_arr[m_id]
            score_counts_arrs[m_id][score_id] += 1
    
    for m_id in range(len(metrices_path)):
        print('wet_metrices: {name}'.format(name=metrices_path[m_id]))
        print('standard_interval_scores: {s_scores}'.format(s_scores=wet_q_scores[m_id][1:-1]))
        # print('standard_interval_nums: {num}'.format(num=score_counts_arrs[m_id]))
        print('standard_interval_nums:', score_counts_arrs[m_id])
    print('-'*80)
    ## ##
    
    for index, row in tqdm(wet_level_df.iterrows()):
        if row['bad_votes'] == 0:
            wet_q_values = [row['gray_mean'], row['gray_std'], row['black_pixel_ratio'], row['otsu_black_ratio'], row['gradient_magnitude'], row['lap_spectrum']]
            wet_level_arr = quality_score_evaluate(wet_q_values, wet_q_scores)
            wet_score = wet_quality_votes(wet_level_arr)
            wet_level_df.loc[wet_level_df['img_fname'] == row['img_fname'], 'wet_votes'] = wet_score
        else:
            wet_level_df.loc[wet_level_df['img_fname'] == row['img_fname'], 'wet_votes'] = -1
            
    for i in range(quality_levels):
        print('i: ', len(wet_level_df[wet_level_df['wet_votes']==i]))
        
    create_wet_level_folders(quality_levels)
    wet_level_df.to_csv(os.path.join(result_folder, 'wet_quality_estimate.csv'))
    
    for index, row in tqdm(wet_level_df.iterrows()):
        img_fname = row['img_fname']
        wet_score = row['wet_votes']
        
        img_path = os.path.join(img_folder, img_fname)
        shutil.copyfile(img_path, os.path.join(result_folder, str(wet_score), img_fname))
        
    level_0_df = wet_level_df[wet_level_df['wet_votes']==0]
    level_1_df = wet_level_df[wet_level_df['wet_votes']==1]
    level_2_df = wet_level_df[wet_level_df['wet_votes']==2]
    
    x_train = np.empty([len(level_0_df)+len(level_1_df)+len(level_2_df), 5])
    y_train = []
    cnt = 0

    for index, row in tqdm(level_0_df.iterrows()):
        wet_q_values = np.array([row['gray_std'], row['black_pixel_ratio'], row['otsu_black_ratio'], row['gradient_magnitude'], row['lap_spectrum']])
        x_train[cnt] = wet_q_values
        y_train.append(0)
        cnt+=1

    for index, row in tqdm(level_1_df.iterrows()):
        wet_q_values = np.array([row['gray_std'], row['black_pixel_ratio'], row['otsu_black_ratio'], row['gradient_magnitude'], row['lap_spectrum']])
        x_train[cnt] = wet_q_values
        y_train.append(1)
        cnt+=1
        
    for index, row in tqdm(level_2_df.iterrows()):
        wet_q_values = np.array([row['gray_std'], row['black_pixel_ratio'], row['otsu_black_ratio'], row['gradient_magnitude'], row['lap_spectrum']])
        x_train[cnt] = wet_q_values
        y_train.append(2)
        cnt+=1
        
    y_train = np.asarray(y_train) 

    print("X_train shape: ", x_train.shape)
    print("y_train shape: ", y_train.shape)
    np.save(os.path.join(result_folder, 'x_train_q5.npy'), x_train)
    np.save(os.path.join(result_folder, 'y_train_q5.npy'), y_train)