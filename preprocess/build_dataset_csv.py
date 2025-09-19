import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import cv2
from dataclasses import make_dataclass
from sklearn.neighbors import KNeighborsClassifier

from wfqa_master.preprocess import find_large_background_area
from wfqa_master.wet_quality_metrices import grayscale_mean, grayscale_std, black_pixel_ratio, otsu_black_ratio, sobel_gradient_magnitude, block_info

knn_trainset_folder = fr"/home/ubuntu/4T_2/Howard/airdunet/preprocess/wfqa_master/trainset_best"
wet_quality_list = ['light', 'medium', 'heavy']
# set image size need to be saved in csv
input_img_shape = (88, 88)

def find_label(img_fname):
    img_label_ids = img_fname.split('.bmp')[0].split('_')
    label = img_label_ids[0] + '_' + img_label_ids[1] + '_' + img_label_ids[2] + '_' + img_label_ids[3]
    return label
def find_test_label(img_fname):
    img_label_ids = img_fname.split('.bmp')[0].split('_')
    label = img_label_ids[0] + '_' + img_label_ids[1] + '_' + img_label_ids[2] 
    return label
def build_csv(input_img_dir, gt_img_dir, csv_fname, flip_mode=False):
    ## Build KNN ##
    print('Build Classification Model')
    print('-'*80)
    # load data
    knn_x_train = np.load(os.path.join(knn_trainset_folder, 'x_train_q5.npy'))
    knn_y_train = np.load(os.path.join(knn_trainset_folder, 'y_train_q5.npy'))
    print("x_train shape: ", knn_x_train.shape)
    print("y_train shape: ", knn_y_train.shape)

    # build knn
    KNN = KNeighborsClassifier(n_neighbors=9, weights='distance')
    KNN.fit(knn_x_train, knn_y_train)
    print('-'*80)
    ## ##
    
    Dataset_Img_info = make_dataclass('Dataset_Img_info', [('input_img_fname', str), ('input_img_wet_quality', int), ('gt_img_fname', str), ('class_id', str), ('preprocess', str), ('input_img_dir', str), ('gt_img_dir', str)])
    img_info_lists = []
    print('Start to build csv')
    for img_fname in tqdm(os.listdir(input_img_dir)):
        if img_fname.find('.bmp') != -1:
            img_gt_fname = img_fname.replace('fake_B', 'real_A')
            # print(img_fname)
            # print(img_gt_fname)
            if os.path.exists(os.path.join(input_img_dir, img_fname)):
                
                img_path = os.path.join(input_img_dir, img_fname)
                gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                # print(gray_img.shape)
                if gray_img.shape == input_img_shape:
                    # print("test")
                    # gray_slide_img = gray_img[:,2:-2]
                    gray_slide_img = gray_img
                    background_img, large_background_loc, white_value = find_large_background_area(gray_slide_img)

                    gray_mean = grayscale_mean(gray_slide_img, background_img)
                    gray_std = grayscale_std(gray_slide_img, background_img)
                    black_pixel_r = black_pixel_ratio(gray_img, background_img)
                    otsu_black_r, _, _ = otsu_black_ratio(gray_slide_img, white_value)
                    gradient_m = sobel_gradient_magnitude(gray_slide_img)
                    lap_pos, power_spectrum_2d_norm = block_info(gray_slide_img)
                    
                    wet_q_values = np.array([gray_std, black_pixel_r, otsu_black_r, gradient_m, power_spectrum_2d_norm])
                    wet_q_values = np.reshape(wet_q_values, (1, -1))
                    pred = KNN.predict(wet_q_values)
                    level = pred[0]
                    
                    label = find_label(img_fname)
                    img_info_lists.append(Dataset_Img_info(img_fname, level, img_gt_fname, label, 'None', input_img_dir, gt_img_dir))
                    if flip_mode:
                        img_info_lists.append(Dataset_Img_info(img_fname, level, img_gt_fname, label+'_VF', 'VF', input_img_dir, gt_img_dir))
    print(len(img_info_lists))
    info_df = pd.DataFrame(img_info_lists)               
    info_df.to_csv(csv_fname, encoding='utf-8', index=False)
    
def build_testset_csv(input_img_dir, csv_fname, flip_mode=False):
    ## Build KNN ##
    print('Build Classification Model')
    print('-'*80)
    # load data
    knn_x_train = np.load(os.path.join(knn_trainset_folder, 'x_train_q5.npy'))
    knn_y_train = np.load(os.path.join(knn_trainset_folder, 'y_train_q5.npy'))
    print("x_train shape: ", knn_x_train.shape)
    print("y_train shape: ", knn_y_train.shape)

    # build knn
    KNN = KNeighborsClassifier(n_neighbors=9, weights='distance')
    KNN.fit(knn_x_train, knn_y_train)
    print('-'*80)
    ## ##
    
    Dataset_Img_info = make_dataclass('Dataset_Img_info', [('input_img_fname', str), ('input_img_wet_quality', int), ('gt_img_fname', str), ('class_id', str), ('preprocess', str), ('input_img_dir', str), ('gt_img_dir', str)])
    img_info_lists = []
    for img_fname in tqdm(os.listdir(input_img_dir)):
        if img_fname.find('.bmp') != -1:
            img_path = os.path.join(input_img_dir, img_fname)
            gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            gray_slide_img = gray_img
            # gray_slide_img = gray_img[:,2:-2]
            background_img, large_background_loc, white_value = find_large_background_area(gray_slide_img)

            gray_mean = grayscale_mean(gray_slide_img, background_img)
            gray_std = grayscale_std(gray_slide_img, background_img)
            black_pixel_r = black_pixel_ratio(gray_img, background_img)
            otsu_black_r, _, _ = otsu_black_ratio(gray_slide_img, white_value)
            gradient_m = sobel_gradient_magnitude(gray_slide_img)
            lap_pos, power_spectrum_2d_norm = block_info(gray_slide_img)
            
            wet_q_values = np.array([gray_std, black_pixel_r, otsu_black_r, gradient_m, power_spectrum_2d_norm])
            wet_q_values = np.reshape(wet_q_values, (1, -1))
            pred = KNN.predict(wet_q_values)
            level = pred[0]
            
            label = find_test_label(img_fname)
            img_info_lists.append(Dataset_Img_info(img_fname, level, "", label, 'None', input_img_dir, ""))
            if flip_mode:
                img_info_lists.append(Dataset_Img_info(img_fname, level, "", label+'_VF', 'VF', input_img_dir, ""))
    info_df = pd.DataFrame(img_info_lists)               
    info_df.to_csv(csv_fname, encoding='utf-8', index=False)
    
if __name__ == '__main__':
    ## build train set and validation set ##
    # input_root_path = fr"/local/SSD1/focal_tech/datasets/n9395_1023_v1_cyc_v2"
    # train_nb_input = os.path.join(input_root_path, "train", "non_binary", "x_train")
    # train_nb_gt = os.path.join(input_root_path, "train", "non_binary", "y_train")
    # val_nb_input = os.path.join(input_root_path, "val", "non_binary", "x_val")
    # val_nb_gt = os.path.join(input_root_path, "val", "non_binary", "y_val")
    
    input_root_path = fr"/mnt/ramdisk/88x88_cyclegan_v2_level5"
    train_nb_input = fr"/mnt/ramdisk/88x88_cyclegan_v2_level5/train/wet"
    train_nb_gt = fr"/mnt/ramdisk/88x88_cyclegan_v2_level5/train/normal"
    val_nb_input = fr"/mnt/ramdisk/88x88_cyclegan_v2_level5/val/wet"
    val_nb_gt = fr"/mnt/ramdisk/88x88_cyclegan_v2_level5/val/normal"

    # # save path
    result_root_path = input_root_path
    train_img_csv_path = os.path.join(result_root_path, "trainset.csv")
    val_img_csv_path = os.path.join(result_root_path, "valset.csv")
    
    build_csv(train_nb_input, train_nb_gt, train_img_csv_path)
    build_csv(val_nb_input, val_nb_gt, val_img_csv_path)
    ## ##
    
    ## build inference/test set ##
    testset_input_path = fr"/mnt/ramdisk/0616_testset_v2"
    test_enroll_input = os.path.join(testset_input_path, 'enroll')
    test_identify_input = os.path.join(testset_input_path, 'identify')
    
    test_enroll_csv_path = os.path.join(test_enroll_input, "testset.csv")
    test_identify_csv_path = os.path.join(test_identify_input, "testset.csv")
    
    build_testset_csv(test_enroll_input, test_enroll_csv_path)
    build_testset_csv(test_identify_input, test_identify_csv_path)
    # ##
    
    # data_frames = []
    # for csv_file in glob.glob(os.path.join(result_root_path, 'result_csv/*.csv')):
    #     df = pd.read_csv(csv_file)
    #     data_frames.append(df)
        
    # combined_df = pd.concat(data_frames, ignore_index=True)
    # combined_df.to_csv(test_matching_csv_path, index=False)