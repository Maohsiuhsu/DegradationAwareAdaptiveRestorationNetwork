import os
import shutil
from pathlib import Path
from contextlib import redirect_stdout
import random
import numpy as np
import cv2
import multiprocessing

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchsummary import summary
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

from helper.init_handler import init_handler
from helper.yaml_processor import yaml_processor
from visualize.draw_loss import draw_csv_loss

from preprocess.load_data import inference_dataset_csv
from model.DenseUNet_ablation import DenseUNet_baseline, DenseUNet_BFM, DenseUNet_SEBlock_BFM, Residual_DenseUNet_SEBlock_BFM, All_in_One_Residual_DenseUNet

plt.switch_backend('agg')

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class Inferencer():
    def __init__(self, model, conf, device):
        self.conf = conf
        os.makedirs(conf["Testing"]["root_path"], exist_ok=True)
        os.makedirs(os.path.join(conf["Testing"]["result_dir"], 'enroll'), exist_ok=True)
        os.makedirs(os.path.join(conf["Testing"]["result_dir"], 'identify'), exist_ok=True)
        # load data
        enroll_inference_dataset = inference_dataset_csv(inference_dir=conf["Testing"]["dataset_dir"], dataset_version='enroll', conf=conf)
        identify_inference_dataset = inference_dataset_csv(inference_dir=conf["Testing"]["dataset_dir"], dataset_version='identify', conf=conf)
        self.inference_datasets = [enroll_inference_dataset, identify_inference_dataset]
    
        self.result_path_root = conf['Testing']['result_dir']
        ###

        # Load model
        self.device = device
        self.model = model.to(device)
        if conf["Testing"]["weight_path"] != "":
            self.model.load_state_dict(torch.load(conf["Testing"]["weight_path"], map_location=torch.device(self.device)))
        self.model.eval()
        ###

    @torch.no_grad()
    def inference(self):
        self.model.eval()
        for inference_dataset in self.inference_datasets:
            data_len = len(inference_dataset)
            for idx in tqdm(range(data_len), desc='Inference'):
                input_img_fname, inference_img, origin_img, img_wetQ_label, origin_h, origin_w = inference_dataset[idx]
                inference_img = inference_img.to(self.device).unsqueeze(0)
                if self.conf['Model']['type']=='airdunet' or self.conf['Model']['type']=='residual_denseunet_v1_air':
                    all_img_result, _, _  = self.model(inference_img, inference_img)
                    all_img_result = all_img_result.cpu().numpy()
                else:
                    all_img_result = self.model(inference_img,).cpu().numpy()
                all_img_result = all_img_result.transpose(0, 2, 3, 1)
                nb_img_result = np.squeeze(all_img_result).astype(np.float32)
    
                nb_img_result = inference_dataset.transform.postprocess(nb_img_result, origin_img, origin_h, origin_w)
                cv2.imwrite(os.path.join(self.result_path_root, inference_dataset.dataset_version, input_img_fname), nb_img_result)

    def draw_loss(self):
        if self.conf['Testing']['draw_loss']:
            if os.path.exists(self.conf['Training']['loss_csv_path']):
                draw_csv_loss(self.conf)

if __name__ == "__main__":
    conf_path = init_handler(module_name="inference.py")
    conf = yaml_processor(config_file_path=conf_path, is_debug=True)
    print("Inference Process")

    if conf['Model']['type']=='denseunet_baseline':
        model=DenseUNet_baseline(model_name=conf["Model"]["name"], conf=conf, device=device)
    elif conf['Model']['type']=='denseunet_bfm':
        model=DenseUNet_BFM(model_name=conf["Model"]["name"], conf=conf, device=device)
    elif conf['Model']['type']=='denseunet_se_bfm':
        model=DenseUNet_SEBlock_BFM(model_name=conf["Model"]["name"], seblock_pos=conf["Model"]["seblock_pos"], conf=conf, device=device)
    elif conf['Model']['type']=='residual_denseunet_se_bfm':
        model=Residual_DenseUNet_SEBlock_BFM(model_name=conf["Model"]["name"], seblock_pos=conf["Model"]["seblock_pos"], conf=conf, device=device)
    elif conf['Model']['type']=='airdunet' or conf['Model']['type']=='residual_denseunet_v1_air':
        model=All_in_One_Residual_DenseUNet(model_name=conf["Model"]["name"], seblock_pos=conf["Model"]["seblock_pos"], conf=conf, device=device)
        
    inferencer = Inferencer(model, conf=conf, device=device)
    inferencer.inference()
    # inferencer.draw_loss()
    print("Model_name:{model_name}".format(model_name=conf["Model"]["name"]))
    print("Inference Process Finish")
