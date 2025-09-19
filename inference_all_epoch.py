import os
import shutil
from pathlib import Path
from contextlib import redirect_stdout
import random
import numpy as np
import cv2
import multiprocessing
from datetime import datetime
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchsummary import summary

from helper.init_handler import init_handler
from helper.yaml_processor import yaml_processor
from helper.email_sender import model_inference_complete, send_email

from preprocess.load_data import inference_dataset_csv
from model.DenseUNet_ablation import DenseUNet_baseline, DenseUNet_BFM, DenseUNet_SEBlock_BFM, Residual_DenseUNet_SEBlock_BFM, All_in_One_Residual_DenseUNet

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Fix the seed for reproducibility
seed = 999
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
g = torch.Generator()
g.manual_seed(seed)

num_process = multiprocessing.cpu_count() // 2

class Inferencer_for_all_epoch():
    def __init__(self, model, conf, device):
        self.conf = conf
        self.device = device
        weight_prefix_name = conf['Testing']['weight_prefix_name']
        self.result_path_root = os.path.join(conf['Testing']['result_dir'], str(Path(weight_prefix_name).name).split('_epoch_ft_')[0])
        
        ## model_weight_path_list ##
        model_weight_root = Path(weight_prefix_name).parent
        self.model_weight_path_list = sorted(model_weight_root.glob(str(Path(weight_prefix_name).name)+'*.pth')) 
        ## ##
        
        ## Load Datasets
        # enroll_inference_dataset = inference_dataset_csv(inference_dir=conf["Testing"]["dataset_dir"], dataset_version='enroll', conf=conf)
        identify_inference_dataset = inference_dataset_csv(inference_dir=conf["Testing"]["dataset_dir"], dataset_version='identify', conf=conf)
        # self.inference_datasets = [enroll_inference_dataset, identify_inference_dataset]
        self.inference_datasets = [identify_inference_dataset]
        ## ##
        
        ## Model Parameter ##
        self.model = model.to(device)
        self.model.eval()
        ## ##
    
    def inference_all_epoch(self):
        self.model.eval()
        for model_weight_path in tqdm(self.model_weight_path_list, desc='Inference_all_epoch'):
            epoch = str(model_weight_path.name).split('.pth')[0].split('_epoch_ft_')[-1]
            inference_epoch_result_root = os.path.join(self.result_path_root, f'epoch{epoch}_result')
            for inference_dataset in self.inference_datasets:
                os.makedirs(os.path.join(inference_epoch_result_root, inference_dataset.dataset_version), exist_ok=True)
            
            self.inference(model_weight_path, inference_epoch_result_root) 
    
    @torch.no_grad()     
    def inference(self, model_weight_path, inference_epoch_result_root):
        self.model.eval()
        self.model.load_state_dict(torch.load(model_weight_path, map_location=torch.device(self.device)))
        for inference_dataset in self.inference_datasets:
            data_len = len(inference_dataset)
            for idx in range(data_len):
                input_img_fname, inference_img, origin_img, img_wetQ_label, origin_h, origin_w = inference_dataset[idx]
                inference_img = inference_img.to(self.device).unsqueeze(0)
                if self.conf['Model']['type']=='airdunet' or self.conf['Model']['type']=='residual_denseunet_v1_air':
                    all_img_result, _, _  = self.model(inference_img, inference_img)
                    all_img_result = all_img_result.cpu().numpy()
                else:
                    all_img_result = self.model(inference_img).cpu().numpy()
                all_img_result = all_img_result.transpose(0, 2, 3, 1)
                nb_img_result = np.squeeze(all_img_result).astype(np.float32)
    
                nb_img_result = inference_dataset.transform.postprocess(nb_img_result, origin_img, origin_h, origin_w)
                cv2.imwrite(os.path.join(inference_epoch_result_root, inference_dataset.dataset_version, input_img_fname), nb_img_result)
    
if __name__ == "__main__":
    # Init
    conf_path = Path(init_handler("train.py"))
    print("Training Process")

    conf = yaml_processor(config_file_path=conf_path, is_debug=True)
    testing_root_path = Path(conf["Testing"]["result_dir"])
    os.makedirs(testing_root_path, exist_ok=True)
    shutil.copyfile(conf_path, testing_root_path / 'config.yaml')
    
    start_time = datetime.now()
    
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
    
    model=model.to(device)
    
    if not (conf['Model']['type']=='airdunet' or conf['Model']['type']=='residual_denseunet_v1_air'):
        file_path = testing_root_path / (conf["Model"]["name"]+'.txt')
        with open(file_path, 'w') as f:
            with redirect_stdout(f):
                summary(model, input_size = [(1, 176, 40)], device=device)
    
    if conf['Testing']['testing'] and conf['Testing']['inference_all_epoch_mode']:
        trainer = Inferencer_for_all_epoch(model=model, conf=conf, device=device)
        trainer.inference_all_epoch()
        
    print("Send email!")
    # content = model_inference_complete(task_name = conf["Model"]["name"], server_name = '121_nasic01')
    # send_email(content)
        
    end_time =  datetime.now()
    print("Model_name:{model_name}".format(model_name=conf["Model"]["name"]))
    print("Total Spend time： {eclipse_time}".format(eclipse_time=(end_time - start_time)))