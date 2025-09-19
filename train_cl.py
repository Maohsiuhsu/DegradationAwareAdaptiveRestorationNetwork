'''
#! batch sampler: recognition model
#! anchor, negetive: https://github.com/XLearning-SCU/2022-CVPR-AirNet/blob/main/utils/dataset_utils.py
'''
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
from statistics import mean
import matplotlib.pyplot as plt

from helper.init_handler import init_handler
from helper.yaml_processor import yaml_processor
from helper.learning_rate_scheduler import learning_rate_scheduler
from helper.loss_csv_logger import loss_csv_logger
from helper.model_checkpoint_saver import model_checkpoint_saver
from helper.early_stopper import early_stopper
from helper.email_sender import model_training_complete, send_email

from visualize.plot_img import plot_imgs
from visualize.draw_loss import draw_loss

from preprocess.load_data import origin_dataset_csv_cl, BalancedBatchSampler, inference_dataset_csv
from model.DenseUNet_ablation import DenseUNet_baseline, DenseUNet_BFM, DenseUNet_SEBlock_BFM, Residual_DenseUNet_SEBlock_BFM, All_in_One_Residual_DenseUNet
from model.resnet import ResNet18
from loss.reconstruction_loss import reconstruction_loss

plt.switch_backend('agg')

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print("Using {device} device".format(device=device))

# Fix the seed for reproducibility
seed = 999
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
torch.backends.cudnn.benchmark = True
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
g = torch.Generator()
g.manual_seed(seed)

num_process = multiprocessing.cpu_count() // 2

class Trainer():
    def __init__(self, model, conf, device):
        self.conf = conf
        self.device = device

        os.makedirs(conf["Training"]["root_path"], exist_ok=True)
        os.makedirs(conf["Testing"]["root_path"], exist_ok=True)
        os.makedirs(conf["Testing"]["valid_result_dir"], exist_ok=True)

        ## Load Datasets
        origin_train_dataset = origin_dataset_csv_cl(dataset_dir=conf["Training"]["dataset_path"], dataset_version="train", device=device, testing_mode=conf["Training"]["debug"], conf=conf)
        origin_val_dataset = origin_dataset_csv_cl(dataset_dir=conf["Training"]["dataset_path"], dataset_version="val", device=device, testing_mode=conf["Training"]["debug"], conf=conf)
        train_batch_sampler = BalancedBatchSampler(origin_train_dataset, batch_size=conf["Model"]["batch_size"], n_classes=3, n_samples=conf["Model"]["batch_size"]//3)
        self.train_loader = DataLoader(dataset=origin_train_dataset, batch_sampler=train_batch_sampler, pin_memory=True, num_workers=num_process, worker_init_fn=seed_worker, generator=g)
        self.val_loader = DataLoader(dataset=origin_val_dataset, batch_size=conf["Model"]["batch_size"], shuffle=False, drop_last=True, pin_memory=True, num_workers=num_process, worker_init_fn=seed_worker, generator=g)
        # self.train_loader = DataLoader(dataset=origin_train_dataset, batch_sampler=train_batch_sampler, pin_memory=True, num_workers=num_process)
        # self.val_loader = DataLoader(dataset=origin_val_dataset, batch_size=conf["Model"]["batch_size"], shuffle=False, drop_last=True, pin_memory=True, num_workers=num_process)
        
        if conf['Testing']['testing']:
            enroll_inference_dataset = inference_dataset_csv(inference_dir=conf["Testing"]["dataset_dir"], dataset_version='enroll', conf=conf)
            identify_inference_dataset = inference_dataset_csv(inference_dir=conf["Testing"]["dataset_dir"], dataset_version='identify', conf=conf)
            self.inference_datasets = [ identify_inference_dataset]
        
            self.result_path_root = conf['Testing']['result_dir']
            
            os.makedirs(self.conf["Testing"]["result_dir"], exist_ok=True)
            os.makedirs(os.path.join(self.conf["Testing"]["result_dir"], 'enroll'), exist_ok=True)
            os.makedirs(os.path.join(self.conf["Testing"]["result_dir"], 'identify'), exist_ok=True)
        ## ##
        
        ## Recognition model ##
        self.recognition_model = ResNet18()
        self.recognition_model = self.recognition_model.to(device)
        self.recognition_model.load_state_dict(torch.load(fr"./weight/resnet_n9395_1023_clear_testnoFlip_randomtriplet_24_0601_2359_epoch_ft_9.pth"))
        for param in self.recognition_model.parameters():
            param.requires_grad = False
        self.recognition_model.eval()
        
        ## Model Parameter ##
        self.model = model.to(device)
        if conf['Training']['training'] and conf["Training"]["weight_path"] != "":
            try:
                self.model.load_state_dict(torch.load(conf["Training"]["weight_path"], map_location=torch.device(device)), strict=False)
            except:
                current_model_dict = self.model.state_dict()
                pretrained_state_dict = torch.load(conf["Training"]["weight_path"], map_location=torch.device(device))
                new_state_dict={k:v if v.size()==current_model_dict[k].size() else current_model_dict[k] for k,v in zip(current_model_dict.keys(), pretrained_state_dict.values()) }
                # for k, v in new_state_dict.items():
                #     print(k, v.shape)
                self.model.load_state_dict(new_state_dict, strict=False)
        elif conf['Testing']['testing'] and conf['Testing']['weight_path'] != '':
            self.model.load_state_dict(torch.load(conf["Testing"]["weight_path"], map_location=torch.device(device)))
        self.optimizer = torch.optim.Adam([
                {'params': self.model.degradation_encoder.parameters(), 'lr': conf["Model"]["cl_lr"], 'betas': (0.7, 0.9)},
                {'params': self.model.degradation_restorer.parameters(), 'lr': conf["Model"]["lr"]}
            ], lr=conf["Model"]["lr"])
        
        if conf["Training"]["debug"]:
            self.max_epoch = 10
            self.cl_epoch = 5
        else:
            self.max_epoch = conf["Model"]["max_epoch"]
            self.cl_epoch = conf["Model"]["cl_epoch"]
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.9, patience=5, cooldown=2)
        self.loss_logger = loss_csv_logger(acc_csv_path=conf["Training"]["loss_csv_path"], loss_types=["reconstruction", "recognition", "contrast"])
        self.ckp_saver = model_checkpoint_saver(model=self.model, ckp_path=conf["Training"]["save_weight_root"], start_save_epoch=self.cl_epoch, save_all_epoch_mode=True)
        
            
        self.reconstruction_rate = conf["Model"]["Loss"]["Reconstruction_Loss"]["rate"]
        self.reconstruction_loss_fn = reconstruction_loss(conf=conf, device=device)
        self.reconstruction_loss_fn = self.reconstruction_loss_fn.to(device)
        
        self.recognition_rate = conf["Model"]["Loss"]["Recognition_Loss"]["rate"]
        self.recognition_loss_fn = nn.CosineEmbeddingLoss()
        self.recognition_loss_fn = self.recognition_loss_fn.to(device)
        
        self.contrast_rate = conf["Model"]["Loss"]["Contrast_Loss"]["rate"]
        self.contrast_loss_fn = nn.CrossEntropyLoss()
        self.contrast_loss_fn = self.contrast_loss_fn.to(device)

        self.early_stop_scheduler = early_stopper(patience=20, star_from_epoch=self.cl_epoch+10)

    def train(self):
        print('='*20)
        print("Model Training: ")
        loss_metrics = {'train_loss': [], "train_reconstruction_loss": [], "train_recognition_loss": [], 'train_contrast_loss': []
                        , 'valid_loss': [], "val_reconstruction_loss": [], "val_recognition_loss": [], 'val_contrast_loss': []}
        for epoch in range(self.max_epoch):
            print("-"*20)
            print("Time: {now_time}, Epoch: {epoch}".format(now_time=datetime.now().strftime("%d/%m/%Y %H:%M:%S"), epoch=epoch))
            if epoch < self.cl_epoch:
                train_loss, train_reconstruction_loss, train_recognition_loss, train_contrast_loss = self.train_loop(epoch=epoch)
                valid_loss = np.inf
                valid_reconstruction_loss = np.inf
                valid_recognition_loss = np.inf
                valid_contrast_loss = np.inf
                print('lr:', self._get_lr())
                print(f'train_loss: {train_loss}, train_reconstruction_loss: {train_reconstruction_loss}, train_recognition_loss: {train_recognition_loss}, train_contrast_loss: {train_contrast_loss}')
                print(f'valid_loss: {valid_loss}, val_reconstruction_loss: {valid_reconstruction_loss}, val_recognition_loss: {valid_recognition_loss}, val_contrast_loss: {valid_contrast_loss}')
            else:
                train_loss, train_reconstruction_loss, train_recognition_loss, train_contrast_loss = self.train_loop(epoch=epoch)
                valid_loss, valid_reconstruction_loss, valid_recognition_loss, valid_contrast_loss = self.val_loop(epoch=epoch)
            
                self.scheduler.step(valid_loss)
                print('lr:', self._get_lr())
                print(f'train_loss: {train_loss}, train_reconstruction_loss: {train_reconstruction_loss}, train_recognition_loss: {train_recognition_loss}, train_contrast_loss: {train_contrast_loss}')
                print(f'valid_loss: {valid_loss}, val_reconstruction_loss: {valid_reconstruction_loss}, val_recognition_loss: {valid_recognition_loss}, val_contrast_loss: {valid_contrast_loss}')
                
            loss_metrics['train_loss'].append(train_loss)
            loss_metrics['train_reconstruction_loss'].append(train_reconstruction_loss)
            loss_metrics['train_recognition_loss'].append(train_recognition_loss)
            loss_metrics['train_contrast_loss'].append(train_contrast_loss)
            
            loss_metrics['valid_loss'].append(valid_loss)
            loss_metrics['val_reconstruction_loss'].append(valid_reconstruction_loss)
            loss_metrics['val_recognition_loss'].append(valid_recognition_loss)
            loss_metrics['val_contrast_loss'].append(valid_contrast_loss)
            
            losses_dict = {'train_loss': train_loss, "train_reconstruction_loss": train_reconstruction_loss, "train_recognition_loss": train_recognition_loss, 'train_contrast_loss': train_contrast_loss
                        , 'valid_loss': valid_loss, "val_reconstruction_loss": valid_reconstruction_loss, "val_recognition_loss": valid_recognition_loss, 'val_contrast_loss': valid_contrast_loss}
            
            self.loss_logger.step(losses_dict=losses_dict, epoch=epoch)
            self.ckp_saver.step(total_val_loss=valid_loss, epoch=epoch)

            if epoch >= self.cl_epoch:
                self.early_stop_scheduler.stop_checker(epoch, valid_loss)
                if self.early_stop_scheduler.early_stop:
                    break
            # self.inference()
        self.ckp_saver.last_save(epoch=epoch)
        draw_loss(loss_metrics, self.conf)
        print("Done!")
        print('='*20)

    def train_loop(self, epoch):
        self.model.train()
        loss_steps = []
        reconstruction_loss_steps = []
        recognition_loss_steps = []
        contrast_loss_steps = []
        contrast_loss_steps = []
        
        for input_imgs, gt_imgs, key_x_imgs, key_y_true_imgs, _ in tqdm(iter(self.train_loader), desc="Train"):
            self.optimizer.zero_grad()
            input_imgs = input_imgs.to(self.device, non_blocking=True)
            gt_imgs = gt_imgs.to(self.device, non_blocking=True)
            key_x_imgs = key_x_imgs.to(self.device, non_blocking=True)
            key_y_true_imgs = key_y_true_imgs.to(self.device, non_blocking=True)
            
            if epoch < self.cl_epoch:
                _, cl_output, cl_target, _ = self.model.degradation_encoder(x_query=input_imgs, x_key=key_x_imgs)
                train_contrast_loss = self.contrast_loss_fn(cl_output, cl_target)
                train_reconstruction_loss = 0.0
                train_recognition_loss = 0.0
                loss = train_contrast_loss
            else:
                pred_imgs, cl_output, cl_target = self.model(input_imgs, key_x_imgs)
            
                if self.recognition_rate != 0.0:
                    _, pred_embeddings = self.recognition_model(pred_imgs)
                    _, gt_embeddings = self.recognition_model(gt_imgs)
                    target = torch.ones(gt_embeddings.shape[0], device=self.device)
                
                train_reconstruction_loss = self.reconstruction_loss_fn(pred_imgs, gt_imgs)
                if self.recognition_rate != 0.0:
                    train_recognition_loss = self.recognition_loss_fn(pred_embeddings, gt_embeddings, target)
                else:
                    train_recognition_loss = 0.0
                train_contrast_loss = self.contrast_loss_fn(cl_output, cl_target)
                # loss = self.reconstruction_rate * train_reconstruction_loss + self.recognition_rate * train_recognition_loss + self.contrast_rate * train_contrast_loss
                loss = self.reconstruction_rate * train_reconstruction_loss + self.recognition_rate * train_recognition_loss
            
            loss.backward()
            self.optimizer.step()
            
            if epoch < self.cl_epoch:
                loss_steps.append(loss.detach().item())
                reconstruction_loss_steps.append(0.0)
                recognition_loss_steps.append(0.0)
                contrast_loss_steps.append(train_contrast_loss.detach().item())
            else:
                loss_steps.append(loss.detach().item())
                reconstruction_loss_steps.append(train_reconstruction_loss.detach().item())
                if self.recognition_rate != 0.0:
                    recognition_loss_steps.append(train_recognition_loss.detach().item())
                else:
                    recognition_loss_steps.append(0.0)
                contrast_loss_steps.append(train_contrast_loss.detach().item())
                    
        avg_loss = mean(loss_steps)
        avg_reconstruction_loss = mean(reconstruction_loss_steps)
        avg_recognition_loss = mean(recognition_loss_steps)
        avg_contrast_loss = mean(contrast_loss_steps)
        return avg_loss, avg_reconstruction_loss, avg_recognition_loss, avg_contrast_loss
    
    @torch.no_grad()
    def val_loop(self, epoch):
        self.model.eval()
        loss_steps = []
        reconstruction_loss_steps = []
        recognition_loss_steps = []
        contrast_loss_steps = []
        for input_imgs, gt_imgs, key_x_imgs, key_y_true_imgs, _ in tqdm(iter(self.val_loader), desc="Valid"):
            input_imgs = input_imgs.to(self.device)
            gt_imgs = gt_imgs.to(self.device)
            key_x_imgs = key_x_imgs.to(self.device)
            key_y_true_imgs = key_y_true_imgs.to(self.device)
            
            if epoch < self.cl_epoch:
                _, cl_output, cl_target, _ = self.model.degradation_encoder(x_query=input_imgs, x_key=key_x_imgs)
                val_contrast_loss = self.contrast_loss_fn(cl_output, cl_target)
                val_reconstruction_loss = 0.0
                val_recognition_loss = 0.0
                loss = val_contrast_loss
            else:
                pred_imgs, cl_output, cl_target = self.model(input_imgs, key_x_imgs)
                
                if self.recognition_rate != 0.0:
                    _, pred_embeddings = self.recognition_model(pred_imgs)
                    _, gt_embeddings = self.recognition_model(gt_imgs)
                    target = torch.ones(gt_embeddings.shape[0], device=self.device)
                
                val_reconstruction_loss = self.reconstruction_loss_fn(pred_imgs, gt_imgs)
                if self.recognition_rate != 0.0:
                    val_recognition_loss = self.recognition_loss_fn(pred_embeddings, gt_embeddings, target)
                else:
                    val_recognition_loss = 0.0
                val_contrast_loss = self.contrast_loss_fn(cl_output, cl_target)
                
                # loss = self.reconstruction_rate * val_reconstruction_loss + self.recognition_rate * val_recognition_loss + self.contrast_rate * val_contrast_loss
                loss = self.reconstruction_rate * val_reconstruction_loss + self.recognition_rate * val_recognition_loss
                
            if epoch < self.cl_epoch:
                loss_steps.append(loss.detach().item())
                reconstruction_loss_steps.append(0.0)
                recognition_loss_steps.append(0.0)
                contrast_loss_steps.append(val_contrast_loss.detach().item())
            else:
                loss_steps.append(loss.detach().item())
                reconstruction_loss_steps.append(val_reconstruction_loss.detach().item())
                if self.recognition_rate != 0.0:
                    recognition_loss_steps.append(val_recognition_loss.detach().item())
                else:
                    recognition_loss_steps.append(0.0)
                contrast_loss_steps.append(val_contrast_loss.detach().item())
        avg_loss = mean(loss_steps)
        avg_reconstruction_loss = mean(reconstruction_loss_steps)
        avg_recognition_loss = mean(recognition_loss_steps)
        avg_contrast_loss = mean(contrast_loss_steps)
        if epoch >= self.cl_epoch:
            plot_imgs(input_imgs_tensor=input_imgs, gt_imgs_tensor=gt_imgs, pred_imgs_tensor=pred_imgs, epoch=epoch, conf=self.conf)
        return avg_loss, avg_reconstruction_loss, avg_recognition_loss, avg_contrast_loss

    @torch.no_grad()
    def inference(self):        
        self.model.eval()
        for inference_dataset in self.inference_datasets:
            data_len = len(inference_dataset)
            for idx in tqdm(range(data_len), desc='Inference'):
                input_img_fname, inference_img, origin_img, img_wetQ_label, origin_h, origin_w = inference_dataset[idx]
                inference_img = inference_img.to(self.device).unsqueeze(0)
                all_img_result, _, _ = self.model(inference_img, inference_img)
                all_img_result = all_img_result.cpu().numpy()
                all_img_result = all_img_result.transpose(0, 2, 3, 1)
                
                nb_img_result = np.squeeze(all_img_result).astype(np.float32)

                nb_img_result = inference_dataset.transform.postprocess(nb_img_result, origin_img, origin_h, origin_w)
                cv2.imwrite(os.path.join(self.result_path_root, inference_dataset.dataset_version, input_img_fname), nb_img_result)
    
    def _get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
        
class Inferencer_for_all_epoch():
    def __init__(self, model, conf, device):
        self.conf = conf
        self.device = device
        if conf['Training']['training']:
            weight_prefix_name = conf['Training']['save_weight_root'] + '_'
        elif conf['Testing']['inference_all_epoch_mode']:
            weight_prefix_name = conf['Testing']['weight_prefix_name']
        self.result_path_root = os.path.join(conf['Testing']['result_dir'], str(Path(weight_prefix_name).name).split('_epoch_ft_')[0])
        
        ## model_weight_path_list ##
        model_weight_root = Path(weight_prefix_name).parent
        self.model_weight_path_list = sorted(model_weight_root.glob(str(Path(weight_prefix_name).name)+'*.pth')) 
        ## ##
        
        ## Load Datasets
        enroll_inference_dataset = inference_dataset_csv(inference_dir=conf["Testing"]["dataset_dir"], dataset_version='enroll', conf=conf)
        identify_inference_dataset = inference_dataset_csv(inference_dir=conf["Testing"]["dataset_dir"], dataset_version='identify', conf=conf)
        self.inference_datasets = [ identify_inference_dataset]
        ## ##
        
        ## Model Parameter ##
        self.model = model.to(device)
        self.model.eval()
        ## ##
    
    def inference_all_epoch(self):
        print('-'*80)
        print('inference for all epoch start')
        self.model.eval()
        for model_weight_path in tqdm(self.model_weight_path_list, desc='Inference_all_epoch'):
            epoch = str(model_weight_path.name).split('.pth')[0].split('_epoch_ft_')[-1]
            
            inference_epoch_result_root = os.path.join(self.result_path_root, f'epoch{epoch}_result')
            for inference_dataset in self.inference_datasets:
                os.makedirs(os.path.join(inference_epoch_result_root, inference_dataset.dataset_version), exist_ok=True)
            
            self.inference(model_weight_path, inference_epoch_result_root) 
        print('inference for all epoch finished')
        print('-'*80)
    
    @torch.no_grad()     
    def inference(self, model_weight_path, inference_epoch_result_root):
        self.model.eval()
        self.model.load_state_dict(torch.load(model_weight_path, map_location=torch.device(self.device)))
        for inference_dataset in self.inference_datasets:
            data_len = len(inference_dataset)
            for idx in range(data_len):
                input_img_fname, inference_img, origin_img, img_wetQ_label, origin_h, origin_w = inference_dataset[idx]
                # print(inference_img.shape)
                
                inference_img= inference_img.to(self.device).unsqueeze(0)
                all_img_result, _, _ = self.model(inference_img, inference_img)
                all_img_result = all_img_result.cpu().numpy()
                all_img_result = all_img_result.transpose(0, 2, 3, 1)
                # print(all_img_result.shape)
                nb_img_result = np.squeeze(all_img_result).astype(np.float32)
                # nb_img_result = np.clip(nb_img_result* 255.0, 0, 255).astype(np.uint8)
                # print(nb_img_result.shape)
                nb_img_result = inference_dataset.transform.postprocess(nb_img_result, origin_img, origin_h, origin_w)
                # print(nb_img_result.shape)
                nb_img_result = np.clip(nb_img_result * 255.0, 0, 255).astype(np.uint8)
                cv2.imwrite(os.path.join(inference_epoch_result_root, inference_dataset.dataset_version, input_img_fname), nb_img_result)
    
if __name__ == "__main__":
    # Init
    conf_path = Path(init_handler("train.py"))
    print("Training Process")

    conf = yaml_processor(config_file_path=conf_path, is_debug=True)
    testing_root_path = Path(conf["Testing"]["valid_result_dir"])
    os.makedirs(testing_root_path, exist_ok=True)
    shutil.copyfile(conf_path, testing_root_path / 'config.yaml')
    
    start_time = datetime.now()
    
    model=All_in_One_Residual_DenseUNet(model_name=conf["Model"]["name"], seblock_pos=conf["Model"]["seblock_pos"], conf=conf, device=device)
    model=model.to(device)
    # file_path = testing_root_path / (conf["Model"]["name"]+'.txt')
    # with open(file_path, 'w') as f:
    #     with redirect_stdout(f):
    #         summary(model, input_size = [(1, 88, 88), (1, 88, 88)], device=device)
    
    if conf['Training']['training']:
        trainer = Trainer(model=model, conf=conf, device=device)
        trainer.train()
        if conf['Testing']['testing']:
            if conf['Testing']['inference_all_epoch_mode']:
                inferencer = Inferencer_for_all_epoch(model=model, conf=conf, device=device)
                inferencer.inference_all_epoch()
            else:
                trainer.inference()
    elif conf['Testing']['testing']:
        if conf['Testing']['inference_all_epoch_mode']:
            inferencer = Inferencer_for_all_epoch(model=model, conf=conf, device=device)
            inferencer.inference_all_epoch()
        else:
            trainer = Trainer(model=model, conf=conf, device=device)
            trainer.inference()
        
    print("Send email!")
    content = model_training_complete(task_name = conf["Model"]["name"], server_name = '121_nasic01')
    # send_email(content)
        
    end_time =  datetime.now()
    print("Model_name:{model_name}".format(model_name=conf["Model"]["name"]))
    print("Total Spend time： {eclipse_time}".format(eclipse_time=(end_time - start_time)))