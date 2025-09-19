import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_msssim import SSIM

class reconstruction_loss(nn.Module):
    def __init__(self, conf, device=None):
        super().__init__()
        self.device = device

        self.mse_rate = conf["Model"]["Loss"]["Reconstruction_Loss"]["mse_rate"]
        self.lap_rate = conf["Model"]["Loss"]["Reconstruction_Loss"]["lap_rate"]
        self.ssim_rate = conf["Model"]["Loss"]["Reconstruction_Loss"]["ssim_rate"]

        self.lap_kernal = torch.tensor([[[[-1, -1, -1],[-1, 8, -1],[-1, -1, -1]]]], dtype=torch.float)
        self.lap_kernal = self.lap_kernal.to(device)
        self.ssim_module = SSIM(data_range=1, size_average=True, channel=1)

    def forward(self, pred, gt):
        gt_copy = gt.clone()
        pred_copy = pred.clone()
        reconstruction_loss = self.single_task_loss(gt_copy, pred_copy)
        return reconstruction_loss

    def single_task_loss(self, gt, pred):
        ## mse ##
        mse_loss = F.mse_loss(gt, pred)
        ## ##

        ## ssim ##
        gt_tensor = gt
        pred_tensor = pred
        ssim_loss = 1.0 - self.ssim_module(pred_tensor, gt_tensor)
        ## ##

        ## laplacian ##
        gt_lap = F.conv2d(input = gt_tensor.float(), weight = self.lap_kernal)
        pred_lap = F.conv2d(input = pred_tensor.float(), weight = self.lap_kernal)

        # origin: lap_loss = torch.square(gt_lap - pred_lap)
        lap_loss = torch.square(gt_lap - pred_lap)
        lap_loss = torch.mean(lap_loss)
        ## ##

        return self.mse_rate * mse_loss + self.lap_rate * lap_loss + self.ssim_rate * ssim_loss