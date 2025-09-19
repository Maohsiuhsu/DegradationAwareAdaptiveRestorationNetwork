import torch
import numpy as np

class model_checkpoint_saver():
    def __init__(self, model, ckp_path="", start_save_epoch=0, save_all_epoch_mode=False):
        super().__init__()
        self.model = model
        self.ckp_path = ckp_path
        self.start_save_epoch = start_save_epoch
        self.save_all_epoch_mode = save_all_epoch_mode
        self.epoch = 0
        self.less_val_loss = np.inf

    def step(self, total_val_loss, epoch):
        self.epoch = epoch
        if self.save_all_epoch_mode and epoch >= self.start_save_epoch:
            # save the checkpoint
            weight_path = self.ckp_path + "_{now_epoch}.pth".format(now_epoch=self.epoch)
            torch.save(self.model.state_dict(), weight_path)
            
            if total_val_loss < self.less_val_loss:
                # update loss
                self._info(total_val_loss, weight_path, save_mode=True, is_least=True)
                self.less_val_loss = total_val_loss
            else:
                self._info(total_val_loss, weight_path, save_mode=True, is_least=False)
        else:
            # don't save the checkpoint
            if total_val_loss < self.less_val_loss:
                # update loss
                self._info(total_val_loss, "", save_mode=False, is_least=True)
                self.less_val_loss = total_val_loss
            else:
                self._info(total_val_loss, "", save_mode=False, is_least=False)
    
    def _info(self, total_val_loss, weight_path="", save_mode=True, is_least=True):
        if save_mode:
            if is_least:
                print("Epoch {epoch} improved from {previous_best} to {current_best}, saving model to {filepath}.".format(epoch=self.epoch, previous_best = self.less_val_loss, current_best = total_val_loss, filepath = weight_path))
            else:
                print("Epoch {epoch} did not improve from {previous_best}, saving model to {filepath}.".format(epoch=self.epoch, previous_best = self.less_val_loss, filepath = weight_path))
        else:
            if is_least:
                print("Epoch {epoch} improved from {previous_best} to {current_best}.".format(epoch=self.epoch, previous_best = self.less_val_loss, current_best = total_val_loss))
            else:
                print("Epoch {epoch} did not improve from {previous_best}.".format(epoch=self.epoch, previous_best = self.less_val_loss))
    
    def last_save(self, epoch):
        if not self.save_all_epoch_mode:
            weight_path = self.ckp_path + "_last{now_epoch}.pth".format(now_epoch=epoch)
            torch.save(self.model.state_dict(), weight_path)
            print("Last Epoch {epoch} saving model to {filepath}.".format(epoch=self.epoch, filepath = weight_path))