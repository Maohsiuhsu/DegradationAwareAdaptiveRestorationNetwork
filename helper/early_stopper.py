import numpy as np

class early_stopper():
    def __init__(self, patience = 20, star_from_epoch = 10) -> None:
        self._the_smallest_val_loss = float("inf")
        self._time_count = 0
        self._patience = patience
        self._start_from_epoch = star_from_epoch
        self.early_stop = False

    def _update_loss(self, current_val_loss):
        if current_val_loss <= self._the_smallest_val_loss:
            self._the_smallest_val_loss = current_val_loss
            self._time_count = 0
        else:
            self._time_count += 1

    def stop_checker(self, current_epoch, current_val_loss):
        if self._start_from_epoch <= current_epoch:
            self._update_loss(current_val_loss)

            if self._time_count >= self._patience:
                self.early_stop = True
                print("Early stop at epoch: {epoch}".format(epoch=current_epoch))
            else:
                self.early_stop = False
                print("Early stop time count: {time_count}".format(time_count=self._time_count))
            
        else:
            self.early_stop = False