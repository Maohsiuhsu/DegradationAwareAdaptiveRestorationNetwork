''' reference
https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/132907dd272e2cc92e3c10e6c4e783a87ff8893d/transformer/Optim.py#L4
https://140.114.85.101:7333/focaltech/pgt-net/-/blob/eric/main/LossLearningRateScheduler.py
https://stackoverflow.com/questions/69576720/implementing-custom-learning-rate-scheduler-in-pytorch
https://www.kaggle.com/code/qiyuange/make-my-own-learning-rate-scheduler-pytorch/notebook
'''
class learning_rate_scheduler():
    def __init__(self, optimizer, decay_threshold = 0.005, decay_rate = 0.95, lr_csv_path = "", loss_type = 'loss'):
        super().__init__()
        self.optimizer = optimizer
        self.base_lr = self.get_lr_scale()
        self.decay_threshold = decay_threshold
        self.decay_rate = decay_rate
        self.loss_type = loss_type
        self.losses = []
        self.lr_csv_path = lr_csv_path
        self.epoch = 0

        with open(self.lr_csv_path, 'w') as f:
            csv_row = "epoch, lr\n"
            f.write(csv_row)

    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self.optimizer.zero_grad(set_to_none=True)

    def step(self):
        self.optimizer.step()

    def step_and_update_lr(self, loss, epoch):
        "Step with the inner optimizer"
        self.losses.append(loss)
        self.epoch = epoch
        self._update_learning_rate()
        self.optimizer.step()

    def get_lr_scale(self):       
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''
        if self.epoch >= 2:
            current_lr = self.get_lr_scale()

            '''loss_diff = self.losses[-2] - self.losses[-1]

            print("loss_diff: ", loss_diff)

            if loss_diff <= self.decay_threshold:
                self.lr = current_lr * self.decay_rate
                print("Epoch {epoch} changing learning rate from {current_lr} to {new_lr}.".format(epoch=self.epoch, current_lr=current_lr, new_lr=self.lr))
                self.lr = current_lr * self.decay_rate
            else:
                self.lr = current_lr
                print("Epoch {epoch} learning rate did not change from {current_lr}.".format(epoch=self.epoch, current_lr=current_lr))

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr

            if (self.epoch == 2):
                mode = 'w'
            else:
                mode = 'a'
            with open(self.lr_csv_path, mode) as f:
                f.write("{now_epoch:d}, {now_lr:f}\n".format(now_epoch = self.epoch, now_lr = current_lr))'''