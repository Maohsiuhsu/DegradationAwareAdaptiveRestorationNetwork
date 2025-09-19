class loss_csv_logger():
    def __init__(self, acc_csv_path, loss_types = []):
        super().__init__()
        self.loss_types = self.loss_type_init(loss_types)
        self.losses_dict = dict()
        for loss_type in self.loss_types:
            self.losses_dict[loss_type] = 0
        self.losses_dict_array = dict()
        for loss_type in self.loss_types:
            self.losses_dict_array[loss_type] = []
        self.acc_csv_path = acc_csv_path
        self.epoch = 0
        with open(self.acc_csv_path, 'w') as f:
            csv_row = "epoch"
            for loss_type in self.loss_types:
                csv_row+=(", " + loss_type)
            csv_row += "\n"
            f.write(csv_row)
    
    def loss_type_init(self, loss_types):
        loss_types_catcher = []
        if len(loss_types) > 0:
            loss_types_catcher.append("train_loss")
            for loss_type in loss_types:
                loss_types_catcher.append("train_"+loss_type+"_loss")
            
            loss_types_catcher.append("valid_loss")
            for loss_type in loss_types:
                loss_types_catcher.append("val_"+loss_type+"_loss")
        else:
            loss_types_catcher.append("loss")
            loss_types_catcher.append("val_loss")
        return loss_types_catcher

    def step(self, losses_dict, epoch):
        self.epoch = epoch
        self.losses_dict = losses_dict
        for loss_type in self.loss_types:
            self.losses_dict_array[loss_type].append(losses_dict[loss_type])

        print("Epoch {epoch} losses_dict: {loss_dict}".format(epoch=self.epoch, loss_dict = losses_dict))

        with open(self.acc_csv_path, 'a') as f:
            csv_row = str(self.epoch)
            for loss_type in self.loss_types:
                csv_row+=(", " + str(losses_dict[loss_type]))
            csv_row+="\n"
            f.write(csv_row)