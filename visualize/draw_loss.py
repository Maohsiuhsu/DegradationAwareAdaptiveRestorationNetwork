import numpy as np
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def draw_loss(loss_metrics, conf):
    print("="*20)
    print("Draw Loss: ")
    min_train_loss = min(loss_metrics['train_loss'])
    min_train_loss_epoch = loss_metrics['train_loss'].index(min_train_loss)
    min_valid_loss = min(loss_metrics['valid_loss'])
    min_valid_loss_epoch = loss_metrics['valid_loss'].index(min_valid_loss)
    print(f'Best pretrained training loss: {min_train_loss:.3f} / at epoch {min_train_loss_epoch}')
    print(f'Best pretrained validate loss: {min_valid_loss:.3f} / at epoch {min_valid_loss_epoch}')

    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 10), dpi=100)
    ax1.set_title('Loss')
    ax1.plot(range(len(loss_metrics['train_loss'])), loss_metrics['train_loss'], color='#4E79A7', linewidth=2, label='train loss')
    ax1.plot(range(len(loss_metrics['valid_loss'])), loss_metrics['valid_loss'], color='#A0CBE8', linewidth=2, label='validation loss')
    ax1.legend(fontsize=12)
    ax1.set_xlabel('epochs', fontsize=15)
    ax1.set_ylabel('loss', fontsize=15)
    ax1.set_ylim(bottom=0)
    ax1.legend(loc='best')
    plt.tight_layout()
    fig1.savefig(os.path.join(conf['Testing']['valid_result_dir'], conf['Model']['name']+'.png'))
    plt.close(fig1)
    
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 10), dpi=100)
    ax2.set_title('Loss')
    ax2.plot(range(len(loss_metrics['train_loss'])), loss_metrics['train_loss'], color='#4E79A7', linewidth=2, label='train loss')
    ax2.plot(range(len(loss_metrics['valid_loss'])), loss_metrics['valid_loss'], color='#A0CBE8', linewidth=2, label='validation loss')
    ax2.axvline(x=float(min_valid_loss_epoch), color='#F28E2B', linewidth=1, label='min validation epoch')
    ax2.legend(fontsize=12)
    ax2.set_xlabel('epochs', fontsize=15)
    ax2.set_ylabel('loss', fontsize=15)
    ax2.set_xticks(np.append(ax2.get_xticks(), float(min_valid_loss_epoch)))
    ax2.set_xlim(left=-float(len(loss_metrics['valid_loss'])/20), right=len(loss_metrics['valid_loss'])+float(len(loss_metrics['valid_loss'])/20))
    ax2.set_ylim(bottom=0)
    ax2.legend(loc='best')
    plt.tight_layout()
    fig2.savefig(os.path.join(conf['Testing']['valid_result_dir'], conf['Model']['name']+'_mark.png'))
    plt.close(fig2)


def draw_csv_loss(conf):
    print("="*20)
    print("Draw CSV Loss: ")
    pre_losses = list()
    pre_val_losses = list()
    pre_min_loss = 100
    pre_min_loss_epoch = None

    pre_min_val_loss = 100
    pre_min_val_loss_epoch = None

    csv_path = conf['Training']['loss_csv_path']
    with open(csv_path) as txt:
        lines = txt.readlines()
        for line in lines[1:]:
            epoch = line.split(',')[0]
            loss = float(line.split(',')[1])
            val_loss = float(line.split(",")[2])
            pre_losses.append(loss)
            pre_val_losses.append(val_loss)
            if loss < pre_min_loss:
                pre_min_loss = loss
                pre_min_loss_epoch = epoch
            if val_loss < pre_min_val_loss:
                pre_min_val_loss = val_loss
                pre_min_val_loss_epoch = epoch
    print(f'Best pretrained training loss: {pre_min_loss:.3f} / at epoch {pre_min_loss_epoch}')
    print(f'Best pretrained validate loss: {pre_min_val_loss:.3f} / at epoch {pre_min_val_loss_epoch}')

    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 10), dpi=100)
    ax1.plot(np.arange(len(pre_losses)), pre_losses, color='#4E79A7', linewidth=2, label='training loss')
    ax1.plot(np.arange(len(pre_val_losses)), pre_val_losses, color='#A0CBE8', linewidth=2, label='validation loss')
    ax1.legend(fontsize=12)
    ax1.set_xlabel('epochs', fontsize=15)
    ax1.set_ylabel('loss', fontsize=15)
    ax1.set_ylim(bottom=0)
    ax1.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(conf['Testing']['valid_result_dir'], conf['Model']['name']+'.png'))
    plt.close(fig1)
    
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 10), dpi=100)
    ax2.set_title('Loss')
    ax2.plot(np.arange(len(pre_losses)), pre_losses, color='#4E79A7', linewidth=2, label='training loss')
    ax2.plot(np.arange(len(pre_val_losses)), pre_val_losses, color='#A0CBE8', linewidth=2, label='validation loss')
    ax2.axvline(x=float(pre_min_val_loss_epoch), color='#F28E2B', linewidth=1, label='min validation epoch')
    ax2.legend(fontsize=12)
    ax2.set_xlabel('epochs', fontsize=15)
    ax2.set_ylabel('loss', fontsize=15)
    ax2.set_xticks(np.append(ax2.get_xticks(), float(pre_min_val_loss_epoch)))
    ax2.set_xlim(left=-float(len(pre_losses)/20), right=len(pre_losses)+float(len(pre_losses)/20))
    ax2.set_ylim(bottom=0)
    ax2.legend(loc='best')
    plt.tight_layout()
    fig2.savefig(os.path.join(conf['Testing']['valid_result_dir'], conf['Model']['name']+'_mark.png'))
    plt.close(fig2)
