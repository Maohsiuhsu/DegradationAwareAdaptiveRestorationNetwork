import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
plt.switch_backend('agg')

def process_img(img_tensor):
    img = img_tensor.cpu().numpy()
    img = img.transpose(0, 2, 3, 1)
    img = img[0, :, :, 0]
    img = np.squeeze(img).astype(np.float32)
    img = (np.clip(img*255, 0, 255)).astype(np.uint8)
    return img

def plot_imgs(input_imgs_tensor, gt_imgs_tensor, pred_imgs_tensor, epoch, conf, name_base=''):
    print("plot validation images\nepoch: {epoch}".format(epoch=epoch))
    plt.subplot(1, 3, 1)
    input_img = process_img(input_imgs_tensor)
    plt.imshow(input_img, cmap='gray')
    plt.title("input")

    plt.subplot(1, 3, 2)
    gt_img = process_img(gt_imgs_tensor)
    plt.imshow(gt_img, cmap='gray')
    plt.title("gt")

    plt.subplot(1, 3, 3)
    pred_img = process_img(pred_imgs_tensor)
    plt.imshow(pred_img, cmap='gray')
    plt.title("pred")

    plt.tight_layout()
    if name_base == '':
        plt.savefig(os.path.join(conf["Testing"]["valid_result_dir"], "valid_{epoch}.jpg".format(epoch=epoch)))
    else:
        plt.savefig(os.path.join(conf["Testing"]["valid_result_dir"], "valid_{epoch}_{name_base}.jpg".format(epoch=epoch, name_base=name_base))) 
    plt.close()
     