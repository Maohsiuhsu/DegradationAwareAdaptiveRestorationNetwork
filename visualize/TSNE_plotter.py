''' reference
https://learnopencv.com/t-sne-for-feature-visualization/
https://towardsdatascience.com/t-sne-machine-learning-algorithm-a-great-tool-for-dimensionality-reduction-in-python-ec01552f1a1e
'''
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

plt.switch_backend('agg')

def label_processor(labels):
    label_arr = np.argmax(labels, axis=1)
    return label_arr

class TSNE_plotter():
    def __init__(self, class_num = 3, save_path = ""):
        self.class_num = class_num
        self.tsne = TSNE(n_components=2, init='pca', learning_rate='auto')
        self.save_path = save_path

    def plot_tnse(self, features, labels, epoch=0, sample_num=1000):
        anchor_ts = self.tsne.fit_transform(features)
        self._plot_embedding(data=anchor_ts, label=labels,  title="t-SNE of humidity degradation embedding", epoch=epoch, sample_num=sample_num)

    def _plot_embedding(self, data, label, title, epoch, sample_num=1000):
        fig, ax = plt.subplots()
        
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)
        color = ['tab:blue', 'tab:orange', 'tab:green']
        plot_label = ['light-wet', 'medium-wet', 'heavy-wet']
        for class_id in range(self.class_num):
            scatter_data = data[label == class_id]
            random_indices = np.random.choice([i for i in range(len(scatter_data))], size=sample_num, replace=False)
            random_scatter_data = scatter_data[random_indices]
            scatter_x = random_scatter_data[:, 0]
            scatter_y = random_scatter_data[:, 1]
            ax.scatter(scatter_x,scatter_y, s=30.0, color=color[class_id], label=plot_label[class_id], edgecolors='none')
        # for i in range(data.shape[0]):
        #     ax.scatter(data[i,0], data[i,1], color=color[label[i]], s=1, label=plot_label[label[i]], edgecolors='none')
        #     ax.text(data[i, 0], data[i, 1], str(label[i]), color=plt.cm.Set1(label[i] / self.class_num),
        #             fontdict={ "weight" :  "bold" ,  "size" : 10})
        
        ax.legend()
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(self.save_path+"/tnse_{epoch}.jpg".format(epoch=epoch))
        plt.close(fig)
        
        print(self.save_path+"/tnse_{epoch}.jpg".format(epoch=epoch))