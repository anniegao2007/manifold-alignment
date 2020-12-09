import sys
import os
sys.path.append(os.getcwd())
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_probability as tfp
import matplotlib
from utils import now
from model import MAGAN
from loader import Loader
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.cm
plt.ion(); fig = plt.figure()


def get_data():
    """Return the sci-car rna and atac data."""
    rna_pca = np.load('/home/anniegao/spatial_magan/data/scicar_rna_pca.npz')['arr_0']
    atac_pca = np.load('/home/anniegao/spatial_magan/data/scicar_atac_pca_sampled.npz')['arr_0']
    rna_pca_components = np.load('/home/anniegao/spatial_magan/data/scicar_rna_pca_100components.npz')['arr_0']
    atac_pca_components = np.load('/home/anniegao/spatial_magan/data/scicar_atac_pca_100components.npz')['arr_0']
    rna_cluster_labels = np.load('/home/anniegao/spatial_magan/data/scicar_rna_cluster_4_labels_phate.npz')['arr_0']
    atac_cluster_labels = np.load('/home/anniegao/spatial_magan/data/scicar_atac_cluster_4_labels_sampled.npz')['arr_0']
    return rna_pca, atac_pca, rna_pca_components, atac_pca_components, rna_cluster_labels, atac_cluster_labels

def correspondence_loss(b1, b2, pcs):
    """
    The correspondence loss.

    :param b1: a tensor representing the object in the graph of the current minibatch from domain one
    :param b2: a tensor representing the object in the graph of the current minibatch from domain two
    :returns a scalar tensor of the correspondence loss
    """
    reconstructed_b1 = b1 @ pcs
    reconstructed_b2 = b2 @ pcs
    corr = tfp.stats.correlation(reconstructed_b1, reconstructed_b2, sample_axis=1, event_axis=None)
    loss = tf.reduce_sum(1 - tf.math.abs(corr))
    return loss

# Load the data
rna_dataset, atac_dataset, rna_pca_components, atac_pca_components, rna_cluster_labels, atac_cluster_labels = get_data()
print("rna data shape: {} atac data shape: {}".format(rna_dataset.shape, atac_dataset.shape))
print("rna PCs shape: {} atac PCs shape: {}".format(rna_pca_components.shape, atac_pca_components.shape))
print('rna cluster labels len: {}'.format(rna_cluster_labels.shape))
print('atac cluster labels len: {}'.format(atac_cluster_labels.shape))

# Prepare the loaders
load_rna = Loader(rna_dataset, labels=rna_cluster_labels, shuffle=True)
load_atac = Loader(atac_dataset, labels=atac_cluster_labels, shuffle=True)
batch_size = 100

# Build the tf graph
magan = MAGAN(dim_b1=rna_dataset.shape[1], dim_b2=atac_dataset.shape[1], correspondence_loss=correspondence_loss, xb1_pcs=rna_pca_components, xb2_pcs=atac_pca_components)

# Data save directory
save_dir = './scicar_models/model3'

# Train
for i in range(1, 10000):
    if i % 100 == 0: print("Iter {} ({})".format(i, now()))
    rna_, rna_labels_ = load_rna.next_batch(batch_size)
    atac_, atac_labels_ = load_atac.next_batch(batch_size)

    magan.train(rna_, atac_)

    # Evaluate the loss and plot
    if i % 500 == 0:
        rna_, rna_labels_ = load_rna.next_batch(10 * batch_size)
        atac_, atac_labels_ = load_atac.next_batch(10 * batch_size)

        lstring = magan.get_loss(rna_, atac_)
        print("{} {}".format(magan.get_loss_names(), lstring))


        rna = magan.get_layer(rna_, atac_, 'xb1')
        atac = magan.get_layer(rna_, atac_, 'xb2')
        rna_gen = magan.get_layer(rna_, atac_, 'Gb1')
        atac_gen = magan.get_layer(rna_, atac_, 'Gb2')

        fig.clf()
        axes = fig.subplots(2, 2, sharex=True, sharey=True)
        axes[0, 0].set_title('Original')
        axes[0, 1].set_title('Generated')
        axes[0, 0].scatter(0, 0, s=45, c='b', label='Batch 1'); axes[0, 0].scatter(0,0, s=100, c='w'); # axes[0, 0].legend(handletextpad=.1, borderpad=.5, loc='center left', bbox_to_anchor=[.02, .5]);
        axes[0, 1].scatter(0, 0, s=45, c='r', label='Batch 2'); axes[0, 1].scatter(0,0, s=100, c='w'); # axes[0, 1].legend(handletextpad=.1, borderpad=.5, loc='center left', bbox_to_anchor=[.02, .5]);
        axes[1, 0].scatter(0, 0, s=45, c='r', label='Batch 2'); axes[1, 0].scatter(0,0, s=100, c='w'); # axes[1, 0].legend(handletextpad=.1, borderpad=.5, loc='center left', bbox_to_anchor=[.02, .5]);
        axes[1, 1].scatter(0, 0, s=45, c='b', label='Batch 1'); axes[1, 1].scatter(0,0, s=100, c='w'); # axes[1, 1].legend(handletextpad=.1, borderpad=.5, loc='center left', bbox_to_anchor=[.02, .5]);

        axes[0, 0].scatter(rna_dataset[:,0], rna_dataset[:,1], s=45, alpha=.5, cmap=matplotlib.cm.jet, c=rna_cluster_labels, marker='.')
        axes[0, 1].scatter(atac_gen[:,0], atac_gen[:,1], s=45, alpha=.5, cmap=matplotlib.cm.jet, c=rna_labels_, marker='.')

        axes[1, 0].scatter(atac_dataset[:,0], atac_dataset[:,1], s=45, alpha=.5, cmap=matplotlib.cm.jet, c=atac_cluster_labels, marker='.')
        axes[1, 1].scatter(rna_gen[:,0], rna_gen[:,1], s=45, alpha=.5, cmap=matplotlib.cm.jet, c=atac_labels_, marker='.')

        fig.canvas.draw()
        plt.savefig('{}/plots/plot_{}'.format(save_dir, i))

    # if i % 5000 == 0:
    #     magan.save(folder=save_dir)

magan.save(folder=save_dir)