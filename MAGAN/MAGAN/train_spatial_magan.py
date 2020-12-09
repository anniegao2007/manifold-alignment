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
    """Return the spatial and scRNAseq data."""
    spatial_expmat = np.load('/home/anniegao/spatial_magan/data/spatial_pca_with_coords.npz')['arr_0']
    spatial_expmat[:,100:] *= 5
    rna_expmat = np.load('/home/anniegao/spatial_magan/data/rna_pca_sampled.npz')['arr_0']
    spatial_pca_components = np.load('/home/anniegao/spatial_magan/data/spatial_pca_100components.npz')['arr_0']
    rna_pca_components = np.load('/home/anniegao/spatial_magan/data/rna_pca_100components.npz')['arr_0']
    spatial_cluster_labels = np.load('/home/anniegao/spatial_magan/data/spatial_cluster_3_labels_phate.npz')['arr_0']
    rna_cluster_labels = np.load('/home/anniegao/spatial_magan/data/rna_cluster_5_labels_sampled.npz')['arr_0']
    return spatial_expmat, rna_expmat, spatial_pca_components, rna_pca_components, spatial_cluster_labels, rna_cluster_labels

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
spatial_dataset, rna_dataset, spatial_pca_components, rna_pca_components, spatial_cluster_labels, rna_cluster_labels = get_data()
print("Spatial data shape: {} scRNAseq data shape: {}".format(spatial_dataset.shape, rna_dataset.shape))
print("Spatial PCs shape: {} scRNAseq PCs shape: {}".format(spatial_pca_components.shape, rna_pca_components.shape))
print('Spatial cluster labels len: {}'.format(spatial_cluster_labels.shape))
print('scRNAseq cluster labels len: {}'.format(rna_cluster_labels.shape))

# Prepare the loaders
load_spatial = Loader(spatial_dataset, labels=spatial_cluster_labels, shuffle=True)
load_rna = Loader(rna_dataset, labels=rna_cluster_labels, shuffle=True)
batch_size = 100

# Build the tf graph
magan = MAGAN(dim_b1=spatial_dataset.shape[1], dim_b2=rna_dataset.shape[1], correspondence_loss=correspondence_loss, xb1_pcs=spatial_pca_components, xb2_pcs=rna_pca_components)

# Data save directory
save_dir = './model25'

# Train
for i in range(1, 10000):
    if i % 100 == 0: print("Iter {} ({})".format(i, now()))
    spatial_, spatial_labels_ = load_spatial.next_batch(batch_size)
    rna_, rna_labels_ = load_rna.next_batch(batch_size)

    magan.train(spatial_, rna_)

    # Evaluate the loss and plot
    if i % 500 == 0:
        spatial_, spatial_labels_ = load_spatial.next_batch(10 * batch_size)
        rna_, rna_labels_ = load_rna.next_batch(10 * batch_size)

        lstring = magan.get_loss(spatial_, rna_)
        print("{} {}".format(magan.get_loss_names(), lstring))


        spatial = magan.get_layer(spatial_, rna_, 'xb1')
        rna = magan.get_layer(spatial_, rna_, 'xb2')
        spatial_gen = magan.get_layer(spatial_, rna_, 'Gb1')
        rna_gen = magan.get_layer(spatial_, rna_, 'Gb2')

        fig.clf()
        axes = fig.subplots(2, 2, sharex=True, sharey=True)
        axes[0, 0].set_title('Original')
        axes[0, 1].set_title('Generated')
        axes[0, 0].scatter(0, 0, s=45, c='b', label='Batch 1'); axes[0, 0].scatter(0,0, s=100, c='w'); # axes[0, 0].legend(handletextpad=.1, borderpad=.5, loc='center left', bbox_to_anchor=[.02, .5]);
        axes[0, 1].scatter(0, 0, s=45, c='r', label='Batch 2'); axes[0, 1].scatter(0,0, s=100, c='w'); # axes[0, 1].legend(handletextpad=.1, borderpad=.5, loc='center left', bbox_to_anchor=[.02, .5]);
        axes[1, 0].scatter(0, 0, s=45, c='r', label='Batch 2'); axes[1, 0].scatter(0,0, s=100, c='w'); # axes[1, 0].legend(handletextpad=.1, borderpad=.5, loc='center left', bbox_to_anchor=[.02, .5]);
        axes[1, 1].scatter(0, 0, s=45, c='b', label='Batch 1'); axes[1, 1].scatter(0,0, s=100, c='w'); # axes[1, 1].legend(handletextpad=.1, borderpad=.5, loc='center left', bbox_to_anchor=[.02, .5]);

        axes[0, 0].scatter(spatial_dataset[:,0], spatial_dataset[:,1], s=45, alpha=.5, cmap=matplotlib.cm.jet, c=spatial_cluster_labels, marker='.')
        axes[0, 1].scatter(rna_gen[:,0], rna_gen[:,1], s=45, alpha=.5, cmap=matplotlib.cm.jet, c=spatial_labels_, marker='.')

        axes[1, 0].scatter(rna_dataset[:,0], rna_dataset[:,1], s=45, alpha=.5, cmap=matplotlib.cm.jet, c=rna_cluster_labels, marker='.')
        axes[1, 1].scatter(spatial_gen[:,0], spatial_gen[:,1], s=45, alpha=.5, cmap=matplotlib.cm.jet, c=rna_labels_, marker='.')

        fig.canvas.draw()
        plt.savefig('{}/plots/plot_{}'.format(save_dir, i))

magan.save(folder=save_dir)