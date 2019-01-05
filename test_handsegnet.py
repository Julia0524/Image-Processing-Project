#
#  Test for HandSegNet
#

from __future__ import print_function, unicode_literals

import tensorflow as tf
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from nets.network import PoseEstimationNetwork
from utils.general import *

# snapshots file path
PATH_TO_HANDSEGNET_SNAPSHOTS = './snapshots_handsegnet/'  

if __name__ == '__main__':
    # images to be shown
    image_list = list()
    image_list.append('./data/1.png')
    image_list.append('./data/2.png')
    image_list.append('./data/3.png')
    image_list.append('./data/4.png')
    image_list.append('./data/5.png')

    # network input
    image_tf = tf.placeholder(tf.float32, shape=(1, 240, 320, 3))
    hand_side_tf = tf.constant([[1.0, 0.0]])  # left hand (true for all samples provided)
    evaluation = tf.placeholder_with_default(True, shape=())

    # build network
    net = PoseEstimationNetwork()
    hand_scoremap_tf = net.HandSegNet(image_tf)
    hand_scoremap_tf = hand_scoremap_tf[-1]

    # Start TF
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # initialize network
    # net.init(sess)
    # retrained version: HandSegNet
    last_cpt = tf.train.latest_checkpoint(PATH_TO_HANDSEGNET_SNAPSHOTS)
    assert last_cpt is not None, "Could not locate snapshot to load. Did you already train the network and set the path accordingly?"
    load_weights_from_snapshot(sess, last_cpt, discard_list=['Adam', 'global_step', 'beta'])

    # Feed image list through network
    for img_name in image_list:
        image_raw = scipy.misc.imread(img_name)
        image_raw = scipy.misc.imresize(image_raw, (240, 320))
        image_v = np.expand_dims((image_raw.astype('float') / 255.0) - 0.5, 0)

        hand_scoremap_v= sess.run([hand_scoremap_tf],feed_dict={image_tf: image_v})

        hand_scoremap_v = np.squeeze(hand_scoremap_v)

        # visualize
        fig = plt.figure(1)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.imshow(image_raw)
        ax2.imshow(np.argmax(hand_scoremap_v, 2))

        plt.show()
