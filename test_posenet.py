#
#  Test for PoseNet
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
PATH_TO_POSENET_SNAPSHOTS = './snapshots_posenet/'  

if __name__ == '__main__':
    # images to be shown
    image_list = list()
    image_list.append('./data/6.png')
    image_list.append('./data/2.png')
    image_list.append('./data/3.png')
    image_list.append('./data/4.png')
    image_list.append('./data/5.png')

    # network input
    image_tf = tf.placeholder(tf.float32, shape=(1, 240, 320, 3))

    # build network
    net = PoseEstimationNetwork()
    hand_scoremap_tf, image_crop_tf, scale_crop_tf, center_tf = net.HandSegCrop(image_tf)
    
    # detect keypoints in 2D
    s = image_crop_tf.get_shape().as_list()
    keypoints_scoremap_tf = net.PoseNet(image_crop_tf)
    keypoints_scoremap_tf = keypoints_scoremap_tf[-1]
    keypoints_scoremap_tf = tf.image.resize_images(keypoints_scoremap_tf, (s[1], s[2]))

    # Start TF
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # Load CheckPoint files 
    last_cpt = tf.train.latest_checkpoint(PATH_TO_HANDSEGNET_SNAPSHOTS)
    assert last_cpt is not None, "Could not locate snapshot to load. Did you already train the network and set the path accordingly?"
    load_weights_from_snapshot(sess, last_cpt, discard_list=['Adam', 'global_step', 'beta'])

    last_cpt = tf.train.latest_checkpoint(PATH_TO_POSENET_SNAPSHOTS)
    assert last_cpt is not None, "Could not locate snapshot to load. Did you already train the network and set the path accordingly?"
    load_weights_from_snapshot(sess, last_cpt, discard_list=['Adam', 'global_step', 'beta'])

    # OR load weights used in the paper
    # net.init(sess, weight_files=['./weights/handsegnet-rhd.pickle',
    #                             './weights/posenet-rhd-stb.pickle'], exclude_var_list=['PosePrior', 'ViewpointNet'])

    
    # Feed image list through network
    for img_name in image_list:
        image_raw = scipy.misc.imread(img_name)
        image_raw = scipy.misc.imresize(image_raw, (240, 320))
        image_v = np.expand_dims((image_raw.astype('float') / 255.0) - 0.5, 0)

        hand_scoremap_v, image_crop_v, scale_v, center_v,\
        keypoints_scoremap_v = sess.run([hand_scoremap_tf, image_crop_tf, scale_crop_tf, center_tf,keypoints_scoremap_tf],
                                                            feed_dict={image_tf: image_v})
        
        hand_scoremap_v = np.squeeze(hand_scoremap_v)
        image_crop_v = np.squeeze(image_crop_v)
        keypoints_scoremap_v = np.squeeze(keypoints_scoremap_v)

        # post processing
        image_crop_v = ((image_crop_v + 0.5) * 255).astype('uint8')
        coord_hw_crop = detect_keypoints(np.squeeze(keypoints_scoremap_v))
        coord_hw = trafo_coords(coord_hw_crop, center_v, scale_v, 256)

        # visualize
        fig = plt.figure(1)
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax1.imshow(image_raw)
        plot_hand(coord_hw, ax1)
        ax2.imshow(image_crop_v)
        plot_hand(coord_hw_crop, ax2)
        ax3.imshow(np.argmax(hand_scoremap_v, 2))
        plt.show()
