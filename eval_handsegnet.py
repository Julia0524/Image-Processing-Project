#
#  Evaluation for HandSegNet and Crop
#

from __future__ import print_function, unicode_literals
import tensorflow as tf
import numpy as np

from data.BinaryDbReader import *
from nets.network import PoseEstimationNetwork
from utils.general import detect_keypoints, trafo_coords, EvalUtil, load_weights_from_snapshot

PATH_TO_HANDSEGNET_SNAPSHOTS = './snapshots_handsegnet/'  

# get dataset
dataset = BinaryDbReader(mode='evaluation', shuffle=False, use_wrist_coord=True, scale_to_size=True)

# build network graph
data = dataset.get()

# build network
net = PoseEstimationNetwork()

# scale input to common size for evaluation
image_scaled = tf.image.resize_images(data['image'], (240, 320))
s = data['image'].get_shape().as_list()
scale = (240.0/s[1], 320.0/s[2])

# feed trough network
hand_scoremap, _, scale_crop, center = net.HandSegCrop(image_scaled)

# Start TF
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tf.train.start_queue_runners(sess=sess)

# initialize network weights
last_cpt = tf.train.latest_checkpoint(PATH_TO_HANDSEGNET_SNAPSHOTS)
assert last_cpt is not None, "Could not locate snapshot to load. Did you already train the network and set the path accordingly?"
load_weights_from_snapshot(sess, last_cpt, discard_list=['Adam', 'global_step', 'beta'])

util = EvalUtil()
# iterate dataset
for i in range(dataset.num_samples):
    # get prediction
    hand_scoremap_v,\
    scale_crop_v, center_v = sess.run([hand_scoremap, scale_crop, center])

    hand_scoremap_v = np.squeeze(hand_scoremap_v)

    if (i % 100) == 0:
        print('%d / %d images done: %.3f percent' % (i, dataset.num_samples, i*100.0/dataset.num_samples))

# Output results
mean, median, auc, _, _ = util.get_measures(0.0, 30.0, 20)
print('Evaluation results:')
print('Average mean EPE: %.3f pixels' % mean)
print('Average median EPE: %.3f pixels' % median)
print('Area under curve: %.3f' % auc)