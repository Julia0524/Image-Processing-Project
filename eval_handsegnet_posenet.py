#
#  Evaluation for HandSegNet and PoseNet
#

from __future__ import print_function, unicode_literals
import tensorflow as tf
import numpy as np

from data.BinaryDbReader import *
from nets.network import PoseEstimationNetwork
from utils.general import *

# flag that allows to load a retrained snapshot(original weights used in the paper are used otherwise)
USE_RETRAINED = False
PATH_TO_POSENET_SNAPSHOTS = './snapshots_posenet/'  # only used when USE_RETRAINED is true
PATH_TO_HANDSEGNET_SNAPSHOTS = './snapshots_handsegnet/'  # only used when USE_RETRAINED is true

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

hand_scoremap, image_crop, scale_crop, center = net.HandSegCrop(image_scaled)
    
# detect keypoints in 2D
s = image_crop.get_shape().as_list()
keypoints_scoremap = net.PoseNet(image_crop)
keypoints_scoremap = keypoints_scoremap[-1]
keypoints_scoremap = tf.image.resize_images(keypoints_scoremap, (s[1], s[2]))

# Start TF
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tf.train.start_queue_runners(sess=sess)

# initialize network weights
# retrained version: HandSegNet
last_cpt = tf.train.latest_checkpoint(PATH_TO_HANDSEGNET_SNAPSHOTS)
assert last_cpt is not None, "Could not locate snapshot to load. Did you already train the network and set the path accordingly?"
load_weights_from_snapshot(sess, last_cpt, discard_list=['Adam', 'global_step', 'beta'])

# retrained version: PoseNet
last_cpt = tf.train.latest_checkpoint(PATH_TO_POSENET_SNAPSHOTS)
assert last_cpt is not None, "Could not locate snapshot to load. Did you already train the network and set the path accordingly?"
load_weights_from_snapshot(sess, last_cpt, discard_list=['Adam', 'global_step', 'beta'])
    
# load weights used in the paper
# net.init(sess, weight_files=['./weights/posenet-rhd-stb.pickle'], exclude_var_list=['PosePrior', 'ViewpointNet'])

util = EvalUtil()
# iterate dataset
for i in range(dataset.num_samples):
    # get prediction
    keypoints_scoremap_v,\
    scale_crop_v, center_v, kp_uv21_gt, kp_vis = sess.run([keypoints_scoremap, scale_crop, center, data['keypoint_uv21'], data['keypoint_vis21']])

    keypoints_scoremap_v = np.squeeze(keypoints_scoremap_v)
    kp_uv21_gt = np.squeeze(kp_uv21_gt)
    kp_vis = np.squeeze(kp_vis)

    # detect keypoints
    coord_hw_pred_crop = detect_keypoints(np.squeeze(keypoints_scoremap_v))
    coord_hw_pred = trafo_coords(coord_hw_pred_crop, center_v, scale_crop_v, 256)
    coord_uv_pred = np.stack([coord_hw_pred[:, 1], coord_hw_pred[:, 0]], 1)

    # scale pred to image size of the dataset (to match with stored coordinates)
    coord_uv_pred[:, 1] /= scale[0]
    coord_uv_pred[:, 0] /= scale[1]

    # some datasets are already stored with downsampled resolution
    scale2orig_res = 1.0
    if hasattr(dataset, 'resolution'):
        scale2orig_res = dataset.resolution

    util.feed(kp_uv21_gt/scale2orig_res, kp_vis, coord_uv_pred/scale2orig_res)

    if (i % 100) == 0:
        print('%d / %d images done: %.3f percent' % (i, dataset.num_samples, i*100.0/dataset.num_samples))

# Output results
mean, median, auc, _, _ = util.get_measures(0.0, 30.0, 20)
print('Evaluation results:')
print('Average mean EPE: %.3f pixels' % mean)
print('Average median EPE: %.3f pixels' % median)
print('Area under curve: %.3f' % auc)