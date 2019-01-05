# 
# PoseNet: train to get score maps of each keypoint
#

from __future__ import print_function, unicode_literals

import tensorflow as tf
import os
import sys

from nets.network import PoseEstimationNetwork
from data.BinaryDbReader import BinaryDbReader
from utils.general import LearningRateScheduler, load_weights_from_snapshot

# training parameters
train_para = {'lr': [1e-4, 1e-5, 1e-6],
              'lr_iter': [10000, 20000],
              'max_iter': 30000,
              'show_loss_freq': 1000,
              'snapshot_freq': 5000,
              'snapshot_dir': 'snapshots_posenet'}

# get dataset
dataset = BinaryDbReader(mode='training',
                         batch_size=8, shuffle=True, use_wrist_coord=False,
                         hand_crop=True, coord_uv_noise=True, crop_center_noise=True)

# build network graph
data = dataset.get()

# build network
evaluation = tf.placeholder_with_default(True, shape=())
net = PoseEstimationNetwork()
keypoints_scoremap = net.PoseNet(data['image_crop'], train=True)
s = data['scoremap'].get_shape().as_list()
keypoints_scoremap = [tf.image.resize_images(x, (s[1], s[2])) for x in keypoints_scoremap]

# Start TF
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tf.train.start_queue_runners(sess=sess)

# Loss
loss = 0.0
s = data['scoremap'].get_shape().as_list()
vis = tf.cast(tf.reshape(data['keypoint_vis21'], [s[0], s[3]]), tf.float32)
for i, pred_item in enumerate(keypoints_scoremap):
    loss += tf.reduce_sum(vis * tf.sqrt(tf.reduce_mean(tf.square(pred_item - data['scoremap']), [1, 2]))) / (tf.reduce_sum(vis) + 0.001)

# Solver
global_step = tf.Variable(0, trainable=False, name="global_step")
lr_scheduler = LearningRateScheduler(values=train_para['lr'], steps=train_para['lr_iter'])
lr = lr_scheduler.get_lr(global_step)
opt = tf.train.AdamOptimizer(lr)
train_op = opt.minimize(loss)

# init weights
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=1, keep_checkpoint_every_n_hours=1.0)

# use weights from previous snapshots to train the network
PATH_TO_POSENET_SNAPSHOTS = './snapshots_posenet/'  
last_cpt = tf.train.latest_checkpoint(PATH_TO_POSENET_SNAPSHOTS)
assert last_cpt is not None, "Could not locate snapshot to load. Did you already train the network and set the path accordingly?"
load_weights_from_snapshot(sess, last_cpt, discard_list=['Adam', 'global_step', 'beta'])

# use initial weights to train the network
'''rename_dict = {'CPM/PoseNet': 'PoseNet2D',
               '_CPM': ''}
load_weights_from_snapshot(sess, './weights/cpm-model-mpii', ['PersonNet', 'PoseNet/Mconv', 'conv5_2_CPM'], rename_dict)
'''

# snapshot dir
if not os.path.exists(train_para['snapshot_dir']):
    os.mkdir(train_para['snapshot_dir'])
    print('Created snapshot dir:', train_para['snapshot_dir'])

# Training loop
print('Starting to train ...')
for i in range(train_para['max_iter']):
    _, loss_v = sess.run([train_op, loss])

    if (i % train_para['show_loss_freq']) == 0:
        print('Iteration %d\t Loss %.1e' % (i, loss_v))
        sys.stdout.flush()

    if (i % train_para['snapshot_freq']) == 0:
        saver.save(sess, "%s/model" % train_para['snapshot_dir'], global_step=i)
        print('Saved a snapshot.')
        sys.stdout.flush()


print('Training finished. Saving final snapshot.')
saver.save(sess, "%s/model" % train_para['snapshot_dir'], global_step=train_para['max_iter'])
