#
#  Evaluation for full pipeline
#

from __future__ import print_function, unicode_literals
import tensorflow as tf
import numpy as np

from data.BinaryDbReader import *
from data.BinaryDbReaderSTB import *
from nets.network import PoseEstimationNetwork
from utils.general import EvalUtil, get_stb_ref_curves, calc_auc

# get dataset
# dataset = BinaryDbReader(mode='evaluation', shuffle=False, use_wrist_coord=False)
dataset = BinaryDbReaderSTB(mode='evaluation', shuffle=False, use_wrist_coord=False)

# build network graph
data = dataset.get()
image_scaled = tf.image.resize_images(data['image'], (240, 320))

# build network
net = PoseEstimationNetwork()

# feed through network
evaluation = tf.placeholder_with_default(True, shape=())
_, _, _, _, _, coord3d_pred = net.Pose3DNet(image_scaled, data['hand_side'], evaluation)
coord3d_gt = data['keypoint_xyz21']

# Start TF
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tf.train.start_queue_runners(sess=sess)

# initialize network with weights used in the paper
net.init(sess, weight_files=['./weights/handsegnet-rhd.pickle',
                             './weights/posenet3d-rhd-stb.pickle'])

util = EvalUtil()
# iterate dataset
for i in range(dataset.num_samples):
    # get prediction
    keypoint_xyz21, keypoint_vis21, keypoint_scale, coord3d_pred_v = sess.run([data['keypoint_xyz21'], data['keypoint_vis21'], data['keypoint_scale'], coord3d_pred])

    keypoint_xyz21 = np.squeeze(keypoint_xyz21)
    keypoint_vis21 = np.squeeze(keypoint_vis21)
    coord3d_pred_v = np.squeeze(coord3d_pred_v)
    keypoint_scale = np.squeeze(keypoint_scale)

    # rescale to meters
    coord3d_pred_v *= keypoint_scale

    # center gt
    keypoint_xyz21 -= keypoint_xyz21[0, :]

    util.feed(keypoint_xyz21, keypoint_vis21, coord3d_pred_v)

    if (i % 100) == 0:
        print('%d / %d images done: %.3f percent' % (i, dataset.num_samples, i*100.0/dataset.num_samples))

# Output results
mean, median, auc, pck_curve_all, threshs = util.get_measures(0.0, 0.050, 20)  # rainier: Should lead to 0.764 / 9.405 / 12.210
print('Evaluation results')
print('Average mean EPE: %.3f mm' % (mean*1000))
print('Average median EPE: %.3f mm' % (median*1000))
print('Area under curve between 0mm - 50mm: %.3f' % auc)

# only use subset that lies in 20mm .. 50mm
pck_curve_all, threshs = pck_curve_all[8:], threshs[8:]*1000.0
auc_subset = calc_auc(threshs, pck_curve_all)
print('Area under curve between 20mm - 50mm: %.3f' % auc_subset)

# Show Figure 9 from the paper
if type(dataset) == BinaryDbReaderSTB:

    import matplotlib.pyplot as plt
    curve_list = get_stb_ref_curves()
    curve_list.append((threshs, pck_curve_all, 'Ours (AUC=%.3f)' % auc_subset))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for t, v, name in curve_list:
        ax.plot(t, v, label=name)
    ax.set_xlabel('threshold in mm')
    ax.set_ylabel('PCK')
    plt.legend(loc='lower right')
    plt.show()
