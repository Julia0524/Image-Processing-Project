# PoseEstimation Network
This is our course project of Image Processing: Pose Estimation from 2D RGB Images.
PoseEstimation Network is a convolutional neural network estimating 3D hand pose from a single 2D RGB image. 

## Environment
- Ubuntu 16.04.2 (xenial)
- Tensorflow 1.3.0 GPU build with CUDA 8.0.44 and CUDNN 5.1
- Python 3.5.2
- tensorflow==1.3.0
- numpy==1.13.0
- scipy==0.18.1
- matplotlib==1.5.3

## Dataset
These are the datasets used in training, testing and evaluation.

### Rendered Hand Pose Dataset (RHD)
- Download the dataset [RHD dataset v. 1.1](https://lmb.informatik.uni-freiburg.de/resources/datasets/RenderedHandposeDataset.en.html)
- Set the variable 'path_to_db' to where the dataset is located on your machine
- Optionally modify 'set' variable to training or evaluation
- Run

		python create_binary_db.py
- This will create a binary file in *./data/bin* according to how 'set' was configured

### Stereo Tracking Benchmark Dataset (STB)
- This dataset is needed for the training and evalution for Pose3DNet
- You can get the dataset presented in Zhang et al., ‘3d Hand Pose Tracking and Estimation Using Stereo Matching’, 2016
- After unzipping the dataset run

		cd ./data/stb/
		matlab -nodesktop -nosplash -r "create_db"
- This will create the binary file *./data/stb/stb_evaluation.bin*


## Training
- Make sure you get the proper environment and dataset
- Start training of HandSegNet with my_training_handsegnet.py
- Start training of PoseNet with my_training_posenet.py
- Start training of PoseNet with my_training_pose3dnet.py

## Testing
There are three test for different parts of the network, which will output responding images:
- test_handsegnet.py: Test HandSegNet on hand segmentation
- test_crop.py: Test HandSegNet on hand segmentation
- test_posenet.py: Test PoseNet on 2D keypoint detection

## Evaluation
There are four evaluation for different parts of the network:
- eval_handsegnet.py: Evaluates HandSegNet on hand segmentation
- eval_posenet.py: Evaluates PoseNet on 2D keypoint detection using ground truth annoation to create hand cropped images
- eval_handsegnet_posenet.py: Evaluates HandSegNet and PoseNet on 2D keypoint detection
- eval_full.py: Evaluates full pipeline on 3D keypoint localization from 2D RGB

## Results
The results of the network is in *./results*, including five folders:
- segnet: Results of HandSegNet on our own testing images
- crop: Results of HandSegNet and cropped images on our own testing images
- posenet: Results of PoseNet on our own testing images
- pose3dnet: Results of Pose3DNet on our own testing images
- RHD: Results of the same parts as above on Rendered Hand Pose Dataset(RHD)