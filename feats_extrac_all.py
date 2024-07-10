'''
Similar to feats_extrac.py, but this script save the flow matrices and the box/mask information instead of averaging flow magnitudes over object pixels.

Author: yuanyuan.yao@kuleuven.be
'''

# import the necessary packages
import numpy as np
import argparse
import cv2
import os
import feutils
import pickle
from mmdet.utils import register_all_modules
from mmdet.apis import init_detector


def box_mask_single(frame, net, args, H, W, FULLMASK=True):
	boxes_list, masks_list, _, elap = feutils.object_seg_mmdetection(frame, net, args, FULLMASK=FULLMASK)
	if len(boxes_list) == 0:
		print('No object detected! Set the box info to NaN!')
		box_3D = np.full((1, 4, 1), np.nan)
		mask_3D = np.full((H, W, 1), False)
	else:
		if len(boxes_list) > 1:
			print('More than one object detected! Only the first one is kept!')
		box_3D = np.expand_dims(np.expand_dims(np.array(boxes_list[0]), axis=0), axis=-1)
		mask_3D = np.expand_dims(np.array(masks_list[0]), axis=-1)
	return box_3D, mask_3D, elap

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-o", "--output", required=True,
	help="path to output video")
ap.add_argument("-dl", "--detectlabel", type=int, default=0,
	help="class of objects to be detected (default: 0 -> person)")
ap.add_argument("-nb", "--nbins", type=int, default=8,
	help="number of bins of the histogram")
ap.add_argument("-c", "--confidence", type=float, default=0.3,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.1,
	help="minimum threshold for pixel-wise mask segmentation")
args = vars(ap.parse_args())


# Choose to use a config and initialize the detector
config_file = 'checkpoints\mask-rcnn_r50-caffe_fpn_ms-poly-3x_coco.py'
# Setup a checkpoint file to load
checkpoint_file = 'checkpoints\mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
# register all modules in mmdet into the registries
register_all_modules()
# build the model from a config file and a checkpoint file
net = init_detector(config_file, checkpoint_file, device='cuda:0')  # or device='cuda:0'


# initialize the video stream, pointer to output video file, and
# frame dimensions
video_path = args["input"]
video_name = video_path.split('/')[-1]
video_id = video_name.split('_')[0]
vs = cv2.VideoCapture(video_path)
writer = None
# try to determine the total number of frames in the video file
try:
	total = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
	print("[INFO] {} total frames in video".format(total))
# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

mag_list = []
ang_list = []
box_list = []
mask_list = []

# First frame
grabbed, frame_prev = vs.read()
H, W = frame_prev.shape[:2]
mag_list.append(np.zeros((H, W, 1)))
ang_list.append(np.zeros((H, W, 1)))
box_first, mask_first, _ = box_mask_single(frame_prev, net, args, H, W)
box_list.append(box_first)
mask_list.append(mask_first)

# loop over frames from the video file stream
while True:
	# read the next frame from the file
	grabbed, frame = vs.read()
	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break
	# Detect objects 
	box, mask, elap_OS = box_mask_single(frame, net, args, H, W)
	box_list.append(box)
	mask_list.append(mask)
	# Compute the optical flow
	magnitude_3D, angle_3D, frame_OF, elap_OF = feutils.optical_flow_FB(frame, frame_prev)
	mag_list.append(magnitude_3D.astype(np.float16))
	ang_list.append(angle_3D.astype(np.float16))
	frame_prev = frame
	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(frame_OF.shape[1], frame_OF.shape[0]), True)
		# some information on processing single frame
		if total > 0:
			print("[INFO] Object segmentation: single frame took {:.4f} seconds".format(elap_OS))
			print("[INFO] Optical flow: single frame took {:.4f} seconds".format(elap_OF))
			print("[INFO] estimated total time to finish: {:.4f}".format((elap_OS+elap_OF) * total))
	# write the output frame to disk
	writer.write(frame_OF)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
print("[INFO] saving features ...")
# if folder does not exist, create it
if not os.path.exists('features_all'):
	os.makedirs('features_all')
features = {'mag': mag_list, 'ang': ang_list, 'box': box_list, 'mask': mask_list}
for feature, feature_list in features.items():
    path = f'features_all/{video_id}_{feature}.npy'
    np.save(path, np.concatenate(tuple(feature_list), axis=-1))
# release the file pointers
print("[INFO] cleaning up ...")
writer.release()
vs.release()