# import the necessary packages
import numpy as np
import argparse
import cv2
import os
import feutils
import pickle
from mmdet.utils import register_all_modules
from mmdet.apis import init_detector

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
(W, H) = (None, None)
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

feature_list = []

# First frame
grabbed, frame_prev = vs.read()
hist_mtx = np.zeros((1, args["nbins"]))
box_mtx = np.zeros((1, 4))
mag_mtx = np.zeros((1, 5))
tc_mtx = np.zeros((1, 3))
feat_1st = np.zeros((1, args["nbins"]+4+5+3))
feature_list.append(feat_1st)

# loop over frames from the video file stream
while True:
	# read the next frame from the file
	grabbed, frame = vs.read()
	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break
	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		H, W = frame.shape[:2]
	# Detect objects 
	bboxes_list, masks_list, scores_list, elap_OS = feutils.object_seg_mmdetection(frame, net, args)
	# Compute the optical flow of the most confidenet detected object
	feature_flow, frame_OF, elap_OF = feutils.optical_flow_mask(frame, frame_prev, bboxes_list, scores_list, masks_list, oneobject=True, nb_bins=8)
	feature_tc, elap_TC = feutils.obj_temp_contrast(frame, frame_prev, bboxes_list, scores_list, masks_list, oneobject=True, ifmask=True)
	feature_boxes = np.concatenate((feature_flow, feature_tc), axis=1)
	feature_list.append(feature_boxes)
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
			print("[INFO] Temporal contrast: single frame took {:.4f} seconds".format(elap_TC))
			print("[INFO] estimated total time to finish: {:.4f}".format((elap_OS+elap_OF+elap_TC) * total))
	# write the output frame to disk
	writer.write(frame_OF)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
print("[INFO] saving features ...")
# if folder features does not exist, create it
if not os.path.exists('features'):
	os.makedirs('features')
save_path = 'features/' + video_id +'_mask.pkl'
open_file = open(save_path, "wb")
pickle.dump(feature_list, open_file)
open_file.close()
# release the file pointers
print("[INFO] cleaning up ...")
writer.release()
vs.release()