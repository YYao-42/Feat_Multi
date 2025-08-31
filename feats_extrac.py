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
ap.add_argument("-vn", "--videoname", required=True,
	help="name of the video")
ap.add_argument("-fn", "--featurename", required=True,
	help="name of the feature to be extracted")
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
video_folder = rf"C:\Users\Gebruiker\Documents\Experiments\downsamp_video"
video_name = args["videoname"]
video_id = video_name.split('_')[0]
video_path = os.path.join(video_folder, video_name)
video_path_output = os.path.join(video_folder, video_name.split('.')[0] + '_output.mp4')
feature_name = args["featurename"]

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
if feature_name == 'ObjFlow':
	feat_1st = np.zeros((1, args["nbins"]+4+5)) # histogram + box info (xc, yc, w, h) + motion info (avg, u, d, l, r)
elif feature_name == 'ObjTempCtr':
	feat_1st = np.zeros((1, 3))
elif feature_name == 'ObjRMSCtr':
	bboxes_list, masks_list, scores_list, _ = feutils.object_seg_mmdetection(frame_prev, net, args)
	feat_1st, _ = feutils.obj_rms_contrast(frame_prev, bboxes_list, scores_list, masks_list, oneobject=True, ratio=2, ifmask=True)
elif feature_name == 'RMSCtr':
	feat_1st, _ = feutils.cal_rms_contrast(frame_prev)
else:
	raise ValueError('Feature name not recognized!')

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
	if 'Obj' in feature_name:
		# Detect objects 
		bboxes_list, masks_list, scores_list, elap_OS = feutils.object_seg_mmdetection(frame, net, args)
	else:
		elap_OS = 0
	# Extract features
	if feature_name == 'ObjFlow':
		feature, frame_OF, elap = feutils.optical_flow_mask(frame, frame_prev, bboxes_list, scores_list, masks_list, oneobject=True, nb_bins=8)
		# check if the video writer is None
		if writer is None:
			# initialize our video writer
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter(video_path_output, fourcc, 30,
				(frame_OF.shape[1], frame_OF.shape[0]), True)
		# write the output frame to disk
		writer.write(frame_OF)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	elif feature_name == 'ObjTempCtr':
		feature, elap = feutils.obj_temp_contrast(frame, frame_prev, bboxes_list, scores_list, masks_list, oneobject=True, ifmask=True)
	elif feature_name == 'ObjRMSCtr':
		feature, elap = feutils.obj_rms_contrast(frame, bboxes_list, scores_list, masks_list, oneobject=True, ifmask=True)
	elif feature_name == 'RMSCtr':
		feature, elap = feutils.cal_rms_contrast(frame)
	else: 
		raise ValueError('Feature name not recognized!')
	# some information on processing single frame
	if total > 0:
		print("[INFO] single frame took {:.4f} seconds".format(elap+elap_OS))
		print("[INFO] estimated total time to finish: {:.4f}".format((elap+elap_OS) * total))
	feature_list.append(feature)
	frame_prev = frame

print("[INFO] saving features ...")
# create features folder if it doesn't exist
if not os.path.exists('features'):
    os.makedirs('features')
save_path = os.path.join('features', video_id + '.pkl')
# try to load existing feature dictionary if file exists and is not empty
if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
    try:
        with open(save_path, "rb") as f:
            feature_dict = pickle.load(f)
    except (EOFError, pickle.UnpicklingError):
        print(f"[WARN] Could not read {save_path}, starting with empty dictionary")
        feature_dict = {}
else:
    feature_dict = {}
# update with new features
feature_dict[feature_name] = feature_list
# save updated dictionary
with open(save_path, "wb") as f:
    pickle.dump(feature_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

# release the file pointers
print("[INFO] cleaning up ...")
if writer is not None:
	writer.release()
vs.release()