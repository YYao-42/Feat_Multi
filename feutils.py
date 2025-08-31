import numpy as np
import cv2 as cv
import time
import copy
import math
import torch
from mmdet.apis import inference_detector


def cal_rms_contrast(frame_gray, mask=None):
	# Calculate root mean square contrast
	start = time.time()
	# if mask is not a None object, then apply mask to the frame:
	if mask is not None:
		frame_gray = frame_gray[mask]
	# change type to float
	frame_gray = frame_gray.astype(float)
	rms = np.sqrt(np.mean((frame_gray - np.mean(frame_gray))**2))
	end = time.time()
	elap = end - start
	return rms, elap


def obj_rms_contrast(frame, boxes, confidences, masks, oneobject=True, ratio=2, ifmask=True):
	start = time.time()
	frame_grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	feature_boxes = []
	if len(boxes)==0:
		print('WARNING: No object detected! Adding NaN values to features.')
		feature_frame = np.full((1, 1), np.nan)
	else:
		if oneobject:
			idxs = [np.argmax(np.array(confidences))]
		elif len(boxes) > 3: # keep the top 3 objects with highest confidence
			idxs = np.argsort(np.array(confidences))[-3:]
		else:
			idxs = np.argsort(np.array(confidences))
		for i in idxs:
			xs, ys, xl, yl, mask = expand_box(boxes[i], masks[i], frame.shape[1], frame.shape[0], ratio=ratio)
			patch = frame_grey[ys:yl, xs:xl]
			try:
				if ifmask:
					rms, _ = cal_rms_contrast(patch, mask)
				else:
					rms, _ = cal_rms_contrast(patch)
			except:
				print('WARNING: empty patches!')
				break
			feature_boxes.append(rms)
		feature_frame = np.array(feature_boxes).reshape(-1, 1)
	end = time.time()
	elap = end - start
	return feature_frame, elap


def cal_tempcontrast(frame_gray, frame_prev_gray, mask=None):
	# if mask is not a None object, then apply mask to the frame:
	if mask is not None:
		frame_gray = frame_gray[mask]
		frame_prev_gray = frame_prev_gray[mask]
	# change type to float
	frame_prev_gray = frame_prev_gray.astype(float)
	frame_gray = frame_gray.astype(float)
	mutempcontr = np.mean(frame_gray-frame_prev_gray)
	abstempcontr = np.mean(np.abs(frame_gray-frame_prev_gray))
	muSqtempcontr = np.mean((frame_gray-frame_prev_gray)**2)
	tempcontr_vec = np.expand_dims(np.array([abstempcontr, muSqtempcontr, mutempcontr]), axis=0)
	return tempcontr_vec


# def get_tempcontrast(video_path):
# 	vs = cv.VideoCapture(video_path)
# 	# First frame
# 	grabbed, frame_prev = vs.read()
# 	tempcontr_mtx = np.zeros((1, 3))
# 	# loop over frames from the video file stream
# 	while True:
# 		# read the next frame from the file
# 		grabbed, frame = vs.read()
# 		# if the frame was not grabbed, then we have reached the end
# 		# of the stream
# 		if not grabbed:
# 			break
# 		# transform to grayscale
# 		frame_prev_gray = cv.cvtColor(frame_prev, cv.COLOR_BGR2GRAY)
# 		frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
# 		tempcontr_vec = cal_tempcontrast(frame_gray, frame_prev_gray)
# 		tempcontr_mtx = np.concatenate((tempcontr_mtx, tempcontr_vec), axis=0)
# 		frame_prev = frame
# 	vs.release()
# 	return tempcontr_mtx


def obj_temp_contrast(frame, frame_prev, boxes, confidences, masks, oneobject=True, ratio=2, ifmask=True):
	start = time.time()
	frame_grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	frame_prev_grey = cv.cvtColor(frame_prev, cv.COLOR_BGR2GRAY)
	feature_boxes = []
	if len(boxes)==0:
		print('WARNING: No object detected! Adding NaN values to features.')
		feature_frame = np.full((1, 3), np.nan)
	else:
		if oneobject:
			idxs = [np.argmax(np.array(confidences))]
		elif len(boxes) > 3: # keep the top 3 objects with highest confidence
			idxs = np.argsort(np.array(confidences))[-3:]
		else:
			idxs = np.argsort(np.array(confidences))
		for i in idxs:
			xs, ys, xl, yl, mask = expand_box(boxes[i], masks[i], frame.shape[1], frame.shape[0], ratio=ratio)
			patch = frame_grey[ys:yl, xs:xl]
			patch_prev = frame_prev_grey[ys:yl, xs:xl]
			try:
				if ifmask:
					tempcontr_vec = cal_tempcontrast(patch, patch_prev, mask)
				else:
					tempcontr_vec = cal_tempcontrast(patch, patch_prev)
			except:
				print('WARNING: empty patches!')
				break
			feature_boxes.append(tempcontr_vec)
		feature_frame = np.concatenate(tuple(feature_boxes), axis=0)
	end = time.time()
	elap = end - start
	return feature_frame, elap


def expand_box(box, mask, frameW, frameH, ratio=2):
	mask_frame = (np.zeros((frameH, frameW))).astype(bool)
	(x, y) = (box[0], box[1])
	(w, h) = (box[2], box[3])
	mask_frame[y:y+h, x:x+w] = mask
	center_x = x + w/2
	center_y = y + h/2
	w_new = int(math.sqrt(ratio)*w)
	h_new = int(math.sqrt(ratio)*h)
	start_x = max(0, math.floor(center_x-w_new/2))
	start_y = max(0, math.floor(center_y-h_new/2))
	end_x = min(frameW, math.ceil(center_x+w_new/2))
	end_y = min(frameH, math.ceil(center_y+h_new/2))
	mask_expand = mask_frame[start_y:end_y, start_x:end_x]
	return start_x, start_y, end_x, end_y, mask_expand
	

def HOOF(magnitude, angle, nb_bins, mask=None, fuzzy=False, normalize=False):
	'''
	Histogram of (fuzzy) oriented optical flow
	Inputs:
	magnitude: magnitude matrix of the optical flow
	angle: angle matrix of the optical flow
	nb_bins: number of bins
	mask: the mask obtained by mask r-cnn
	fuzzy: whether include fuzzy matrix https://ieeexplore.ieee.org/document/7971947/. Default is False.
	nomalize: whether normalize the histogram to be a pdf. Default is False such that we don't lose information (sum of the magnitude), 
	          which gives us more freedom on post-processing (like smoothing).
	Outputs:
	hist: normalized and weighted orientation histogram with size 1 x nb_bins
	Note: The normalized histogram does not sum to 1; instead, np.sum(hist)*2pi/nb_bins = 1
	'''
	if mask is not None:
		magnitude = magnitude[mask]
		angle = angle[mask]
	else:
		# Flatten mag/ang matrices
		magnitude = magnitude.flatten()
		angle = angle.flatten()
	# Deal with circular continuity
	for i in range(len(angle)):
		while (angle[i] < 0 or angle[i] > 2*np.pi):
			angle[i] = angle[i] - np.sign(angle[i])*2*np.pi
	# Normalized histogram weighted by magnitudes
	if fuzzy:
		x = np.linspace(0, 2*np.pi, nb_bins*2+1)
		bin_mid = x[[list(range(1, 2*nb_bins, 2))]]
		nb_bins_dense = nb_bins*5
		x = np.linspace(0, 2*np.pi, nb_bins_dense*2+1)
		bin_dense_mid = x[[list(range(1, 2*nb_bins_dense, 2))]]
		diff_mtx = np.minimum(np.abs(bin_mid-bin_dense_mid.T), 2*np.pi-np.abs(bin_mid-bin_dense_mid.T))
		sigma = 0.1 # May not be the best value
		coe_mtx = np.exp(-diff_mtx**2/2/sigma**2) # fuzzy matrix
		hist_dense, _ = np.histogram(angle, nb_bins_dense, range=(0, 2*np.pi), weights=magnitude, density=False)
		hist = np.expand_dims(hist_dense, axis=0)@coe_mtx
		hist = hist/np.sum(hist)/2/np.pi*nb_bins
	else:
		hist, _ = np.histogram(angle, nb_bins, range=(0, 2*np.pi), weights=magnitude, density=normalize)
		hist = np.expand_dims(hist, axis=0)
	return hist


def object_seg_mmdetection(frame, net, args, FULLMASK=False):
	'''
	Use pretrained Mask R-CNN network to perform object segmentation
	Inputs:
	frame: current frame
	net: mask r-cnn net for object segmentation (using mmdetection package)
	args: pre-defined parameters
	FULLMASK: whether return the full mask or the mask in the bounding box. Default is False.
	Outputs:
	boxes: boxes of the detected objects
	confidences: confidences of detected objects 
	masks: masks of the object
	elap: processing time
	'''
	start = time.time()
	result = inference_detector(net, frame)
	result_mask = torch.logical_and(result.pred_instances['labels'] == args["detectlabel"], result.pred_instances['scores'] > args["confidence"])
	bboxes = result.pred_instances['bboxes'][result_mask,:]
	masks = result.pred_instances['masks'][result_mask,:,:]
	scores = result.pred_instances['scores'][result_mask]
	bboxes = bboxes.cpu().numpy().astype(int)
	masks = masks.cpu().numpy()
	scores = scores.cpu().numpy()
	if FULLMASK:
		masks_list = [masks[i,:,:] for i in range(masks.shape[0])]
	else:
		masks_list = [masks[i,bboxes[i,1]:bboxes[i,3],bboxes[i,0]:bboxes[i,2]] for i in range(masks.shape[0])]
	scores_list = [scores[i] for i in range(scores.shape[0])]
	bboxes_H = bboxes[:,3] - bboxes[:,1]
	bboxes_W = bboxes[:,2] - bboxes[:,0]
	# replace the last two columns with the height and width of the bounding box
	bboxes[:,2] = bboxes_W
	bboxes[:,3] = bboxes_H
	bboxes_list = [bboxes[i,:] for i in range(bboxes.shape[0])]
	end = time.time()
	elap = end-start
	return bboxes_list, masks_list, scores_list, elap


def optical_flow_FB(frame, frame_prev):
	'''
	Inputs:
	frame: current frame
	frame_prev: previous frame
	Outputs:
	hist: orientation histogram
	box_info: NaN (Keep here just to have the same number of outputs as optical_flow_box)
	mag: average magnitude (all direction/left/right/up/down) 
	frame_OF: modified current frame 
	elap: processing time
	'''
	start = time.time()
	cv.imshow("input", frame)
	frame_grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	frame_prev_grey = cv.cvtColor(frame_prev, cv.COLOR_BGR2GRAY)
	flow = cv.calcOpticalFlowFarneback(frame_prev_grey, frame_grey, None, 0.5, 3, 15, 3, 5, 1.2, 0)
	flow_horizontal = flow[..., 0]
	flow_vertical = flow[..., 1]
	# Computes the magnitude and angle of the 2D vectors
	# DON'T TRUST cv.cartToPolar
	# magnitude, angle = cv.cartToPolar(flow_horizontal, flow_vertical, angleInDegrees=False)
	magnitude = np.absolute(flow_horizontal+1j*flow_vertical)
	angle = np.angle(flow_horizontal+1j*flow_vertical)
	magnitude_3D = np.expand_dims(magnitude, axis=-1)
	angle_3D = np.expand_dims(angle, axis=-1)
	hsv = np.zeros_like(frame)
	hsv[..., 0] = angle*180/np.pi/2
	hsv[..., 1] = 255
	hsv[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
	frame_OF = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
	cv.imshow("optical flow", frame_OF)
	end = time.time()
	elap = end - start
	return magnitude_3D, angle_3D, frame_OF, elap


def optical_flow_mask(frame, frame_prev, boxes, confidences, masks, detect_label='person', oneobject=True, nb_bins=8, ratio=2):
	'''
	Inputs:
	frame: current frame
	frame_prev: previous frame
	boxes: boxes of the detected objects
	confidences: confidences of detected objects 
	classIDs: indices of the classes objects belong to; use together with LABELS to get the corresponding labels 
	masks: masks of the object
	LABELS: all labels
	COLORS: colors assigned to labels
	oneobject: if only select one object with highest confidence
	Outputs:
	hist: orientation histogram
	center_xy: x and y coordinates of the center of the box
	mag: average magnitude (all direction/left/right/up/down) 
	frame_OF: modified current frame 
	elap: processing time
	'''
	start = time.time()
	cv.imshow("input", frame)
	frame_OF = copy.deepcopy(frame)
	frame_grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	frame_prev_grey = cv.cvtColor(frame_prev, cv.COLOR_BGR2GRAY)
	feature_boxes = []
	if len(boxes)==0:
		print('WARNING: No object detected! Adding NaN values to features.')
		# hist = np.full((1, nb_bins), np.nan)
		# box_info = np.full((1, 4), np.nan)
		# mag = np.full((1, 5), np.nan)
		feature_frame = np.full((1, nb_bins+4+5), np.nan)
	else:
		if oneobject:
			idxs = [np.argmax(np.array(confidences))]
		elif len(boxes) > 3: # keep the top 3 objects with highest confidence
			idxs = np.argsort(np.array(confidences))[-3:]
		else:
			idxs = np.argsort(np.array(confidences))
		for i in idxs:
			# Attention: mask need to be modified as well
			xs, ys, xl, yl, mask = expand_box(boxes[i], masks[i], frame.shape[1], frame.shape[0], ratio=ratio)
			box_info = np.expand_dims(np.array([(xs+xl)/2, (ys+yl)/2, int(xl-xs), int(yl-ys)]), axis=0)
			patch = frame_grey[ys:yl, xs:xl]
			patch_prev = frame_prev_grey[ys:yl, xs:xl]
			try:
				flow_patch = cv.calcOpticalFlowFarneback(patch_prev, patch, None, 0.5, 3, 15, 3, 5, 1.2, 0)
			except:
				print('WARNING: empty patches!')
				break
			flow_horizontal = flow_patch[..., 0]
			flow_vertical = flow_patch[..., 1]
			# Computes the magnitude and angle of the 2D vectors
			# DON'T TRUST cv.cartToPolar
			# magnitude, angle = cv.cartToPolar(flow_horizontal, flow_vertical, angleInDegrees=False)
			magnitude = np.absolute(flow_horizontal+1j*flow_vertical)
			angle = np.angle(flow_horizontal+1j*flow_vertical)
			if magnitude.mean() > 1e200:
				print("ABNORMAL!")
			mag = []
			mag.append([
				magnitude[mask].mean(), # avg magnitude
				flow_horizontal[np.logical_and(flow_horizontal>=0, mask)].mean(),  # up
				flow_horizontal[np.logical_and(flow_horizontal<=0, mask)].mean(),  # down
				flow_vertical[np.logical_and(flow_vertical<=0, mask)].mean(),  # left
				flow_vertical[np.logical_and(flow_vertical>=0, mask)].mean()  # right
			])
			mag = np.asarray(mag)
			hist = HOOF(magnitude, angle, nb_bins, mask=mask, fuzzy=False, normalize=False)
			feature_boxes.append(np.concatenate((hist, mag, box_info), axis=1))
			hsv = np.zeros_like(frame[ys:yl, xs:xl, :])
			hsv[..., 0] = angle*180/np.pi/2
			# hsv[..., 0] = 255
			hsv[..., 1] = 255
			hsv[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
			bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
			frame_OF[ys:yl, xs:xl, :] = bgr
			text = "{}: {:.4f}".format(detect_label, confidences[i])
			cv.putText(frame_OF, text, (xs, ys - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, 192, 2)
			blended = ((0.4 * np.array(192)) + (0.6 * bgr[mask])).astype("uint8")
			frame_OF[ys:yl, xs:xl, :][mask] = blended
		cv.imshow("object detection + optical flow", frame_OF)
		feature_frame = np.concatenate(tuple(feature_boxes), axis=0)
	end = time.time()
	elap = end - start
	return feature_frame, frame_OF, elap

