import numpy as np
import cv2 as cv
import time
from mmdet.apis import init_detector, inference_detector
import torch


def get_frame_size_and_fps(video_path):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open file")
        exit()
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS)
    cap.release()
    return width, height, fps


def add_QR_code(frame, QR_code_path, start_x, start_y):
    QR_code = cv.imread(QR_code_path)
    end_x = start_x + QR_code.shape[1]
    end_y = start_y + QR_code.shape[0]
    frame[start_y:end_y, start_x:end_x, :] = QR_code
    return frame


def add_progress_bar(frame, progress, bar_height=20):
    H, W = frame.shape[:2]
    bar_width = int(W * 0.8)
    start_x = int(W * 0.1)
    start_y = int(H * 0.9)
    end_x = start_x + bar_width
    end_y = start_y + bar_height
    frame[start_y:end_y, start_x:end_x, :] = 255
    end_x = start_x + int(bar_width * progress)
    frame[start_y:end_y, start_x:end_x, :] = 0
    frame[start_y:end_y, start_x:end_x, 0] = 255
    return frame


def addText(frame, text, pos, color=(255,255,255), font=cv.FONT_HERSHEY_SIMPLEX, fontScale=3, thickness=3):
    return cv.putText(frame, text, pos, font, fontScale, color, thickness, cv.LINE_AA)


def object_seg_maskrcnn(frame, net, args, LABELS, detect_label='person'):
	'''
	Use pretrained Mask R-CNN network to perform object segmentation
	Inputs:
	frame: current frame
	net: mask r-cnn net for object segmentation
	args: pre-defined parameters
	LABELS: labels of all classes
	detect_label: class to be detected; Default: person
	Outputs:
	boxes: boxes of the detected objects
	confidences: confidences of detected objects 
	classIDs: indices of the classes objects belong to; use together with LABELS to get the corresponding labels 
	masks: masks of the object
	elap: processing time
	'''
	start = time.time()
	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	blob = cv.dnn.blobFromImage(frame, swapRB=True, crop=False)
	net.setInput(blob)
	(boxes_info, masks_info) = net.forward(["detection_out_final",
		"detection_masks"])
	# initialize our lists of detected bounding boxes, confidences,
	# and masks, respectively
	boxes = []
	confidences = []
	classIDs = []
	masks = []
	# loop over the number of detected objects
	for i in range(0, boxes_info.shape[2]):
		# extract the class ID of the detection along with the
		# confidence (i.e., probability) associated with the
		# prediction
		classID = int(boxes_info[0, 0, i, 1])
		confidence = boxes_info[0, 0, i, 2]
		# filter out weak predictions by ensuring the detected
		# probability is greater than the minimum probability
		if confidence > args["confidence"] and LABELS[classID]==detect_label:
			# scale the bounding box coordinates back relative to the
			# size of the frame and then compute the width and the
			# height of the bounding box
			(H, W) = frame.shape[:2]
			box = boxes_info[0, 0, i, 3:7] * np.array([W, H, W, H])
			(startX, startY, endX, endY) = box.astype("int")
			boxW = endX - startX
			boxH = endY - startY
			# extract the pixel-wise segmentation for the object,
			# resize the mask such that it's the same dimensions of
			# the bounding box, and then finally threshold to create
			# a *binary* mask
			mask = masks_info[i, classID]
			mask = cv.resize(mask, (boxW, boxH),
				interpolation=cv.INTER_NEAREST)
			mask = (mask > args["threshold"])
			# extract the ROI of the image but *only* extracted the
			# masked region of the ROI
			# roi = frame[startY:endY, startX:endX][mask]
			boxes.append([startX, startY, boxW, boxH])
			confidences.append(float(confidence))
			classIDs.append(classID)
			masks.append(mask)

	end = time.time()
	elap = end-start
	return boxes, confidences, classIDs, masks, elap


def object_seg_mmdetection(frame, net, args):
	'''
	Use pretrained Mask R-CNN network to perform object segmentation
	Inputs:
	frame: current frame
	net: mask r-cnn net for object segmentation (using mmdetection package)
	args: pre-defined parameters
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
	bboxes = bboxes.cpu().numpy().astype(int)
	masks = masks.cpu().numpy()
	masks_list = [masks[i,bboxes[i,1]:bboxes[i,3],bboxes[i,0]:bboxes[i,2]] for i in range(masks.shape[0])]
	bboxes_H = bboxes[:,3] - bboxes[:,1]
	bboxes_W = bboxes[:,2] - bboxes[:,0]
	# replace the last two columns with the height and width of the bounding box
	bboxes[:,2] = bboxes_W
	bboxes[:,3] = bboxes_H
	bboxes_list = [bboxes[i,:] for i in range(bboxes.shape[0])]
	end = time.time()
	elap = end-start
	return bboxes_list, masks_list, elap