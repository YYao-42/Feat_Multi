'''
This script is used to insert flickering boxes into videos and merge them together with cross-fading effect.
Put the videos to be merged in a folder and pass its path to --input.
The fps of the videos is required to be specified by --fps. [can be modified to read from the videos]
The output videos will be saved in the folder specified by --output.

The script can be easily adapted to inserting boxes only or concatenating videos only.
For the latter case, make sure that the videos have the same fps and the same resolution.

Author: yuanyuan.yao@kuleuven.be
'''

import numpy as np
import cv2 as cv
import os
import vputils
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video folder")
ap.add_argument("-o", "--output", required=True,
	help="path to output video folder")
ap.add_argument("-fps", "--fps", type=int, required=True, 
	help="fps of videos")
ap.add_argument('-co', '--concatonly', action='store_true',
    help='Include if boxes are inserted already and we only need to concatenate different clips')
ap.add_argument("-bint", "--blkint", type=int, default=30,
	help="blink interval")
ap.add_argument("-bdur", "--blkdur", type=float, default=0.5,
	help="blink duration")
ap.add_argument("-ext", "--extperc", type=float, default=0.1,
	help="W and H of frames are extended by ext%")
ap.add_argument("-box", "--boxperc", type=float, default=0.1,
	help="side length of the flickering box as the percentage of the original H of the frame")
ap.add_argument("-fdur", "--fadedur", type=int, default=4,
	help="duration of cross fading")
args = vars(ap.parse_args())

input_folder = args["input"]
if not input_folder.endswith('/'):
    input_folder = input_folder + '/'
video_list = os.listdir(input_folder)
fps = args["fps"]
blink_interval = args["blkint"]
blink_duration = args["blkdur"]
ext_percent = args["extperc"]
box_percent = args["boxperc"]
fade_duration = args["fadedur"]

if args["concatonly"]:
    # If flickering boxes have been inserted
    print("[INFO] Flickering box has been inserted")
    video_path = input_folder + video_list[0]
    cap_ori = cv.VideoCapture(video_path)
    ret, frame = cap_ori.read()
    H, W = frame.shape[:2]
    H_extend = int(H*(1+ext_percent))
    W_extend = int(W*(1+ext_percent))
    box_len = int(H*box_percent)
    cap_ori.release()
else:
    print("[INFO] Inserting flickering box into videos ...")
    for video in video_list:
        print('currently working on: ' + video)
        video_path = input_folder + video
        write_path = args["output"] + video[:-4] + '_box.avi'
        H_extend, W_extend, box_len = vputils.add_flickering_box(video_path, write_path, fps, blink_interval, blink_duration, ext_percent, box_percent)

video_box_list = [video for video in os.listdir(args["output"]) if video.endswith('_box.avi')]
write_path = args["output"] + 'exp.avi'

print("[INFO] Concatenating videos ...")
# initialize writer
fourcc = cv.VideoWriter_fourcc(*"MJPG")
writer = cv.VideoWriter(write_path, fourcc, fps, (W_extend, H_extend), True)
# append 1s black frames at the beginning
for i in range(fps):
    black = np.zeros([H_extend, W_extend, 3], dtype=np.uint8)
    writer.write(black)
# do cross fading at the boundary of two videos
last_frame = None
for video_box in video_box_list:
    print('currently working on: ' + video_box)
    video_box_path = args["output"] + video_box
    cap = cv.VideoCapture(video_box_path)
    ret, first_frame = cap.read()
    if (last_frame is not None) and (first_frame is not None):
        frame_list = vputils.cross_fading(last_frame, first_frame, fade_duration, fps, box_len)
        for frame in frame_list:
            writer.write(frame)
    else:
        writer.write(first_frame)
    while (True):
        ret, frame = cap.read()
        if not ret:
            break
        else:
            last_frame = frame
            writer.write(frame)
# append 1s grey frames at the end
for i in range(fps):
    grey = np.zeros([H_extend, W_extend, 3], dtype=np.uint8)
    grey.fill(169)
    writer.write(grey)
writer.release()
cap.release()