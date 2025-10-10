'''
This script is used to overlay two videos together.
It has three steps:
    - divide videos into pairs and overlay the videos in each pair (videos in each pair should have the same size and frame rate)
    - prepend content (QR code and instructions) to the overlayed video and make sure the video length is a multiple of 1 minute
    - concatenate the prepended videos into two trials
        
Author: yuanyuan.yao@kuleuven.be
'''

import numpy as np
import cv2 as cv
import os
import argparse
import vputils
import pickle
from feutils import extract_box_info_folder
from vputils import clean_boxes_info, expand_boxes


def get_weights(nb_frames, change_time, fps, target_weight, ATT, CROPONLY, change_duration_s=2):
    change_start_point = int(change_time * fps)
    change_duration = int(change_duration_s * fps)
    if ATT:
        weights = np.ones(nb_frames)
        if not CROPONLY:
            weights[change_start_point:change_start_point + change_duration] = np.linspace(1, target_weight, change_duration)
            weights[change_start_point + change_duration:] = target_weight
    else:
        weights = np.zeros(nb_frames)
        if not CROPONLY:
            weights[change_start_point:change_start_point + change_duration] = np.linspace(0, target_weight, change_duration)
            weights[change_start_point + change_duration:] = target_weight
    return weights


def overlay_two_videos(input_folder, output_folder, pair, MODE, change_time=120, max_time=np.inf, w1=0.5, w2=0.5, target_size=None, v1_box_info=None, v2_box_info=None, CROPONLY=False):
    video_1_ID = pair[0][:2]
    video_2_ID = pair[1][:2]
    att_ID = video_1_ID if MODE == 0 else video_2_ID
    if target_size is not None:
        if CROPONLY:
            crop_output_folder = os.path.join(output_folder, 'crop')
            if not os.path.exists(crop_output_folder):
                os.makedirs(crop_output_folder)
            output_path = os.path.join(crop_output_folder, att_ID + '_Crop.avi')
        else:
            output_path = os.path.join(output_folder, video_1_ID + '_' + video_2_ID + '_' + att_ID + '_Crop_Pair.avi')
    else:
        output_path = os.path.join(output_folder, video_1_ID + '_' + video_2_ID + '_' + att_ID + '_Pair.avi')
    if os.path.exists(output_path):
        print('The video pair: ', pair, ' already exists!')
        return
    print('Currently overlaying video pair: ', pair)
    # load the videos
    cap1 = cv.VideoCapture(os.path.join(input_folder, pair[0]))
    cap2 = cv.VideoCapture(os.path.join(input_folder, pair[1]))
    if not cap1.isOpened() or not cap2.isOpened():
        print("Cannot open file")
        exit()
    # get the frame rate of the videos
    fps1 = round(cap1.get(cv.CAP_PROP_FPS))
    fps2 = round(cap2.get(cv.CAP_PROP_FPS))
    # get number of frames of the videos
    nb_frames1 = int(cap1.get(cv.CAP_PROP_FRAME_COUNT))
    nb_frames2 = int(cap2.get(cv.CAP_PROP_FRAME_COUNT))
    # get the size of the videos
    frame_width1 = int(cap1.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height1 = int(cap1.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_width2 = int(cap2.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height2 = int(cap2.get(cv.CAP_PROP_FRAME_HEIGHT))
    assert fps1 == fps2, 'The frame rate of the two videos are not the same!'
    assert frame_width1 == frame_width2 and frame_height1 == frame_height2, 'The size of the two videos are not the same!'
    nb_frames = min(int(fps1 * max_time), nb_frames1, nb_frames2)
    if MODE == 0:
        v1_weights = get_weights(nb_frames, change_time, fps1, w1, ATT=True, CROPONLY=CROPONLY, change_duration_s=2)
        v2_weights = get_weights(nb_frames, change_time, fps2, w2, ATT=False, CROPONLY=CROPONLY, change_duration_s=2)
    else:
        v1_weights = get_weights(nb_frames, change_time, fps1, w1, ATT=False, CROPONLY=CROPONLY, change_duration_s=2)
        v2_weights = get_weights(nb_frames, change_time, fps2, w2, ATT=True, CROPONLY=CROPONLY, change_duration_s=2)
    # create an output video writer (avi)
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter(output_path, fourcc, fps1, (frame_width1, frame_height1))
    frame_count = 0
    while frame_count < nb_frames:
        # Read frames from the two videos
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        # Break the loop when either of the videos ends
        if not ret1 or not ret2:
            break
        if v1_box_info is not None:
            x_start, y_start, w, h = v1_box_info['box_info'][frame_count, :]
            frame1 = frame1[int(y_start):int(y_start + h), int(x_start):int(x_start + w), :]
        if v2_box_info is not None:
            x_start, y_start, w, h = v2_box_info['box_info'][frame_count, :]
            frame2 = frame2[int(y_start):int(y_start + h), int(x_start):int(x_start + w), :]
        # overlay the two frames
        frame_overlaid = cv.addWeighted(frame1, v1_weights[frame_count], frame2, v2_weights[frame_count], 0)
        # change the data type of the frame to uint8
        frame_overlaid = frame_overlaid.astype(np.uint8)
        # if the size of frame_ori does not match the target size, put the video in the center of the canvas
        if frame_overlaid.shape[0] != frame_height1 or frame_overlaid.shape[1] != frame_width1:
            frame = np.zeros((frame_height1, frame_width1, 3), dtype=np.uint8)
            x_start = int((frame_width1 - frame_overlaid.shape[1]) / 2)
            y_start = int((frame_height1 - frame_overlaid.shape[0]) / 2)
            frame[y_start:y_start + frame_overlaid.shape[0], x_start:x_start + frame_overlaid.shape[1], :] = frame_overlaid
        else:
            frame = frame_overlaid
        if not CROPONLY:
            # add a box at the top right corner
            if frame_count < int(change_time * fps1):
                frame[:120, -120:, :] = 128
            elif frame_count < int((change_time + 2) * fps1):
                frame[:120, -120:, :] = 0
            else:
                frame[:120, -120:, :] = 255
        # write the frame into the output video
        out.write(frame)
        frame_count += 1
    cap1.release()
    cap2.release()
    out.release()
    if CROPONLY:
        vputils.rescale_480(output_path, os.path.join(output_path[:-4] + '_480.avi'))


def add_prepended_content(merged_output, video_name, prepend_output):
    print('Currently prepending content to video: ', video_name)
    frame_width = args["canvaswidth"]
    frame_height = args["canvasheight"]
    video_pair_path = os.path.join(merged_output, video_name)
    cap = cv.VideoCapture(video_pair_path)
    fps = round(cap.get(cv.CAP_PROP_FPS))
    nb_frame = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    nb_frames_left = vputils.get_nb_prepend_frames(nb_frame, fps)
    # write video
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    output_path = os.path.join(prepend_output, video_name)
    out = cv.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    for append_frame_id in range(nb_frames_left):
        frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        progress = (append_frame_id + 1) / nb_frames_left
        vputils.add_progress_bar(frame, progress)
        sec = append_frame_id // fps
        if sec < 5:
            vputils.addText(frame, 'Please stay still for a moment', (192, 500))
        else:
            vputils.add_QR_code(frame, 'images/qrcode.png', 100, 100)
            vputils.addText(frame, 'Get some rest', (950, 200), fontScale=2, thickness=3)
            vputils.addText(frame, 'Task: Please focus on the character', (950, 400), fontScale=1.5, thickness=2)
            vputils.addText(frame, 'appearing at the beginning of the video', (950, 500), fontScale=1.5, thickness=2)
        out.write(frame)
    while True:
        ret, frame_ori = cap.read()
        if not ret:
            break
        # if the size of frame_ori does not match the target size, put the video in the center of the canvas
        if frame_ori.shape[0] != frame_height or frame_ori.shape[1] != frame_width:
            frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            x_start = int((frame_width - frame_ori.shape[1]) / 2)
            y_start = int((frame_height - frame_ori.shape[0]) / 2)
            frame[y_start:y_start + frame_ori.shape[0], x_start:x_start + frame_ori.shape[1], :] = frame_ori
        else:
            frame = frame_ori
        out.write(frame)
    cap.release()
    out.release()


def concat_videos(prepend_dir, concat_dir, MODE, RANDOMIZE=True):
    video_to_concat = [video for video in os.listdir(prepend_dir) if video.split('_')[MODE]==video.split('_')[2]]
    if RANDOMIZE:
        np.random.shuffle(video_to_concat)
    video_sequence = [video[:8] for video in video_to_concat]
    with open(concat_dir + 'Sequence_Trial_' + str(MODE + 1) + '.txt', 'w') as f:
        for video in video_sequence:
            f.write(video + '\n')
    output_path = os.path.join(concat_dir, 'Trial_' + str(MODE + 1) + '.avi') 
    video_to_concat = [os.path.join(prepend_dir, video) for video in video_to_concat]
    frame_width, frame_height, fps = vputils.get_frame_size_and_fps(video_to_concat[0])
    # initialize writer
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    for video in video_to_concat:
        print('Currently concatenating video: ', video)
        cap = cv.VideoCapture(video)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        cap.release()
    out.release()


def boxes_update_pair(v1_box_path, v2_box_path):
    v1_box_info = pickle.load(open(v1_box_path, 'rb'))
    v2_box_info = pickle.load(open(v2_box_path, 'rb'))
    v1_box_info['box_info'] = clean_boxes_info(v1_box_info['box_info'], v1_box_info["fps"], smooth=True)
    v2_box_info['box_info'] = clean_boxes_info(v2_box_info['box_info'], v2_box_info["fps"], smooth=True)
    # the shape of box_info is (num_frames, 4), and each row is [x, y, w, h]
    # find the max w and h for each video
    v1_max_w = np.max(v1_box_info['box_info'][:, 2])
    v1_max_h = np.max(v1_box_info['box_info'][:, 3])
    v2_max_w = np.max(v2_box_info['box_info'][:, 2])
    v2_max_h = np.max(v2_box_info['box_info'][:, 3])
    # the shape of the canvas is the max w and h of the two videos
    canvas_height = int(max(v1_max_h, v2_max_h))
    canvas_width = int(max(v1_max_w, v2_max_w)*1.2)
    len_v1 = v1_box_info['box_info'].shape[0]
    len_v2 = v2_box_info['box_info'].shape[0]
    len_final = min(len_v1, len_v2)
    v1_box_updated = np.zeros((len_final, 4))
    v2_box_updated = np.zeros((len_final, 4))
    v1_center_xy = np.zeros((len_final, 2))
    v2_center_xy = np.zeros((len_final, 2))
    for i in range(len_final):
        ori_x_start, ori_y_start, ori_width, ori_height = v1_box_info['box_info'][i, :]
        v1_box_updated[i, :], v1_center_xy[i, :]  = expand_boxes(canvas_height, canvas_width, ori_height, ori_width, ori_x_start, ori_y_start, v1_box_info['frame_height'], v1_box_info['frame_width'])
        ori_x_start, ori_y_start, ori_width, ori_height = v2_box_info['box_info'][i, :]
        v2_box_updated[i, :], v2_center_xy[i, :]  = expand_boxes(canvas_height, canvas_width, ori_height, ori_width, ori_x_start, ori_y_start, v2_box_info['frame_height'], v2_box_info['frame_width'])
    v1_box_info['box_info'] = v1_box_updated
    v2_box_info['box_info'] = v2_box_updated
    # add the center coordinates to the box info
    v1_box_info['center_xy'] = v1_center_xy
    v2_box_info['center_xy'] = v2_center_xy
    # save the box info as a pickle file
    with open(v1_box_path[:-8] + '_processed.pkl', 'wb') as f:
        pickle.dump(v1_box_info, f)
    with open(v2_box_path[:-8] + '_processed.pkl', 'wb') as f:
        pickle.dump(v2_box_info, f)
    target_size = (canvas_width, canvas_height)
    return v1_box_info, v2_box_info, target_size
    

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-ol', '--overlay', action='store_true',
    help='Include if overlay the two videos in each pair')
ap.add_argument('-pp', '--prepend', action='store_true',
    help='Include if prepend content to the overlayed video')
ap.add_argument('-ct', '--concat', action='store_true',
    help='Include if concatenate the prepended videos')
ap.add_argument('-cobj', '--cropobj', action='store_true',
    help='Include if crop the object from the video and then overlay the two videos in each pair')
ap.add_argument('-cobjo', '--cropobjonly', action='store_true',
    help='Include if only need to crop the object from the video for feature extraction')
ap.add_argument("-dl", "--detectlabel", type=int, default=0,
	help="class of objects to be detected (default: 0 -> person)")
ap.add_argument("-maxt", "--maxtime", type=int, default=600,
	help="maximum time of the video to be processed")
ap.add_argument("-ch", "--canvasheight", type=int, default=1080,
        help="height of the canvas")
ap.add_argument("-cw", "--canvaswidth", type=int, default=1920,
        help="width of the canvas")
ap.add_argument("-confi", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())


video_dict = {'01': 'the dancer in a white T-shirt', '13': 'the dancer in a blue shirt',
            #   '02': 'the mime actor', '12': 'the magician',
              '03': 'the acrobat actress on a unicycle', '05': 'the dancer in a black shirt',
              '04': 'the sitting magician', '09': 'the dancer',
              '06': 'the mime actor with a briefcase', '16': 'the mime actor with a hat',
              '07': 'the acrobat actor wearing a vest', '14': 'the sitting mime actress',
              '08': 'the sitting magician', '15': 'the dancer in a red shirt'}

input_folder = rf"videos\ORI"
output_folder = rf"C:\Users\Gebruiker\Documents\Experiments\svad_video"

extract_box_info_folder(input_folder, video_dict, args, MODE='OVERLAY')

overlay_output = os.path.join(output_folder, 'pair')
if not os.path.exists(overlay_output):
    os.makedirs(overlay_output)
prepend_output = os.path.join(output_folder, 'prepend')
if not os.path.exists(prepend_output):
    os.makedirs(prepend_output)
concat_output = os.path.join(output_folder, 'concat')
if not os.path.exists(concat_output):
    os.makedirs(concat_output)

if args["overlay"]:
    video_pairs = vputils.generate_video_pairs(input_folder, video_dict)
    for pair in video_pairs:
        v1_box_path = 'videos/OVERLAY/box_info/' + pair[0][:2] + '_box_info_raw.pkl'
        v2_box_path = 'videos/OVERLAY/box_info/' + pair[1][:2] + '_box_info_raw.pkl'
        v1_box_info, v2_box_info, target_size = boxes_update_pair(v1_box_path, v2_box_path) if args["cropobj"] or args["cropobjonly"] else (None, None, None)
        crop_only = True if args["cropobjonly"] else False
        overlay_two_videos(input_folder, overlay_output, pair, MODE=0, change_time=120, max_time=args["maxtime"], w1=0.5, w2=0.5, target_size=target_size, v1_box_info=v1_box_info, v2_box_info=v2_box_info, CROPONLY=crop_only)
        overlay_two_videos(input_folder, overlay_output, pair, MODE=1, change_time=120, max_time=args["maxtime"], w1=0.5, w2=0.5, target_size=target_size, v1_box_info=v1_box_info, v2_box_info=v2_box_info, CROPONLY=crop_only)
overlayed_video_list = [video for video in os.listdir(overlay_output) if video.endswith('.avi')]

if args["prepend"]:
    for pair in overlayed_video_list:
        add_prepended_content(overlay_output, pair, prepend_output)

if args["concat"]:
    concat_videos(prepend_output, concat_output, MODE=0, RANDOMIZE=True)
    concat_videos(prepend_output, concat_output, MODE=1, RANDOMIZE=True)

