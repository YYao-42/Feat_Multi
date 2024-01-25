'''
This script is used to overlay two videos together.
It has three steps:
    - divide videos into pairs and overlay the videos in each pair (videos in each pair should have the same length, size, and frame rate)
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
import copy
import scipy
import pandas as pd
import mmcv
import mmengine
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules


def generate_video_pairs(dir_parent, video_dict=None):
    # list the videos in the video_dict (the keys)
    if video_dict is not None:
        video_list = [video for video in os.listdir(dir_parent) if video.startswith(tuple(video_dict.keys()))]
    else:
        # list the videos in the directory that do not end with '_drop.mp4'
        video_list = [video for video in os.listdir(dir_parent) if not video.endswith('_drop.mp4')]
    # get the number of frames of each video
    num_frames = []
    for video in video_list:
        video_path = dir_parent + video
        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            print("Cannot open file")
            exit()
        num_frames.append(int(cap.get(cv.CAP_PROP_FRAME_COUNT)))
        cap.release()
    # sort the video list based on the number of frames
    video_list = [x for _, x in sorted(zip(num_frames, video_list))]
    # divide the video list into pairs
    video_pairs = []
    for i in range(0, len(video_list), 2):
        video_pairs.append([video_list[i], video_list[i + 1]])
    return video_pairs


def extract_box_info_folder(folder_path, video_dict, args):
    # Choose to use a config and initialize the detector
    config_file = 'checkpoints\mask-rcnn_r50-caffe_fpn_ms-poly-3x_coco.py'
    # Setup a checkpoint file to load
    checkpoint_file = 'checkpoints\mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
    # register all modules in mmdet into the registries
    register_all_modules()
    # build the model from a config file and a checkpoint file
    net = init_detector(config_file, checkpoint_file, device='cuda:0')  # or device='cuda:0'
    # create a folder to store the box info
    box_info_output = 'videos/OVERLAY/box_info/'
    if not os.path.exists(box_info_output):
        os.makedirs(box_info_output)
    # for videos in the video_dict (the keys), check whether the box info file exists
    # if not, generate the box info file
    for video in video_dict.keys():
        video_name = [v for v in os.listdir(folder_path) if v.startswith(video)][0]
        box_info_path = box_info_output + video + '_box_info_mm.pkl'
        if not os.path.exists(box_info_path):
            print('[INFO] Currently generating box info for video: ', video_name)
            # in the dir folder, search for the video with file name starts with the video ID
            video_path = folder_path + video_name
            box_info, fps, frame_width, frame_height = extract_box_info_video(video_path, net, args)
            # save box_info, fps, frame_width, frame_height as a dictionary
            box_info = {'box_info': box_info, 'fps': fps, 'frame_width': frame_width, 'frame_height': frame_height}
            # save the box info as a pickle file
            with open(box_info_path, 'wb') as f:
                pickle.dump(box_info, f)
        else:
            print('[INFO] Box info for video: ', video_name, ' already exists!')


def extract_box_info_video(video_path, net, args):
    vs = cv.VideoCapture(video_path)
    fps = round(vs.get(cv.CAP_PROP_FPS))
    frame_width = int(vs.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vs.get(cv.CAP_PROP_FRAME_HEIGHT))
    max_frame = int(fps * args["maxtime"])
    try:
        total = min(int(vs.get(cv.CAP_PROP_FRAME_COUNT)), max_frame)
        print("[INFO] {} total frames in video".format(total))
    # an error occurred while trying to determine the total
    # number of frames in the video file
    except:
        print("[INFO] could not determine # of frames in video")
        print("[INFO] no approx. completion time can be provided")
        total = -1
    box_list = []
    frame_count = 0
    # loop over frames from the video file stream
    while frame_count < max_frame:
        # read the next frame from the file
        grabbed, frame = vs.read()
        # if the frame was not grabbed, then we have reached the end
        # of the stream
        if not grabbed:
            break
        # Detect objects 
        boxes, _, elap_OS = vputils.object_seg_mmdetection(frame, net, args)
        if len(boxes) > 1:
            print('More than one object detected! Only the first one is kept!')
        if len(boxes) == 0:
            print('No object detected! Set the box info to NaN!')
            boxes.append([np.nan, np.nan, np.nan, np.nan])
        box_list.append(np.expand_dims(np.array(boxes[0]), axis=0))
        if total > 0:
            print("[INFO] estimated total time to finish: {:.4f}".format(elap_OS * total))
            total = -1
        frame_count += 1
    box_info = np.concatenate(tuple(box_list), axis=0)
    vs.release()
    return box_info, fps, frame_width, frame_height


def overlay_two_videos(dir_parent, dir_output, pair, max_time=np.inf, target_size=None, v1_box_info=None, v2_box_info=None):
        print('Currently overlaying video pair: ', pair)
        # load the videos
        video_1_ID = pair[0][:2]
        video_2_ID = pair[1][:2]
        cap1 = cv.VideoCapture(dir_parent + pair[0])
        cap2 = cv.VideoCapture(dir_parent + pair[1])
        if not cap1.isOpened() or not cap2.isOpened():
            print("Cannot open file")
            exit()
        # get the frame rate of the videos
        fps1 = round(cap1.get(cv.CAP_PROP_FPS))
        fps2 = round(cap2.get(cv.CAP_PROP_FPS))
        # get the size of the videos
        frame_width1 = int(cap1.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height1 = int(cap1.get(cv.CAP_PROP_FRAME_HEIGHT))
        frame_width2 = int(cap2.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height2 = int(cap2.get(cv.CAP_PROP_FRAME_HEIGHT))
        assert fps1 == fps2, 'The frame rate of the two videos are not the same!'
        assert frame_width1 == frame_width2 and frame_height1 == frame_height2, 'The size of the two videos are not the same!'
        max_frame = int(fps1 * max_time)
        # create an output video writer (avi)
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        if target_size is not None:
            output_path = dir_output + video_1_ID + '_' + video_2_ID + '_Crop_Pair.avi'
            out = cv.VideoWriter(output_path, fourcc, fps1, target_size)
            output_path1 = dir_output + video_1_ID + '_Crop.avi'
            out1 = cv.VideoWriter(output_path1, fourcc, fps1, target_size)
            output_path2 = dir_output + video_2_ID + '_Crop.avi'
            out2 = cv.VideoWriter(output_path2, fourcc, fps1, target_size)
        else:
            output_path = dir_output + video_1_ID + '_' + video_2_ID + '_Pair.avi'
            out = cv.VideoWriter(output_path, fourcc, fps1, (frame_width1, frame_height1))
        frame_count = 0
        while frame_count < max_frame:
            # Read frames from the two videos
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            # Break the loop when either of the videos ends
            if not ret1 or not ret2:
                break
            if v1_box_info is not None:
                x_start, y_start, w, h = v1_box_info['box_info'][frame_count, :]
                frame1 = frame1[int(y_start):int(y_start + h), int(x_start):int(x_start + w), :]
                out1.write(frame1)
            if v2_box_info is not None:
                x_start, y_start, w, h = v2_box_info['box_info'][frame_count, :]
                frame2 = frame2[int(y_start):int(y_start + h), int(x_start):int(x_start + w), :]
                out2.write(frame2)
            # overlay the two frames
            frame = cv.addWeighted(frame1, 0.5, frame2, 0.5, 0)
            # write the frame into the output video
            out.write(frame)
            frame_count += 1
        cap1.release()
        cap2.release()
        out.release()
        if target_size is not None:
            out1.release()
            out2.release()


def get_nb_prepend_frames(nb_vid_frames, fs, force=True):
    min = nb_vid_frames // (fs * 60)
    sec = (nb_vid_frames % (fs * 60)) // fs
    if sec < 30 or force:
        target_video_len = (60 * fs) * (min + 1)
    else:
        target_video_len = (60 * fs) * (min + 2)
    nb_frames_left = target_video_len - nb_vid_frames
    return nb_frames_left


def add_prepended_content(MODE, stitched_output, video_name, prepend_output, descrip_dict):
    print('Currently prepending content to video: ', video_name)
    video_pair_path = stitched_output + video_name
    pair_ID = video_name[:5].split('_')
    video_ID_attend = pair_ID[0] if MODE == 0 else pair_ID[1]
    # get the description of the attended video
    video_descrip_attend = descrip_dict[video_ID_attend]
    cap = cv.VideoCapture(video_pair_path)
    fps = round(cap.get(cv.CAP_PROP_FPS))
    nb_frame = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    nb_frames_left = get_nb_prepend_frames(nb_frame, fps)
    # write video
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    output_path = prepend_output + video_name[:-4] + '_MODE_' + str(MODE) + '.avi'
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
            vputils.addText(frame, 'Next Task: ', (950, 200), fontScale=2, thickness=3)
            vputils.addText(frame, 'Please focus on', (950, 400), fontScale=1.5, thickness=2)
            vputils.addText(frame, video_descrip_attend, (950, 500), fontScale=1.5, thickness=2)
        out.write(frame)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    cap.release()
    out.release()


def concat_videos(prepend_dir, concat_dir, ifDUAL=False):
    if ifDUAL:
        video_to_concat = [video for video in os.listdir(prepend_dir) if 'DUAL' in video]
        output_path = concat_dir + 'Trial_2' + '.avi'
    else:
        video_to_concat = [video for video in os.listdir(prepend_dir) if 'DUAL' not in video]
        output_path = concat_dir + 'Trial_1' + '.avi'
    video_to_concat.sort()
    video_to_concat = [prepend_dir + video for video in video_to_concat]
    frame_width, frame_height, fps = vputils.get_frame_size_and_fps(video_to_concat[0])
    # initialize writer
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    for video in video_to_concat:
        cap = cv.VideoCapture(video)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        cap.release()
    out.release()


def expand_boxes(target_height, target_width, ori_height, ori_width, ori_x_start, ori_y_start, ori_frame_height, ori_frame_width):
    target_height = min(target_height, ori_frame_height)
    target_width = min(target_width, ori_frame_width)
    # expand the boxes to the size of the canvas, with the center of the box unchanged
    new_x_start = int(max(0, ori_x_start - (target_width - ori_width) / 2))
    new_y_start = int(max(0, ori_y_start - (target_height - ori_height) / 2))
    new_x_end = min(ori_frame_width, new_x_start + target_width)
    new_y_end = min(ori_frame_height, new_y_start + target_height)
    # adjust the start coordinates if the end coordinates are out of the frame
    new_x_start = new_x_end - target_width
    new_y_start = new_y_end - target_height
    assert new_x_start >= 0 and new_y_start >= 0, 'New box out of frame!'
    new_box_info = [new_x_start, new_y_start, target_width, target_height]
    return new_box_info


def clean_boxes_info(box_info, fps, smooth=True):
    y = copy.deepcopy(box_info)
    _, nb_col = y.shape
    for i in range(nb_col):
        # interpolate NaN values (linearly)
        nans, x= np.isnan(y[:,i]), lambda z: z.nonzero()[0]
        if any(nans):
            f1 = scipy.interpolate.interp1d(x(~nans), y[:,i][~nans], fill_value='extrapolate')
            y[:,i][nans] = f1(x(nans))
        # find outliers and replace them with interpolated values
        # outliers are defined as values that are more than 3 standard deviations away from the mean
        outliers = np.abs(y[:,i] - np.mean(y[:,i])) > 3*np.std(y[:,i])
        if any(outliers):
            f1 = scipy.interpolate.interp1d(x(~outliers), y[:,i][~outliers], fill_value='extrapolate')
            y[:,i][outliers] = f1(x(outliers))
        if smooth:
            window_size = int(fps*3)  # Set the size of the moving window
            y[:,i] = pd.Series(y[:,i]).rolling(window=window_size, min_periods=1, center=True).mean().values
    return y


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
    canvas_width = int(max(v1_max_w, v2_max_w)*2)
    len_v1 = v1_box_info['box_info'].shape[0]
    len_v2 = v2_box_info['box_info'].shape[0]
    len_final = min(len_v1, len_v2)
    v1_box_updated = np.zeros((len_final, 4))
    v2_box_updated = np.zeros((len_final, 4))
    for i in range(len_final):
        ori_x_start, ori_y_start, ori_width, ori_height = v1_box_info['box_info'][i, :]
        v1_box_updated[i, :] = expand_boxes(canvas_height, canvas_width, ori_height, ori_width, ori_x_start, ori_y_start, v1_box_info['frame_height'], v1_box_info['frame_width'])
        ori_x_start, ori_y_start, ori_width, ori_height = v2_box_info['box_info'][i, :]
        v2_box_updated[i, :] = expand_boxes(canvas_height, canvas_width, ori_height, ori_width, ori_x_start, ori_y_start, v2_box_info['frame_height'], v2_box_info['frame_width'])
    v1_box_info['box_info'] = v1_box_updated
    v2_box_info['box_info'] = v2_box_updated
    target_size = (canvas_width, canvas_height)
    return v1_box_info, v2_box_info, target_size
    

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--input", required=True,
# 	help="path to input video")
# ap.add_argument("-o", "--output", required=True,
# 	help="path to output video")
ap.add_argument('-ol', '--overlay', action='store_true',
    help='Include if overlay the two videos in each pair')
ap.add_argument("-m", "--mask-rcnn", default='mask-rcnn',
	help="base path to mask-rcnn directory")
ap.add_argument('-cobj', '--cropobj', action='store_true',
    help='Include if crop the object from the video')
# ap.add_argument('-GPU', '--GPU', action='store_true',
#     help='Include if use GPU acceleration')
ap.add_argument("-dl", "--detectlabel", type=int, default=0,
	help="class of objects to be detected (default: 0 -> person)")
ap.add_argument("-maxt", "--maxtime", type=int, default=600,
	help="maximum time of the video to be processed")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())


video_dict = {'01': 'the dancer in a white shirt', '05': 'the dancer in a black shirt',
              '03': 'the acrobat actress on a unicycle', '08': 'the sitting magician',
              '06': 'the mime actor with a briefcase', '04': 'the sitting magician',
              '09': 'the dancer', '07': 'the acrobat actor',
              '13': 'the dancer in a blue shirt', '15': 'the dancer in a red shirt',
              '16': 'the mime arctor with a hat', '14': 'the sitting mime actress'}
dir_parent = '../Feature extraction/videos/ori_video/'

# video_dict = {'13': 'the dancer in a blue shirt', '15': 'the dancer in a red shirt'}
# dir_parent = '../Feature extraction/videos/ori_video/'

extract_box_info_folder(dir_parent, video_dict, args)

overlayed_output = 'videos/OVERLAY/pairs/'
if not os.path.exists(overlayed_output):
    os.makedirs(overlayed_output)

if args["overlay"]:
    video_pairs = generate_video_pairs(dir_parent, video_dict)
    for pair in video_pairs:
        # find the box info for the two videos in the pair
        v1_box_path = 'videos/OVERLAY/box_info/' + pair[0][:2] + '_box_info_mm.pkl'
        v2_box_path = 'videos/OVERLAY/box_info/' + pair[1][:2] + '_box_info_mm.pkl'
        v1_box_info, v2_box_info, target_size = boxes_update_pair(v1_box_path, v2_box_path)
        overlay_two_videos(dir_parent, overlayed_output, pair, args["maxtime"], target_size, v1_box_info, v2_box_info)
overlayed_video_list = [video for video in os.listdir(overlayed_output) if video.endswith('.avi')]

# # canvas_height = 1080
# # canvas_width = 1920

# OVERLAYED = False
# PREPENDED = True
# CONCATENATED = True

# # if a output directory does not exist, create one
# overlayed_output = 'videos/OVERLAY/pairs/'
# if not os.path.exists(overlayed_output):
#     os.makedirs(overlayed_output)
# prepend_output = 'videos/OVERLAY/prepend/'
# if not os.path.exists(prepend_output):
#     os.makedirs(prepend_output)
# concat_output = 'videos/OVERLAY/concat/'
# if not os.path.exists(concat_output):
#     os.makedirs(concat_output)

# if not OVERLAYED:
#     dir_parent = '../Feature extraction/videos/ori_video/'
#     video_pairs = generate_video_pairs(dir_parent)
#     for pair in video_pairs:
#         overlay_two_videos(dir_parent, overlayed_output, pair, max_time)
# overlayed_video_list = [video for video in os.listdir(overlayed_output) if video.endswith('.avi')]

# if not PREPENDED:
#     MODE = 0
#     for stitched_pair in overlayed_video_list:
#         add_prepended_content(MODE, overlayed_output, stitched_pair, prepend_output, video_dict)
#     MODE = 1
#     for stitched_pair in overlayed_video_list:
#         add_prepended_content(MODE, overlayed_output, stitched_pair, prepend_output, video_dict)

# if not CONCATENATED:
#     concat_videos(prepend_output, concat_output)
#     concat_videos(prepend_output, concat_output, ifDUAL=True)

# if OVERLAYED and PREPENDED and CONCATENATED:
#     print('You are all set!')