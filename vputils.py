import numpy as np
import cv2 as cv
import os
import copy
import scipy
import pandas as pd
import subprocess

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


def generate_video_pairs(dir_parent, video_dict=None):
    # list the videos in the video_dict (the keys)
    if video_dict is not None:
        video_list = []
        video_IDs = list(video_dict.keys())
        for ID in video_IDs:
            video_list.append([video for video in os.listdir(dir_parent) if video.startswith(ID)][0])
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


def rescale_480(input_file, output_file):
    # Build the ffmpeg command to transform the video
    command = [
        'ffmpeg',
        '-i', input_file,
        '-vf', 'scale=854:480,fps=30',  
        '-c:a', 'copy',  # Copy audio codec without re-encoding
        '-c:v', 'libx264',  # Use H.264 video codec
        '-crf', '23',  # Constant Rate Factor (quality), adjust as needed
        output_file
    ]
    # Execute the ffmpeg command
    subprocess.run(command, check=True)
    

def get_nb_prepend_frames(nb_vid_frames, fs, force=False):
    min = nb_vid_frames // (fs * 60)
    sec = (nb_vid_frames % (fs * 60)) // fs
    if sec < 30 or force:
        target_video_len = (60 * fs) * (min + 1)
    else:
        target_video_len = (60 * fs) * (min + 2)
    nb_frames_left = target_video_len - nb_vid_frames
    return nb_frames_left


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
    center_xy = [new_x_start + target_width / 2, new_y_start + target_height / 2]
    return new_box_info, center_xy


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