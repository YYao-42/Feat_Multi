import numpy as np
import cv2 as cv
import os
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
        '-vf', 'scale=854:480',
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


