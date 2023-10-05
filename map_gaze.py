'''
This script is used to map the gaze points to the original video frames.
The synchronization between the video and the gaze data is based on QR code detection. 

Author: yuanyuan.yao@kuleuven.be
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2 as cv
from vputils import get_frame_size_and_fps


def create_ref_timeline(time_array, start_time_ns):
    # find a time point that is closest to the start time
    start_idx = np.argmin(np.abs(time_array - start_time_ns))
    start_time_ns = time_array[start_idx]
    # Create a reference time array
    time_array_ref = time_array - start_time_ns
    return time_array_ref, start_idx


def get_gaze_from_frame_idx(frame_idx, fps, time_array, ref_time_array, gaze_df):
    ref_time_ns = frame_idx * 1e9 / fps
    time_idx = np.argmin(np.abs(ref_time_array - ref_time_ns))
    time_ns = time_array[time_idx]
    # find the id in gaze_df where the time is closest to the time_ns
    gaze_idx = np.argmin(np.abs(gaze_df['timestamp [ns]'] - time_ns))
    gaze = gaze_df.iloc[gaze_idx]
    return gaze


def visual_gaze(visual_video_path, ori_video_path, fps, width, height, time_array, ref_time_array, gaze_points):
    frame_idx = 0
    fourcc = cv.VideoWriter_fourcc(*"MJPG")
    writer = cv.VideoWriter(visual_video_path, fourcc, fps, (width, height), True)
    # Load video from the given path
    cap = cv.VideoCapture(ori_video_path)
    if not cap.isOpened():
        print("Cannot open file")
        exit()
    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            gaze = get_gaze_from_frame_idx(frame_idx, fps, time_array, ref_time_array, gaze_points)
            if gaze['gaze detected on surface']:
                x = int(gaze['gaze position on surface x [normalized]']*width)
                y = int(gaze['gaze position on surface y [normalized]']*height)
                print('frame_idx: ', frame_idx, 'x: ', x, 'y: ', y)
                cv.circle(frame, (x, y), 25, (0, 0, 255), -1)
            # Display the resulting frame
            writer.write(frame)
            frame_idx += 1
            # Press Q on keyboard to exit
            # if cv.waitKey(1) & 0xFF == ord('q'):
            #     break
        # Break the loop
        else:
            break
    writer.release()
    cap.release()


if __name__ == "__main__":

    Trial_Name = 'Trial_1'
    path_raw = 'data/' + Trial_Name + '/Raw/'
    path_map = 'data/' + Trial_Name + '/Marker_Mapper/'

    # Use Pandas to read the csv file
    # Get the section id
    section_info = pd.read_csv(path_raw + 'sections.csv')
    section_id = section_info['section id'][0]
    # Get world timestamps
    world_timestamps = pd.read_csv(path_raw + '2023-09-25_17-13-05-453d15aa/world_timestamps.csv')
    time_array = world_timestamps['timestamp [ns]'].values
    # Get the mapped gaze points
    gaze_points = pd.read_csv(path_map + 'gaze.csv')
    gaze_points = gaze_points[gaze_points['section id'] == section_id]
    timeline_gaze = gaze_points['timestamp [ns]'].values
  

    # Option 1: use the time point when the QR code is detected to decide the start time
    QR_ifdetected = np.load('QR_detected_pyzbar.npy')
    detected_idx_1st = np.where(QR_ifdetected == 1)[0][0]
    start_time_ns = time_array[detected_idx_1st] - int(30*1e9) # 30 seconds before the first QR code is detected
    ref_time_array, start_idx = create_ref_timeline(time_array, start_time_ns)

    ori_video_path = "videos/video_test_size_120_fps_30_len_7.avi"
    visual_video_path = "videos/visual_gaze_2_new.avi"
    width, height, fps = get_frame_size_and_fps(ori_video_path)
    visual_gaze(visual_video_path, ori_video_path, fps, width, height, time_array, ref_time_array, gaze_points)
