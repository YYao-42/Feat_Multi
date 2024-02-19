'''
This script is used to map the gaze points to the original video frames.
The synchronization between the video and the gaze data is based on QR code detection. 
- Download the time series CSV and scene video from Pupil Cloud to 'data/Pilot_X/Raw/', and the enrichment data to 'data/Pilot_X/Marker_Mapper/'.
- Set 'Pilot_Name' and 'Trial' to their respective values, then execute the script.
    - The script will first detect the QR code in the scene video and then visualize and save the detection results.
    - Based on these results, users are prompted to accept or adjust the start and end timings of the videos. If the number of frames with QR codes differs from expectations, users can refer to the visualization video to decide on adjustments. (If the number of start timings is not consistent with the number of videos, the user may need to plot the detection results and mark the start and end timings manually.)
    - The script will then map the gaze points to the original video frames and save the results (to the folder that saves experiment data). The visualization of the gaze points can also be saved as a video.
- If start and end timings are correct, and gaze points are accurately mapped to video frames (by comparing the results to the results on Pupil Cloud), delete the visualization videos (due to their large file size).

Author: yuanyuan.yao@kuleuven.be
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import cv2 as cv
import os
from pyzbar.pyzbar import decode
from vputils import get_frame_size_and_fps, get_nb_prepend_frames


def detect_QR_code_pyzbar(frame):
    ifdetected = False
    # Convert image to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    try:
        QR_detector = decode(gray)
    except:
        return frame, ifdetected
    if len(QR_detector) > 0:
        ifdetected = True
        for QR_code in QR_detector:
            code_info = QR_code.data.decode('utf-8')
            code_points = QR_code.polygon
            # Draw bounding box around the QR code
            for p in code_points:
                point = (p.x, p.y)
                cv.circle(frame, tuple(point), 5, (0, 0, 255), -1)
            # show the QR code info
            cv.putText(frame, code_info, (int(code_points[0].x), int(code_points[0].y - 10)),
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return frame, ifdetected


def visual_QR_codes(visual_video_path, test_video_path, index_path, fps, width, height):
    QR_detected = []
    fourcc = cv.VideoWriter_fourcc(*"MJPG")
    writer = cv.VideoWriter(visual_video_path, fourcc, fps, (width, height), True)
    # Load video from the given path
    cap = cv.VideoCapture(test_video_path)
    if not cap.isOpened():
        print("Cannot open file")
        exit()
    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            # Detect QR code
            frame, ifdetected = detect_QR_code_pyzbar(frame)
            QR_detected.append(ifdetected)
            # Display the resulting frame
            writer.write(frame)
            # Press Q on keyboard to exit
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        # Break the loop
        else:
            break
    # save the QR detection results
    QR_detected = np.array(QR_detected)
    np.save(index_path, QR_detected)
    writer.release()
    cap.release()


def find_edge(data_TF, fs=30, UP=True):
    if UP == False:
        data_TF = data_TF[::-1]
    data_T_idx = np.where(data_TF)[0]
    data_cleaned_idx = [data_T_idx[0]] 
    for i in data_T_idx:
        if i - data_cleaned_idx[-1] > fs*200:
            data_cleaned_idx.append(i)
    if UP == False:
        data_cleaned_idx = len(data_TF) - np.array(data_cleaned_idx) - 1
    return sorted(data_cleaned_idx)


def get_start_end_idx(QR_ifdetected, fs):
    start_idx = find_edge(QR_ifdetected, fs, UP=True)
    end_idx = find_edge(QR_ifdetected, fs, UP=False)
    assert len(start_idx) == len(end_idx) == nb_videos, "The number of start and end timings should be the same as the number of videos!"
    # check whether the distance between the start and end matches the number of the prepended frames
    for i in range(nb_videos):
        video = video_sequence[i]
        video_id = video.split('_')[0]
        feature_path = 'features/' + video_id + '_mask.pkl'
        feature = pickle.load(open(feature_path, 'rb'))
        len_video = len(feature)
        nb_prepended = get_nb_prepend_frames(len_video, fs)
        print('video: ', video)
        print('nb of frames with QR code (in theory): ', nb_prepended-fs*5) # 5 seconds of instruction
        print('nb of frames with QR code (based on detection): ', end_idx[i]-start_idx[i]+1)
        # Ask the user whether to adjust the start and end timings
        print('The start and end timings are: ', start_idx[i], end_idx[i])
        print('Do you want to adjust the start and end timings?')
        print('If yes, please input the new start and end timings (format: start_idx,end_idx)')
        print('If no, please input "no"')
        user_input = input()
        if user_input != 'no':
            start_idx[i], end_idx[i] = tuple(map(int, user_input.split(',')))
        else:
            continue
    # Save the start and end timings with key as the video name
    start_end_timings = dict(zip(video_sequence, zip(start_idx, end_idx)))
    np.save(path_raw + folder_name + '/start_end_idx.npy', start_end_timings)


def get_video_sequence(path_sequence_file):
    with open(path_sequence_file, 'r') as file:
        # read a list of lines into data, do not include \n
        data = file.read().split('\n')
        # remove the empty string, if exists
        data = list(filter(None, data))
    return data


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


def get_gaze(gaze_path, ori_video_path, fps, time_array, ref_time_array, gaze_points, magic_ratio):
    frame_idx = 0
    gaze_list = []
    # Load video from the given path
    cap = cv.VideoCapture(ori_video_path)
    if not cap.isOpened():
        print("Cannot open file")
        exit()
    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, _ = cap.read()
        if ret:
            gaze = get_gaze_from_frame_idx(frame_idx, fps, time_array, ref_time_array, gaze_points)
            if gaze['gaze detected on surface']:
                x_norm = gaze['gaze position on surface x [normalized]']
                y_norm = gaze['gaze position on surface y [normalized]']
                if magic_ratio is not None:
                    x_norm = (x_norm - 0.5) * magic_ratio + 0.5
                    x_norm = max(0, min(x_norm, 1)) # make sure the x_norm is in the range of [0, 1]
                x = int(x_norm*width)
                y = int(y_norm*height)
            else:
                print('frame_idx: ', frame_idx, 'gaze not on surface')
                x = y = None
            gaze_list.append((x, y))
            frame_idx += 1
        else:
            break
    cap.release()
    # save the gaze_list to the same folder as the ori_video_path
    gaze_list = np.array(gaze_list)
    np.save(gaze_path, gaze_list)


def visual_gaze(visual_video_path, ori_video_path, fps, width, height, time_array, ref_time_array, gaze_points, magic_ratio):
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
                x_norm = gaze['gaze position on surface x [normalized]']
                y_norm = gaze['gaze position on surface y [normalized]']
                if magic_ratio is not None:
                    x_norm = (x_norm - 0.5) * magic_ratio + 0.5
                    x_norm = max(0, min(x_norm, 1)) # make sure the x_norm is in the range of [0, 1]
                x = int(x_norm*width)
                y = int(y_norm*height)
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

    Pilot_Name = 'Pilot_1'
    Trial = 2
    fs = 30 # frame rate of the original video, the frame rate of the scene video is actually 29.97 (difference ignored here)
    path_raw = 'data/' + Pilot_Name + '/Raw/'
    path_map = 'data/' + Pilot_Name + '/Marker_Mapper/'
    path_sequence_file = '../../Experiments/data/Two_Obj/Overlay/' + Pilot_Name + '/Sequence_Trial_' + str(Trial) + '.txt'
    video_sequence = get_video_sequence(path_sequence_file)
    nb_videos = len(video_sequence)

    # Get the section id
    section_info = pd.read_csv(path_raw + 'sections.csv')
    # Sort the section_info by the start time
    section_info = section_info.sort_values(by=['section start time [ns]'])
    # Get the info of the current trial, the section id and the name of the folder that contains the scene video
    trial_info = section_info.iloc[Trial-1]
    section_id = trial_info['section id']
    folder_name = trial_info['recording name'].replace(':', '-') + '-' + trial_info['recording id'].split('-')[0]
    # Get world timestamps
    world_timestamps = pd.read_csv(path_raw + folder_name + '/world_timestamps.csv')
    time_array = world_timestamps['timestamp [ns]'].values
    # Get the mapped gaze points
    gaze_points = pd.read_csv(path_map + 'gaze.csv')
    gaze_points = gaze_points[gaze_points['section id'] == section_id]
    timeline_gaze = gaze_points['timestamp [ns]'].values

    # Detect the QR code in the scene video
    path_scene_video = [path_raw + folder_name + '/' + f for f in os.listdir(path_raw + folder_name) if f.endswith('.mp4')][0]
    path_visual_QR = path_raw + folder_name + '/visual.avi'
    path_index = path_raw + folder_name + '/QR_detected.npy'
    # if file already exists, load it
    if os.path.exists(path_index):
        QR_ifdetected = np.load(path_index)
    else:
        width, height, fps = get_frame_size_and_fps(path_scene_video)
        visual_QR_codes(path_visual_QR, path_scene_video, path_index, fps, width, height)
        QR_ifdetected = np.load(path_index)
    # Find the start and end timings
    if os.path.exists(path_raw + folder_name + '/start_end_idx.npy'):
        start_end_dict = np.load(path_raw + folder_name + '/start_end_idx.npy', allow_pickle=True).item()
    else:
        get_start_end_idx(QR_ifdetected, fs)
    
    for video in video_sequence:
        video_start_idx = start_end_dict[video][1]+1
        start_time_ns = time_array[video_start_idx]
        ref_time_array, _ = create_ref_timeline(time_array, start_time_ns)
        video_dir = 'videos/OVERLAY/pairs/'
        ori_video_path = [video_dir + f for f in os.listdir(video_dir) if video in f][0]
        width, height, fps = get_frame_size_and_fps(ori_video_path)
        gaze_path = '../../Experiments/data/Two_Obj/Overlay/' + Pilot_Name + '/' + video + '_gaze.npy'
        # check if the file exists
        if os.path.exists(gaze_path):
            print('The gaze file already exists!')
        else:
            get_gaze(gaze_path, ori_video_path, fps, time_array, ref_time_array, gaze_points, magic_ratio=1.77/1.12)
        visual_video_path = path_raw + folder_name + '/' + video + '_gaze.avi'
        visual_gaze(visual_video_path, ori_video_path, fps, width, height, time_array, ref_time_array, gaze_points, magic_ratio=1.77/1.12)

    

