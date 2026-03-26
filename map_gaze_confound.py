'''
This script is used to map the gaze points to the original video frames.
The synchronization between the video and the gaze data is based on QR code detection. 
- Download the time series CSV and scene video from Pupil Cloud to 'data/Pilot_X/Raw/', and the enrichment data to 'data/Pilot_X/Marker_Mapper/'.
- Set 'Pilot_Name' and 'Trial' to their respective values, then execute the script.
    - The script will first detect the QR code in the scene video and then visualize and save the detection results.
    - Based on these results, users are prompted to accept or adjust the start and end timings of the videos. If the time of the presence of QR codes differs from expectations, users can refer to the visualization video to decide on adjustments. (If the number of start timings is not consistent with the number of videos, the user may need to plot the detection results and mark the start and end timings manually.)
    - The script will then map the gaze points to the original video frames and save the results (to the folder that saves experiment data). The visualization of the gaze points can also be saved as a video.
- If start and end timings are correct, and gaze points are accurately mapped to video frames (by comparing the results to the results on Pupil Cloud), delete the visualization videos (due to their large file size).

Author: yuanyuan.yao@kuleuven.be
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import cv2 as cv
import copy
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
    '''
    Detect and visualize the QR codes in the scene video. Visualization is necessary since the QR code is not always detected due to motion blur or other reasons. Then if the time of the presence of QR codes is not consistent with expectations, users can refer to the visualization video to decide on adjustments.
    '''
    QR_detected = []
    timestamps = []
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
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
            # Save the timestamp of each frame (in milliseconds), since the time lags are not consistent
            timestamps.append(cap.get(cv.CAP_PROP_POS_MSEC))
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
    # save the QR detection results and the timestamps in pkl
    QR_result = {'QR_detected': QR_detected, 'timestamps': timestamps}
    with open(index_path, 'wb') as f:
        pickle.dump(QR_result, f)
    writer.release()
    cap.release()
    return QR_result


def find_edge(QR_results, UP=True):
    '''
    Find the points where the detection result changes from False to True (UP=True) or True to False (UP=False)
    '''
    data_TF = np.array(QR_results['QR_detected'])
    time_ms = np.array(QR_results['timestamps'])
    if UP == False:
        data_TF = data_TF[::-1] # reverse the data, then it becomes the case of UP=True
    data_T_idx = np.where(data_TF)[0]
    data_cleaned_idx = [data_T_idx[0]] 
    for i in data_T_idx:
        if (time_ms[i] - time_ms[data_cleaned_idx[-1]])/1000 > 235: # as we know that the shortest video is 4 minutes long
            data_cleaned_idx.append(i)
    if UP == False:
        data_cleaned_idx = len(data_TF) - np.array(data_cleaned_idx) - 1
    return sorted(data_cleaned_idx)


def show_surrounding_frames(video_path, frame_idx, title, nb_surrounding_frames=20):
    vid = cv.VideoCapture(video_path)
    # a square of frames with the given frame in the middle, and the number of surrounding frames on each side is nb_surrounding_frames//2
    frame_indices = list(range(max(0, frame_idx - nb_surrounding_frames//2), frame_idx)) + \
                    list(range(frame_idx, min(int(vid.get(cv.CAP_PROP_FRAME_COUNT)), frame_idx + nb_surrounding_frames//2)))
    nb_frames = len(frame_indices)
    nb_cols = int(np.ceil(nb_frames/5))
    nb_rows = int(np.ceil(nb_frames/nb_cols))
    figure, ax = plt.subplots(nb_rows, nb_cols, sharey=True)
    for i, idx in enumerate(frame_indices):
        vid.set(cv.CAP_PROP_POS_FRAMES, idx)
        ret, frame = vid.read()
        if ret:
            row_idx = i // nb_cols
            col_idx = i % nb_cols
            ax[row_idx, col_idx].imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            if idx == frame_idx:
                ax[row_idx, col_idx].set_title(f'Frame {idx} (current frame)', color='red')
            else:
                ax[row_idx, col_idx].set_title(f'Frame {idx}')
            ax[row_idx, col_idx].axis('off')
    figure.tight_layout()
    figure.suptitle(title)
    plt.show()
    return
    

def infer_start_from_video_len(start_idx, end_idx, video_sequence, fs_ori):
    '''
    Infer the start timings based on the video length. Each video segment contains two videos, so there should be nb_videos/2 timings inferred from the QR code detection results, and the other nb_videos/2 timings inferred based on the length of each video.
    '''
    nb_videos = len(video_sequence)
    start_inferred = [start_idx[0]]
    end_inferred = [end_idx[0]]
    start_current = start_idx[0]
    end_current = end_idx[0]
    for i in range(nb_videos//2-1):
        video_1 = video_sequence[2*i]
        video_2 = video_sequence[2*i+1]
        start_next = start_current + (get_video_len(video_1) + get_video_len(video_2) + 20) * fs_ori
        end_next = end_current + (get_video_len(video_1) + get_video_len(video_2) + 20) * fs_ori
        if any(np.abs(start_next - np.array(start_idx)) < 5*fs_ori):
            # find the closest element in start_idx to start_next, and replace start_next with that element
            start_next = start_idx[np.argmin(np.abs(start_next - np.array(start_idx)))]
        if any(np.abs(end_next - np.array(end_idx)) < 5*fs_ori):
            # find the closest element in end_idx to end_next, and replace end_next with that element
            end_next = end_idx[np.argmin(np.abs(end_next - np.array(end_idx)))]
        start_inferred.append(start_next)
        end_inferred.append(end_next)
        start_current = start_next
        end_current = end_next
    return start_inferred, end_inferred


def refine_start_end(QR_results, video_sequence, fs_ori):
    start_idx = find_edge(QR_results, UP=True)
    end_idx = find_edge(QR_results, UP=False)
    # remove elments in both start_idx and end_idx
    start_idx_update = [x for x in start_idx if x not in end_idx]
    end_idx = [x for x in end_idx if x not in start_idx]
    start_idx = start_idx_update
    if len(start_idx) == len(end_idx) == nb_videos//2:
        print('The number of start and end timings is consistent with the number of videos!')
    else:
        print('The number of start and end timings is not consistent with the number of videos!')
        print('Infer the start and end timings based on the video length.')
        start_idx, end_idx = infer_start_from_video_len(start_idx, end_idx, video_sequence, fs_ori)
    return start_idx, end_idx


def get_starts(QR_results, video_sequence, path_visual_QR, fs_ori):
    '''
    Get the start timings of videos.
    Use edge detection to find the onset and offset of the QR code presence. Then make adjustments based on whether the time of the presence of QR codes is consistent with expectations and user input.
    After the QR code disappears, there are 3 seconds of showing fixation cross before playing the video. So the video start time is the QR code end time + 3 seconds + 1 frame.
    Each video segment contains two videos, so there should be nb_videos/2 start timings inferred from the QR code detection results, and the other nb_videos/2 timings inferred based on the length of each video.
    '''
    time_ms = np.array(QR_results['timestamps'])
    start_idx, end_idx = refine_start_end(QR_results, video_sequence, fs_ori)
    video_starts = {}
    # check whether the distance between the start and end matches the number of the prepended frames
    for i in range(nb_videos//2):
        video_1 = video_sequence[2*i]
        video_2 = video_sequence[2*i+1]
        print("Synchronizing videos: ", video_1, " and ", video_2)
        print('QR code time of presence (in theory): ', 3) # 3 seconds of QR code
        print('QR code time of presence (based on detection): ', (time_ms[end_idx[i]+1]-time_ms[start_idx[i]])/1000)
        # Ask the user whether to adjust the start and end timings
        print('The start and end indices are: ', start_idx[i], end_idx[i])
        print('Do you want to adjust the start and end timings?')
        print('If yes, please input the new start and end timings (format: start_idx,end_idx)')
        print('If no, please input "no"')
        show_surrounding_frames(path_visual_QR, start_idx[i], 'Starting Frames')
        show_surrounding_frames(path_visual_QR, end_idx[i], 'Ending Frames')
        user_input = input()
        if user_input != 'no':
            start_idx[i], end_idx[i] = tuple(map(int, user_input.split(',')))
            print('QR code time of presence (based on detection, adjusted): ', (time_ms[end_idx[i]+1]-time_ms[start_idx[i]])/1000)
        else:
            pass
        start_v1 = end_idx[i] + 3 * fs_ori + 1
        start_v2 = end_idx[i] + (3 + get_video_len(video_1)) * fs_ori + 1
        video_starts[video_1] = start_v1
        video_starts[video_2] = start_v2
    # Save the start timings with key as the video name
    np.save(path_raw + folder_name + '/video_starts.npy', video_starts)
    return video_starts


def get_video_sequence(path_sequence_file):
    with open(path_sequence_file, 'r') as file:
        # read a list of lines into data, do not include \n
        data = file.read().split('\n')
        # remove the empty string, if exists
        data = list(filter(None, data))
    return data


def get_video_len(video_name):
    if 'speed0.75' in video_name:
        video_len = 160
    else:
        video_len = 120
    return video_len


def get_gaze_of_each_frame(world_time_ns, gaze_df, fixations_df=None, blink_df=None, fs=None):
    gaze_time = gaze_df['timestamp [ns]']
    # find the id in gaze_df where the time is closest to the time_ns
    gaze_idx = np.argmin(np.abs(gaze_time - world_time_ns))
    gaze = gaze_df.iloc[gaze_idx]
    if fixations_df is not None:
        fixation_endtime = fixations_df['end timestamp [ns]']
        # find the id in fixations_df where the time is closest to the time_ns
        fixation_end_idx = np.argmin(np.abs(fixation_endtime - world_time_ns))
        fixation_end = fixations_df.iloc[fixation_end_idx]
        saccade =  np.abs(gaze['timestamp [ns]'] - fixation_end['end timestamp [ns]']) < 1e9/fs/2
    else:
        saccade = None
    if blink_df is not None:
        blink_starttime = blink_df['start timestamp [ns]']
        blink_endtime = blink_df['end timestamp [ns]']
        blink = np.any(np.logical_and(world_time_ns > blink_starttime, world_time_ns < blink_endtime))
    else:
        blink = None
    return gaze, saccade, blink


def get_gaze(gaze_path, video_name, fps_ori, width_ori, height_ori, start_idx_scene_video, time_array, gaze_points, fixations, blinks, magic_ratio):
    world_time_ns = time_array[start_idx_scene_video]
    frame_idx = 0
    gaze_list = []
    video_len = get_video_len(video_name)
    nb_frames = int(video_len * fps_ori)
    while frame_idx < nb_frames:
        world_time_ns = time_array[start_idx_scene_video] + frame_idx * 1e9 / fps_ori
        gaze, saccade, blink = get_gaze_of_each_frame(world_time_ns, gaze_points, fixations, blinks, fs_ori)
        if gaze['gaze detected on surface']:
            x_norm = gaze['gaze position on surface x [normalized]']
            y_norm = gaze['gaze position on surface y [normalized]']
            if magic_ratio is not None:
                x_norm = (x_norm - 0.5) * magic_ratio + 0.5
                x_norm = max(0, min(x_norm, 1)) # make sure the x_norm is in the range of [0, 1]
            x = int(x_norm*width_ori)
            y = int(y_norm*height_ori)
        else:
            print('frame_idx: ', frame_idx, 'gaze not on surface')
            x = y = None
        gaze_list.append((x, y, saccade, blink))
        frame_idx += 1
    gaze_list = np.array(gaze_list)
    np.save(gaze_path, gaze_list)


def visual_gaze(visual_video_path, ori_video_path, fps_ori, width_ori, height_ori, video_sequence):
    video_all = []
    nb_frames_all = []
    for i, video in enumerate(video_sequence):
        if i % 2 == 0:
            video_all.append('instruction.avi')
            nb_frames_all.append(int(20 * fps_ori))
        video_all.append(video)
        nb_frames_all.append(int(get_video_len(video) * fps_ori))
    nb_frames_cumsum = np.cumsum(nb_frames_all)
    frame_idx = 0
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    writer = cv.VideoWriter(visual_video_path, fourcc, fps_ori, (width_ori, height_ori), True)
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
            # check which video the current frame belongs to by finding the first value in nb_frames_cumsum that is larger or equal to frame_idx
            video_idx = np.where(nb_frames_cumsum > frame_idx)[0][0]
            video_name = video_all[video_idx]
            if video_name != 'instruction.avi':
            # load gaze points
                gaze_path = path_raw + video_name[:-4] + '_gaze.npy'
                gaze_array = np.load(gaze_path, allow_pickle=True)
                # get the gaze point for the current frame
                gaze = gaze_array[frame_idx - (nb_frames_cumsum[video_idx-1] if video_idx > 0 else 0), :2]
                x, y = gaze[0], gaze[1]
                if x is not None and y is not None:
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

    Pilot_Name = 'Pilot_8'
    REGENERATE = False
    fs_ori = 30
    path_raw = 'data/Confound/' + Pilot_Name + '/Raw/'
    path_map = 'data/Confound/' + Pilot_Name + '/Marker_Mapper/'
    path_sequence_file = rf'videos\CONFOUND\concatenate\video_order_2026-01-19.txt'
    ori_video_path = rf'videos\CONFOUND\concatenate\concatenate_2026-01-19.avi'
    video_sequence = get_video_sequence(path_sequence_file)
    nb_videos = len(video_sequence)
 
    # Get the section id
    section_info = pd.read_csv(path_raw + 'sections.csv')
    # Get the info of the current trial, the section id and the name of the folder that contains the scene video
    trial_info = section_info.iloc[0]
    folder_name = trial_info['recording name'].replace(':', '-') + '-' + trial_info['recording id'].split('-')[0]
    # Get world timestamps
    world_timestamps = pd.read_csv(path_raw + folder_name + '/world_timestamps.csv')
    time_array = world_timestamps['timestamp [ns]'].values
    # Get the mapped gaze points
    gaze_points = pd.read_csv(path_map + 'gaze.csv')
    # Get fixation points
    fixation_points = pd.read_csv(path_map + 'fixations.csv')
    # Get blink points
    blink_points = pd.read_csv(path_raw + folder_name + '/blinks.csv')

    # Detect the QR code in the scene video
    path_scene_video = [path_raw + folder_name + '/' + f for f in os.listdir(path_raw + folder_name) if f.endswith('.mp4')][0]
    path_visual_QR = path_raw + folder_name + '/visual_QR.mp4'
    path_index = path_raw + folder_name + '/QR_results.pkl'
    # if file already exists, load it
    if os.path.exists(path_index):
        with open(path_index, 'rb') as f:
            QR_result = pickle.load(f)
    else:
        width_scene, height_scene, fps_scene = get_frame_size_and_fps(path_scene_video)
        QR_result = visual_QR_codes(path_visual_QR, path_scene_video, path_index, fps_scene, width_scene, height_scene)
    # Find the start and end timings
    if os.path.exists(path_raw + folder_name + '/video_starts.npy'):
        video_starts = np.load(path_raw + folder_name + '/video_starts.npy', allow_pickle=True).item()
    else:
        video_starts = get_starts(QR_result, video_sequence, path_visual_QR, fs_ori=fs_ori)
    
    for video in video_sequence:
        video_start_idx = video_starts[video]
        width_ori, height_ori, _ = get_frame_size_and_fps(ori_video_path)
        gaze_path = path_raw + video[:-4] + '_gaze.npy'
        # check if the file exists
        if os.path.exists(gaze_path) and not REGENERATE:
            print('The gaze file already exists!')
        else:
            get_gaze(gaze_path, video, fs_ori, width_ori, height_ori, video_start_idx, time_array, gaze_points, fixation_points, blink_points, magic_ratio=1.77/1.12)
    # visual_video_path = path_raw + folder_name + '/gaze.mp4'
    # visual_gaze(visual_video_path, ori_video_path, fs_ori, width_ori, height_ori, video_sequence)

    

