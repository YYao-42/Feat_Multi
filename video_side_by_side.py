import numpy as np
import cv2 as cv
import os


def select_pattern(pattern_list, canvas_height, canvas_width):
    if len(pattern_list) == 0:
        pattern_list = ['v1_l', 'v1_r', 'v1_u', 'v1_d']
    # randomly select a pattern
    pattern = np.random.choice(pattern_list)
    if pattern == 'v1_l':
        v1_pos = (0, int(canvas_height / 4))
        v2_pos = (int(canvas_width / 2), int(canvas_height / 4))
    elif pattern == 'v1_r':
        v1_pos = (int(canvas_width / 2), int(canvas_height / 4))
        v2_pos = (0, int(canvas_height / 4))
    elif pattern == 'v1_u':
        v1_pos = (int(canvas_width / 4), 0)
        v2_pos = (int(canvas_width / 4), int(canvas_height / 2))
    elif pattern == 'v1_d':
        v1_pos = (int(canvas_width / 4), int(canvas_height / 2))
        v2_pos = (int(canvas_width / 4), 0)
    else:
        print('Pattern not found!')
        exit()
    # remove the selected pattern from the list
    pattern_list.remove(pattern)
    return pattern_list, v1_pos, v2_pos, pattern


def resize_frame(frame, canvas_height, canvas_width):
    frame_height, frame_width = frame.shape[:2]
    max_height = int(canvas_height / 2)
    max_width = int(canvas_width / 2)
    canvas_ratio = canvas_height / canvas_width
    frame_ratio = frame_height / frame_width
    # fix the ratio, expand or shrink the frame
    if frame_ratio > canvas_ratio:
        frame = cv.resize(frame, (int(max_height / frame_ratio), max_height))
    else:
        frame = cv.resize(frame, (max_width, int(max_width * frame_ratio)))
    return frame


def generate_video_pairs(dir_parent):
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


dir_parent = '../Feature extraction/videos/ori_video/'
video_pairs = generate_video_pairs(dir_parent)
canvas_height = 1080
canvas_width = 1920
# if a output directory does not exist, create one
dir_output = 'videos/pairs/'
if not os.path.exists(dir_output):
    os.makedirs(dir_output)
for pair in video_pairs:
    print('Currently processing video pair: ', pair)
    generate_pattern = []
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
    assert fps1 == fps2, 'The frame rate of the two videos are not the same!'
    # create an output video writer (avi)
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    output_path = dir_output + video_1_ID + '_' + video_2_ID + '.avi'
    out = cv.VideoWriter(output_path, fourcc, fps1, (canvas_width, canvas_height))
    pattern_list = ['v1_l', 'v1_r', 'v1_u', 'v1_d']
    frame_count = 0
    while True:
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        # create a white box on the top right corner
        canvas[:120, -120:, :] = 255
        # Read frames from the two videos
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        # Break the loop when either of the videos ends
        if not ret1 or not ret2:
            break
        # resize the frame
        frame1 = resize_frame(frame1, canvas_height, canvas_width)
        frame2 = resize_frame(frame2, canvas_height, canvas_width)
        # change a pattern per 30 seconds
        if frame_count % (fps1*30) == 0:
            pattern_list, v1_pos, v2_pos, pattern = select_pattern(pattern_list, canvas_height, canvas_width)
            generate_pattern.append(pattern)
        # place the two videos on the canvas
        canvas[v1_pos[1]:v1_pos[1] + frame1.shape[0], v1_pos[0]:v1_pos[0] + frame1.shape[1], :] = frame1
        canvas[v2_pos[1]:v2_pos[1] + frame2.shape[0], v2_pos[0]:v2_pos[0] + frame2.shape[1], :] = frame2
        frame_count += 1
        out.write(canvas)
    # save the generated pattern
    pattern_path = dir_output + video_1_ID + '_' + video_2_ID + '_pattern.txt'
    with open(pattern_path, 'w') as f:
        for pattern in generate_pattern:
            f.write("%s\n" % pattern)
    cap1.release()
    cap2.release()
    out.release()
