'''
This script is used to create a video for testing the synchronization between EEG, NEON and PC.
Video content:
    - 4 markers in the four corners of the frame (always present) -> for gaze mapping
    - a white box (present in the second half of each minute) -> signal to photodiode
    - a QR code (present in the second half of each minute) that either
        - stays static
        - moves in a constant velocity
        - moves in a random velocity
        - occurs randomly
We then analyze the scene camera video and align the timestamps of the QR code with the timestamps of the photodiode signal.
Markers and the QR code need to be big enough to be robustly detected. 
        
Author: yuanyuan.yao@kuleuven.be
'''

import cv2 as cv
import numpy as np
import os
from vputils import add_QR_code


# Read markers from a directory and resize them to the same size
def read_markers(dir_path, size=100):
    markers = []
    for filename in os.listdir(dir_path):
        if filename.startswith('Marker_'):
            img = cv.imread(os.path.join(dir_path,filename))
            if img is not None:
                img = cv.resize(img, (size, size))
                markers.append(img)
    return markers


# Add markers to the four corners of a frame
def add_markers(frame, markers):
    size = markers[0].shape[0]
    frame[0:size, 0:size, :] = markers[0]
    frame[0:size, -size:, :] = markers[1]
    frame[-size:, 0:size, :] = markers[2]
    frame[-size:, -size:, :] = markers[3]
    return frame


def coordinates_next_frame(x0, y0, vx, vy, xrange, yrange):
    x = x0 + vx
    y = y0 + vy
    x1, x2 = xrange
    y1, y2 = yrange
    x_new = (x - x1) % (x2 - x1) + x1
    y_new = (y - y1) % (y2 - y1) + y1
    return int(x_new), int(y_new)


def random_pos_or_velo(xrange, yrange):
    x1, x2 = xrange
    y1, y2 = yrange
    x_rand = np.random.randint(x1, x2)
    y_rand = np.random.randint(y1, y2)
    return x_rand, y_rand


def generate_test_video(video_path, fps, len_min, marker_size=100):
    fourcc = cv.VideoWriter_fourcc(*"MJPG")
    writer = cv.VideoWriter(video_path, fourcc, fps, (854, 480), True)
    num_frames = int(fps*60*len_min)
    markers = read_markers('images/', marker_size)
    start_x_init = 300
    start_y_init = 150
    for i in range(num_frames):
        minute = i // (fps*60)         # get the minute
        second = i // fps - minute*60  # get the second
        frame = np.zeros((480, 854, 3), dtype=np.uint8)
        if second >= 30:
            xrange = (150, 400)
            yrange = (0, 180)
            vxrange = (-10, 10)
            vyrange = (-5, 5)
            if minute == 0:
                start_x, start_y = coordinates_next_frame(start_x_init, start_y_init, 0, 0, xrange, yrange)
            elif minute == 1:
                start_x, start_y = coordinates_next_frame(start_x_init, start_y_init, 3, 0, xrange, yrange)
            elif minute == 2:
                start_x, start_y = coordinates_next_frame(start_x_init, start_y_init, 3, -2, xrange, yrange)
            elif minute == 3:
                if i % (3*fps) == 0:
                    start_x, start_y = random_pos_or_velo(xrange, yrange)
            elif minute == 4:
                if i % (2*fps) == 0:
                    vx_rand, vy_rand = random_pos_or_velo(vxrange, vyrange)
                start_x, start_y = coordinates_next_frame(start_x_init, start_y_init, vx_rand, vy_rand, xrange, yrange)
            elif minute == 5:
                if second % 2 == 0:
                    vx_rand, vy_rand = random_pos_or_velo(vxrange, vyrange)
                start_x, start_y = coordinates_next_frame(start_x_init, start_y_init, vx_rand, vy_rand, xrange, yrange)
            elif minute == 6:
                start_x, start_y = coordinates_next_frame(start_x_init, start_y_init, 0, 0, xrange, yrange)
            else:
                raise ValueError('video too long')
            start_x_init = start_x
            start_y_init = start_y
            frame = add_QR_code(frame, 'images/qrcode_email.png', start_x, start_y)
            frame[200:300, -100:, :] = 255
        frame = add_markers(frame, markers)
        writer.write(frame)
    writer.release()


if __name__ == '__main__':
    marker_size = 120
    fps = 30
    len_min = 7
    video_test_path = 'videos/video_test_size_' + str(marker_size) + '_fps_' + str(fps) + '_len_' + str(len_min) + '.avi'
    generate_test_video(video_test_path, fps, len_min, marker_size)