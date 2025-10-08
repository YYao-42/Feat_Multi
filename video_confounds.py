'''
This script is used to create experimental videos for investigating possible confounds in visual attention decoding.
Baseline: The objects in the videos are cropped based on the bounding boxes. The spatial contrast is controlled to a target std of 40 (0-255 scale).

Effects:
    - Size: scale the cropped objects to 40% or 70% of their original width and height
    - Spatial contrast: adjust the contrast of the cropped objects to meet a target std (10, 3)
    - Speed: change the playback speed of the video (0.75x, 1.5x)
    - Camera viewpoint: play the video at different viewpoints
        
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
import vputils
import pandas as pd
from feutils import extract_box_info_folder
from datetime import datetime

def boxes_update(box_paths, max_len=150):
    box_info_changed = []
    w_max_global = 0
    h_max_global = 0
    for path in box_paths:
        box_info = pickle.load(open(path, 'rb'))
        max_nb_samples = max_len * box_info['fps']
        box_info['box_info'] = box_info['box_info'][:max_nb_samples, :]
        box_info['box_info'] = vputils.clean_boxes_info(box_info['box_info'], box_info["fps"], smooth=True)
        w_max = np.max(box_info['box_info'][:, 2])
        h_max = np.max(box_info['box_info'][:, 3])
        if w_max > w_max_global:
            w_max_global = w_max
        if h_max > h_max_global:
            h_max_global = h_max
        box_info_changed.append(box_info)
    canvas_height = int(h_max_global)
    canvas_width = int(w_max_global*1.2)
    for path, box_info in zip(box_paths, box_info_changed):
        box_updated = np.zeros((max_nb_samples, 4))
        center_xy = np.zeros((max_nb_samples, 2))
        for i in range(max_nb_samples):
            ori_x_start, ori_y_start, ori_width, ori_height = box_info['box_info'][i, :]
            box_updated[i, :], center_xy[i, :]  = vputils.expand_boxes(canvas_height, canvas_width, ori_height, ori_width, ori_x_start, ori_y_start, box_info['frame_height'], box_info['frame_width'])
        box_info['box_info'] = box_updated
        box_info['center_xy'] = center_xy
        with open(path[:-8] + '_processed.pkl', 'wb') as f:
            pickle.dump(box_info, f)
    target_size = (canvas_width, canvas_height)
    return box_info_changed, target_size

def baseline_video(input_path, output_path, box_info, target_size, target_std=40, max_len=150):
    sigma_global = vputils.compute_global_std(input_path, max_len=max_len)
    contrast_scale_factor = target_std / max(sigma_global, 1e-6) if target_std else 1.0
    cap = cv.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open input video at {input_path}")
        return
    w  = target_size[0]
    h  = target_size[1]
    fps = cap.get(cv.CAP_PROP_FPS)
    total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter(output_path, fourcc, fps, (w, h))
    print(f"Processing (cropping): {input_path}")
    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx > box_info.shape[0]:
            print("\nNo more box info. Stopping early.")
            break
        # crop the frame based on the box info
        x_start, y_start, width, height = box_info[frame_idx-1, :].astype(int)
        cropped_frame = frame_bgr[y_start:y_start+height, x_start:x_start+width, :]

        # Convert to YUV (OpenCV uses 8-bit Y,U,V in 0..255; BT.601 matrix)
        yuv = cv.cvtColor(cropped_frame, cv.COLOR_BGR2YUV)
        y_min, y_max = 0, 255
        Y = yuv[:, :, 0].astype(np.float32)
        mu = float(np.mean(Y))
        Y_adj = mu + contrast_scale_factor * (Y - mu)
        # If your source is studio-range, consider clipping to [16,235] instead of [0,255]
        Y_adj = np.clip(Y_adj, y_min, y_max).astype(np.uint8)
        yuv[:, :, 0] = Y_adj
        out_bgr = cv.cvtColor(yuv, cv.COLOR_YUV2BGR)
        out.write(out_bgr)
        print(f"Frame {frame_idx}/{total}", end="\r")

    print(f"\nDone. Saved to {output_path}")
    cap.release()
    out.release()
    cv.destroyAllWindows()

def instruction_video(output_path, canvas_width, canvas_height, fps=30, relax_duration=14, QR_duration=3, cross_duration=3):
    nb_relax_frames = int(relax_duration * fps)
    nb_instruction_frames = int(QR_duration * fps)
    nb_cross_frames = int(cross_duration * fps)
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter(output_path, fourcc, fps, (canvas_width, canvas_height))
    for frame_idx in range(nb_relax_frames):
        frame = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        frame = vputils.add_progress_bar(frame, progress=frame_idx/nb_relax_frames)
        frame = vputils.addText(frame, "Relax", (canvas_width//2 - 200, canvas_height//2), fontScale=6)
        out.write(frame)
    for frame_idx in range(nb_instruction_frames):
        frame = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        frame = vputils.add_QR_code(frame, 'images/qrcode.png', 100, 100)
        frame = vputils.add_progress_bar(frame, progress=frame_idx/nb_instruction_frames)
        vputils.addText(frame, 'Please sit still', (950, 500), fontScale=2, thickness=3)
        out.write(frame)
    for frame_idx in range(nb_cross_frames):
        frame = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        vputils.addText(frame, 'Always fixating on the cross while attending to the video', (100, 300), fontScale=2, thickness=3)
        frame = vputils.add_fixation_cross(frame, cross_size=60, cross_thickness=5, alpha=0.5)
        vputils.add_arrow(frame, (canvas_width//2 - 50, canvas_height//2 - 190), (canvas_width//2, canvas_height//2 - 45), thickness=3)
        frame = vputils.add_progress_bar(frame, progress=frame_idx/nb_cross_frames)
        out.write(frame)
    out.release()
    cv.destroyAllWindows()

def concatenate_videos(output_path, canvas_width, canvas_height, fps=30):
    # aggregate all the created videos
    video_paths = []
    for folder in [baseline_output, size_output, contrast_output, speed_output]:
        for fn in os.listdir(folder):
            if fn.endswith('.avi') and fn[:2] in video_dict.keys():
                video_paths.append(os.path.join(folder, fn))
    # shuffle the video order
    np.random.shuffle(video_paths)
    # save the video order to a txt file
    date = datetime.today().strftime('%Y-%m-%d')
    order_path = os.path.join(concat_output, f'video_order_{date}.txt')
    with open(order_path, 'w') as f:
        for vp in video_paths:
            f.write(f"{os.path.basename(vp)}\n")
    # insert instruction video every 2 videos
    video_paths_all = []
    instruction_flags = []
    for i, vp in enumerate(video_paths):
        if i % 2 == 0:
            video_paths_all.append(instruction_path)
            instruction_flags.append(1)
        video_paths_all.append(vp)
        instruction_flags.append(0)

    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter(output_path, fourcc, fps, (canvas_width, canvas_height))
    for INS, vp in zip(instruction_flags, video_paths_all):
        cap = cv.VideoCapture(vp)
        if not cap.isOpened():
            print(f"Error: Could not open input video at {vp}")
            continue
        print(f"Processing (concatenate): {vp}")
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if not INS:
                canvas = np.zeros((args['canvasheight'], args['canvaswidth'], 3), dtype=np.uint8)
                # put the frame in the center of the canvas
                h, w = frame_bgr.shape[:2]
                y_start = (args['canvasheight'] - h) // 2
                x_start = (args['canvaswidth'] - w) // 2
                canvas[y_start:y_start+h, x_start:x_start+w, :] = frame_bgr
                canvas[:120, -120:, :] = 255 # white square at the top-right corner
                canvas = vputils.add_fixation_cross(canvas, cross_size=60, cross_thickness=5, alpha=0.5)
            else:
                canvas = frame_bgr
            out.write(canvas)
        cap.release()
    out.release()
    cv.destroyAllWindows()
    print(f"\nDone. Saved to {output_path}")

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-base', '--baseline', action='store_true',
    help='Create baseline videos with centered objects and controlled spatial contrast')
ap.add_argument('-csize', '--changesize', action='store_true',
    help='Create videos with changed object sizes')
ap.add_argument('-cct', '--changecontrast', action='store_true',
    help='Create videos with changed spatial contrast')
ap.add_argument('-csp', '--changespeed', action='store_true',
    help='Create videos with changed playback speed')
ap.add_argument('-concat', '--concatenate', action='store_true',
    help='Concatenate videos under different conditions')
ap.add_argument("-dl", "--detectlabel", type=int, default=0,
	help="class of objects to be detected (default: 0 -> person)")
ap.add_argument("-maxt", "--maxtime", type=int, default=120,
	help="maximum length of the video in seconds")
ap.add_argument("-ch", "--canvasheight", type=int, default=1080,
        help="height of the canvas")
ap.add_argument("-cw", "--canvaswidth", type=int, default=1920,
        help="width of the canvas")
ap.add_argument("-confi", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

input_folder = rf"videos\ORI"
output_folder = rf"videos\CONFOUND"

video_dict = {'04': 'magician with poker', '06': 'mime actor with a briefcase', '07': 'acrobat actor wearing a vest',
              '16': 'mime actor with a hat'}

baseline_output = os.path.join(output_folder, 'baseline')
if not os.path.exists(baseline_output):
    os.makedirs(baseline_output)
size_output = os.path.join(output_folder, 'size')
scale_factors = [0.4, 0.7]
if not os.path.exists(size_output):
    os.makedirs(size_output)
contrast_output = os.path.join(output_folder, 'contrast')
target_stds = [3, 10]
if not os.path.exists(contrast_output):
    os.makedirs(contrast_output)
speed_output = os.path.join(output_folder, 'speed')
speed_factors = [0.75, 1.5]
if not os.path.exists(speed_output):
    os.makedirs(speed_output)

concat_output = os.path.join(output_folder, 'concatenate')
if not os.path.exists(concat_output):
    os.makedirs(concat_output)

# check if instruction video exists, if not, create one
instruction_path = rf"videos\CONFOUND\instruction.avi"
if not os.path.exists(instruction_path):
    instruction_video(instruction_path, args['canvaswidth'], args['canvasheight'])

if args['baseline']:
    video_paths = [os.path.join(input_folder, fn) for fn in os.listdir(input_folder) if fn.endswith('.mp4') and fn[:2] in video_dict.keys()]
    # extract box info
    extract_box_info_folder(input_folder, video_dict, args)
    box_info_folder = os.path.join(output_folder, 'box_info')
    box_paths = [os.path.join(box_info_folder, f"{vn}_box_info_raw.pkl") for vn in video_dict.keys()]
    box_info_changed, target_size = boxes_update(box_paths, max_len=args['maxtime'])
    for video_path, box_info in zip(video_paths, box_info_changed):
        video_id = os.path.basename(video_path)[:2]
        output_path = os.path.join(baseline_output, f"{video_id}_baseline.avi")
        baseline_video(video_path, output_path, box_info['box_info'], target_size, target_std=40, max_len=args['maxtime'])

video_paths = [os.path.join(baseline_output, fn) for fn in os.listdir(baseline_output) if fn.endswith('.avi') and fn[:2] in video_dict.keys()]
for video_path in video_paths:
    video_id = os.path.basename(video_path)[:2]
    if args['changesize']:
        for scale_factor in scale_factors:
            output_path = os.path.join(size_output, f"{video_id}_size{scale_factor}.avi")
            vputils.adjust_video_size_ffmpeg(video_path, output_path, scale_factor=scale_factor)
    if args['changecontrast']:
        for target_std in target_stds:
            output_path = os.path.join(contrast_output, f"{video_id}_std{target_std}.avi")
            vputils.adjust_video_contrast_yuv(video_path, output_path, target_std=target_std)
    if args['changespeed']:
        for speed in speed_factors:
            output_path = os.path.join(speed_output, f"{video_id}_speed{speed}.avi")
            vputils.adjust_video_speed_ffmpeg(video_path, output_path, speed=speed)

if args['concatenate']:
    date = datetime.today().strftime('%Y-%m-%d')
    concatenate_path = os.path.join(concat_output, f'concatenate_{date}.avi')
    concatenate_videos(concatenate_path, args['canvaswidth'], args['canvasheight'])