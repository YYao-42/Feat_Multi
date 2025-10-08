import numpy as np
import cv2 as cv
import os
import copy
import scipy
import pandas as pd
import subprocess
import json


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


def add_fixation_cross(frame, cross_size=20, cross_thickness=2, color=(0, 0, 255), alpha=0.5):
    H, W = frame.shape[:2]
    center_x = W // 2
    center_y = H // 2
    # Make a copy to draw the cross on
    overlay = frame.copy()
    # Draw on overlay
    cv.line(overlay, (center_x - cross_size // 2, center_y),
                      (center_x + cross_size // 2, center_y),
                      color, cross_thickness)
    cv.line(overlay, (center_x, center_y - cross_size // 2),
                      (center_x, center_y + cross_size // 2),
                      color, cross_thickness)
    # Blend overlay with original frame
    blended = cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    return blended


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


def add_arrow(frame, start_point, end_point, color=(255,255,255), thickness=5):
    return cv.arrowedLine(frame, start_point, end_point, color, thickness, tipLength=0.3)


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


def compute_global_std(input_path, max_len=None):
    cap = cv.VideoCapture(input_path)
    vals = []
    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_idx += 1
        if max_len and frame_idx > max_len * cap.get(cv.CAP_PROP_FPS):
            break
        yuv = cv.cvtColor(frame_bgr, cv.COLOR_BGR2YUV)
        Y = yuv[:,:,0].astype(np.float32)
        vals.append(np.std(Y))
    cap.release()
    return np.mean(vals)  # average std across frames


def adjust_video_contrast_yuv(input_path, output_path, target_std=None, max_len=None):
    sigma_global = compute_global_std(input_path, max_len=max_len)
    contrast_scale_factor = target_std / max(sigma_global, 1e-6) if target_std else 1.0
    cap = cv.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open input video at {input_path}")
        return
    w  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS)
    total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter(output_path, fourcc, fps, (w, h))
    print(f"Processing (YUV luma): {input_path}")

    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_idx += 1
        if max_len and frame_idx > max_len * fps:
            print("\nReached max frame limit. Stopping early.")
            break
        # Convert to YUV (OpenCV uses 8-bit Y,U,V in 0..255; BT.601 matrix)
        yuv = cv.cvtColor(frame_bgr, cv.COLOR_BGR2YUV)
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


def adjust_video_size_ffmpeg(input_path, output_path, scale_factor):
    # Use ffprobe to get video resolution
    probe_cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        input_path
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    width = info["streams"][0]["width"]
    height = info["streams"][0]["height"]

    # Calculate new size
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # Call ffmpeg to resize
    command = [
        "ffmpeg",
        "-i", input_path,
        "-vf", f"scale={new_width}:{new_height}",
        "-c:a", "copy",  # copy audio
        output_path
    ]
    subprocess.run(command, check=True)
    print(f"Video resized from {width}x{height} to {new_width}x{new_height}: {output_path}")


def adjust_video_speed_ffmpeg(input_path, output_path, speed, mode="dup", out_fps=30.0, crf=18, vcodec="libx264", ffmpeg_overwrite=True):
    """
    Change playback speed of a video file. The output duration changes (no frozen tail),
    and the output stream is generated at `out_fps` by the filter graph.
    """
    if speed <= 0:
        raise ValueError("speed must be > 0")

    vf_parts = [f"setpts=PTS/{speed:.6g}"]  # speed up (2x -> PTS/2), slow down (<1)

    if mode == "dup":
        # Sample frames at out_fps based on the new timestamps (after setpts).
        vf_parts.append(f"fps={out_fps:.6g}")
    elif mode == "interpolate":
        # Motion-compensated interpolation to exactly out_fps.
        vf_parts.append(f"minterpolate=mi_mode=mci:mc_mode=aobmc:vsbmc=1:fps={out_fps:.6g}")
    else:
        raise ValueError("mode must be one of: 'dup', 'interpolate'")

    vf = ",".join(vf_parts)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y" if ffmpeg_overwrite else "-n",
        "-i", input_path,
        "-filter:v", vf,
        # Let the filtergraph dictate timing & length; don't force CFR at the muxer.
        "-vsync", "vfr",          # or use "-vsync", "0"
        # DO NOT set "-r" here; the fps/minterpolate filter already created the cadence.
        "-c:v", vcodec,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        output_path,
    ]

    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        raise RuntimeError(f"ffmpeg failed with code {proc.returncode}")
    else:
        tail = "\n".join(proc.stdout.splitlines()[-10:])
        print(tail or "Done.")

