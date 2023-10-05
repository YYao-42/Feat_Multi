import numpy as np
import cv2 as cv


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

