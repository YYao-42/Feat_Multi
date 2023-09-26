import cv2 as cv
import numpy as np
from pyzbar.pyzbar import decode
import os


# Detect whether there is a QR code in the frame
def detect_QR_code(frame):
    ifdetected = False
    # Convert image to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Detect QR code
    detector = cv.QRCodeDetector()
    try:
        decoded_info, points, _ = detector.detectAndDecode(gray)
    except:
        return frame, ifdetected
    if points is not None:
        ifdetected = True
        for code_info, code_points in zip(decoded_info, points):
            if code_info:
                print("QR Code Data:", code_info)
                # Draw bounding box around the QR code
                for point in code_points:
                    point = point.astype(int)
                    cv.circle(frame, tuple(point), 5, (0, 0, 255), -1)
    return frame, ifdetected


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
            print("QR Code Info:", code_info)
            # Draw bounding box around the QR code
            for p in code_points:
                point = (p.x, p.y)
                cv.circle(frame, tuple(point), 5, (0, 0, 255), -1)
    return frame, ifdetected


# Get the frame size and fps of a video
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


def visual_QR_codes(visual_video_path, test_video_path, fps, width, height):
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
            frame, ifdetected = detect_QR_code(frame)
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
    np.save('QR_detected.npy', QR_detected)
    writer.release()
    cap.release()



# main function
if __name__ == "__main__":
    test_video_path = "videos/test_2.mp4"
    visual_video_path = "videos/visual_2.avi"
    width, height, fps = get_frame_size_and_fps(test_video_path)
    visual_QR_codes(visual_video_path, test_video_path, fps, width, height)
    # detect whether a QR code is in the image
    # frame = cv.imread('images/qrcode_email.png')
    # frame = detect_QR_code(frame)
    # cv.imshow('frame', frame)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
