'''
This script is used to process the scene camera video. More specifically, it detects the QR code in the video and then visualizes and saves the detection results.
Two detection methods are implemented: one is based on OpenCV, the other is based on pyzbar.
    - OpenCV: returns more 'True's, but for many of which the bounding box is not accurate and the QR code info is not interpretable. 
    - pyzbar (recommended): returns less 'True's, but for all of which the bounding box is accurate and the QR code info is interpretable. In other words, the results are cleaner. 
        
Author: yuanyuan.yao@kuleuven.be
'''

import cv2 as cv
import numpy as np
from pyzbar.pyzbar import decode
from vputils import get_frame_size_and_fps


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
        if decoded_info != '':
        # WARNING (if exist multiple QR codes): decoded_info is a string, points is a list of numpy arrays.
            for code_info, code_points in zip(decoded_info, points):
                # Draw bounding box around the QR code
                for point in code_points:
                    point = point.astype(int)
                    cv.circle(frame, tuple(point), 5, (0, 0, 255), -1)
                # show the QR code info
                cv.putText(frame, code_info, (int(code_points[0][0]), int(code_points[0][1] - 10)),
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            text = "Unable to decode info!"
            for code_points in points:
                for point in code_points:
                    point = point.astype(int)
                    cv.circle(frame, tuple(point), 5, (0, 0, 255), -1)
                cv.putText(frame, text, (int(code_points[0][0]), int(code_points[0][1] - 10)),
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
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



# main function
if __name__ == "__main__":
    test_video_path = "videos/P1T1.mp4"
    visual_video_path = "videos/visual_P1T1.avi"
    index_path = 'QR_detected_pyzbar.npy'
    width, height, fps = get_frame_size_and_fps(test_video_path)
    visual_QR_codes(visual_video_path, test_video_path, index_path, fps, width, height)