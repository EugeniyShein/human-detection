import numpy as np
import cv2
import os
import time

from ImageProcessor import Processor
from Logger import Logger

localtime = time.localtime()
timeString = time.strftime("%Y%m%d%H%M%S", localtime)

directory = '../' + timeString + '/'

if not os.path.exists(directory):
    os.makedirs(directory)

logger = Logger(directory)

cap = cv2.VideoCapture("../SecurityCameraMomentsOfAllTime.mp4")
image_processor = Processor()

size = (int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))

fourcc = cv2.cv.CV_FOURCC(*'XVID')
writer = cv2.VideoWriter(directory + "video.avi", fourcc, 10.0, size)


def get_file_name():
    localtime = time.localtime()
    return directory + time.strftime("%Y%m%d%H%M%S", localtime) + '.jpg'


def log(filtered):
    current_time = time.ctime()
    for h in filtered:
        logger.write_log(current_time, h)


while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    out, visible = image_processor.process_frame(frame)

    if len(visible) != 0:
        log(visible)
        cv2.imwrite(get_file_name(), frame)

    cv2.imshow('frame', out)
    writer.write(out)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
writer.release()
cv2.destroyAllWindows()
logger.close()
