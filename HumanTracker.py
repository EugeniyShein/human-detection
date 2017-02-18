import numpy as np
import cv2
import csv
import time
import math
import itertools
import os

localtime = time.localtime()
timeString = time.strftime("%Y%m%d%H%M%S", localtime)

old_found_filtered = []
found_speed = []

cam_distance = 10
width = 640

directory = './' + timeString + '/'

if not os.path.exists(directory):
    os.makedirs(directory)

myfile = open(directory + "data.csv", 'wb')
wr = csv.writer(myfile, delimiter='|', quoting=csv.QUOTE_ALL)


def calculate_speed(old_found_filtered, found_filtered, timestemp):
    if len(old_found_filtered) > 0:
        pairs = sort_nearest(found_filtered)
        found_speed[:] = []
        for pair in pairs:
            found_speed.append(calc_speed_by_two(pair, timestemp))
    else:
        found_speed[:] = [0.0] * len(found_filtered)


def sort_nearest(found_filtered):
    out = []
    new_found_filtered_list = found_filtered[:]
    for old_item in old_found_filtered:
        if len(new_found_filtered_list) > 0:
            out.append((old_item, new_found_filtered_list[0]))
            new_found_filtered_list.remove(new_found_filtered_list[0])
        else:
            break

    return out


def log(found_filtered):
    current_time = time.ctime()
    for r in itertools.product(found_filtered, found_speed):
        wr.writerow([current_time, r[0], r[1]])
        myfile.flush()


def calc_speed_by_two(rectangles, timestemp):
    x1, y1 = get_rectangles_coordinates(rectangles[0])
    x2, y2 = get_rectangles_coordinates(rectangles[1])
    distance = calculateDistance(x1, y1, x2, y2) / (cam_distance * 100)
    return distance / timestemp


def get_rectangles_coordinates(rectangle):
    return rectangle[3] - rectangle[0], rectangle[2] - rectangle[1]


def calculateDistance(x1, y1, x2, y2):
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


def draw_detections(img, rects, thickness=1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15 * w), int(0.05 * h)
        cv2.rectangle(img, (x + pad_w, y + pad_h), (x + w - pad_w, y + h - pad_h), (0, 255, 0), thickness)


def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh


def get_file_name():
    localtime = time.localtime()
    return directory + time.strftime("%Y%m%d%H%M%S", localtime) + '.jpg'


cap = cv2.VideoCapture(0)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
prev_time = time.time()

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    frame = cv2.GaussianBlur(frame, (3, 3), -1)

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    found, w = hog.detectMultiScale(gray, winStride=(8, 8), padding=(32, 32), scale=1.05)
    found_filtered = []
    for ri, r in enumerate(found):
        for qi, q in enumerate(found):
            if ri != qi and inside(r, q):
                break
        else:
            found_filtered.append(r)
    draw_detections(frame, found)
    draw_detections(frame, found_filtered, 3)

    current_time = time.time()

    if len(found_filtered) != 0:
        calculate_speed(old_found_filtered, found_filtered, current_time - prev_time)
        old_found_filtered = found_filtered
        log(found_filtered)
        cv2.imwrite(get_file_name(), frame)

    prev_time = current_time

    cv2.imshow('frame', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
myfile.close()
