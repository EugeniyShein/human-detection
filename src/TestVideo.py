import numpy as np
import cv2
import time

GREEN = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_PLAIN

MHI_DURATION = 0.075
MAX_TIME_DELTA = 0.25
MIN_TIME_DELTA = 0.05
AREA = 0.5
MIN_THICKNESS = 10

cap = cv2.VideoCapture('../IMG_0075.mp4')

fgbg = cv2.BackgroundSubtractorMOG()
width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
size = (height, width)

screen_area = height * width

prev_bg = np.zeros(size, np.uint8)
motion_history = np.zeros(size, np.float32)

fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
frame_count = 0
frame_interval_normal = int(1000.0 / fps)
frame_interval = frame_interval_normal

se = np.ones((5, 5), dtype='uint8')


def filter_inside(found):
    pre_filtered = []
    for q in found:
        rx, ry, rw, rh = q
        area = rw * rh
        delta = float(area) / float(screen_area)
        if delta < AREA and rw > MIN_THICKNESS and rh > MIN_THICKNESS:
            pre_filtered.append(q)

    found_filtered = []

    for ri, r in enumerate(pre_filtered):
        for qi, q in enumerate(pre_filtered):
            if ri != qi and inside(r, q):
                break
        else:
            found_filtered.append(r)
    return found_filtered


def inside(r, q):
    if area_inside(r, q):
        return True

    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh


def area_inside(r, q):  # returns None if rectangles don't intersect
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q

    axmax, axmin = rx + rw / 2, rx - rw / 2
    aymax, aymin = ry + rh / 2, ry - rh / 2

    bxmax, bxmin = qx + qw / 2, qx - qw / 2
    bymax, bymin = qy + qh / 2, qy - qh / 2

    dx = min(axmax, bxmax) - max(axmin, bxmin)
    dy = min(aymax, bymax) - max(aymin, bymin)
    if (dx >= 0) and (dy >= 0):
        area = float(dx) * float(dy)
        r_area = float(rw) * float(rh)
        q_area = float(qw) * float(qh)
        delta = area / min(r_area, q_area)
        return delta >= AREA
    return False


def draw_detections(img, rects, thickness=1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15 * w), int(0.05 * h)
        cv2.rectangle(img, (x + pad_w, y + pad_h), (x + w - pad_w, y + h - pad_h), (0, 255, 0), thickness)


while (1):

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.GaussianBlur(frame, (3, 3), -1)
    display = frame.copy()
    fgmask = fgbg.apply(frame)

    diff = cv2.absdiff(fgmask, prev_bg)

    diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, se)
    diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, se)

    prev_bg = fgmask

    frame_count += 1
    timestamp = float(frame_count) / fps

    cv2.updateMotionHistory(diff, motion_history, timestamp, MHI_DURATION)
    #mgrad_mask, mgrad_orient = cv2.calcMotionGradient(motion_history, MAX_TIME_DELTA, MIN_TIME_DELTA, apertureSize=5)
    mseg_mask, mseg_bounds = cv2.segmentMotion(motion_history, timestamp, MAX_TIME_DELTA)

    mseg_bounds = filter_inside(mseg_bounds)

    draw_detections(display, mseg_bounds, 3)

    cv2.imshow('frame', display)

    k = cv2.waitKey(30) & 0xff

    if k == 'q':
        break

cap.release()
cv2.destroyAllWindows()
