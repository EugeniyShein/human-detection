import numpy as np
import cv2
import time

import math

MHI_DURATION = 0.075
MAX_TIME_DELTA = 0.25
MIN_TIME_DELTA = 0.05
DISTANCE_TO_CAMERA = 10
HUMAN_SECONDS_TO_LIVE = 10
ERROR = 0.25
NEAREST = 3
AREA = 0.5
MIN_THICKNESS = 10


def filter_inside(found, screen_area):
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

    merged = merge_bounds(found_filtered)
    return merged


def merge_bounds(mseg_bounds):
    bounds = []
    operated = 0
    mseg_bounds_copy = mseg_bounds[:]
    while operated < len(mseg_bounds):
        operated += 1
        rect = get_biggest_rect(mseg_bounds_copy)
        remove_rect_from_list(mseg_bounds_copy, rect)
        result_rect = rect.copy()

        for r in mseg_bounds_copy[:]:
            if is_close_rect(rect, r):
                operated += 1
                remove_rect_from_list(mseg_bounds_copy, r)
                result_rect = merge_rect(result_rect, r)

        bounds.append(result_rect)
    return bounds


def get_biggest_rect(mseg_bounds):
    biggest = mseg_bounds[0]
    if len(mseg_bounds) > 1:
        for rect in mseg_bounds:
            x, y, w, h = biggest
            rx, ry, rw, rh = rect
            if w * h < rw * rh:
                biggest = rect
    return biggest


def is_close_rect(rect, smaller_rect):
    rx, ry, rw, rh = rect
    dist = calculate_distance_for_rect(rect, smaller_rect)
    w2 = float(rw) ** 2.0
    h2 = float(rh) ** 2.0
    diagonal = (max(w2, h2) - min(w2, h2)) ** 0.5
    return dist < diagonal * 1.25


def merge_rect(rect_a, rect_b):
    rx, ry, rw, rh = rect_a
    qx, qy, qw, qh = rect_b
    x = int((rx + qx) / 2)
    y = int((ry + qy) / 2)
    w = int(max(rx, qx) - min(rx, qx) + rw / 2.0 + qw / 2.0)
    h = int(max(ry, qy) - min(ry, qy) + rh / 2.0 + qh / 2.0)
    rect = x, y, w, h
    return rect


def inside(r, q):
    if area_inside(r, q, AREA, 1):
        return True

    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh


def area_inside(r, q, target, error):  # returns None if rectangles don't intersect
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q

    axmax, axmin = rx + ((rw / 2) * error), rx - ((rw / 2) * error)
    aymax, aymin = ry + ((rh / 2) * error), ry - ((rh / 2) * error)

    bxmax, bxmin = qx + ((qw / 2) * error), qx - ((qw / 2) * error)
    bymax, bymin = qy + ((qh / 2) * error), qy - ((qh / 2) * error)

    dx = min(axmax, bxmax) - max(axmin, bxmin)
    dy = min(aymax, bymax) - max(aymin, bymin)
    if (dx >= 0) and (dy >= 0):
        area = float(dx) * float(dy)
        r_area = float(rw) * float(rh)
        q_area = float(qw) * float(qh)
        delta = area / min(r_area, q_area)
        return delta >= target
    return False


def is_human_in_movement_area(rect, movement):
    for movement_area in movement:
        if area_inside(rect, movement_area, 1, 2):
            return True
    return False


def get_rect_params(rect):
    x, y, width, height = rect
    return width, height


def get_center_coordinates(rect):
    x, y, width, height = rect
    return x, y


def calculate_distance_for_rect(rect1, rect2):
    x1, y1 = get_center_coordinates(rect1)
    x2, y2 = get_center_coordinates(rect2)
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


def draw_detections(img, rects, thickness=1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15 * w), int(0.05 * h)
        cv2.rectangle(img, (x + pad_w, y + pad_h), (x + w - pad_w, y + h - pad_h), (0, 255, 0), thickness)


def calc_speed_by_two(old_position, new_position, timestemp):
    distance = calculate_distance_for_rect(old_position, new_position) / (DISTANCE_TO_CAMERA * 100)
    return distance / timestemp


def get_rect_area(rect):
    x, y, width, height = rect
    return width * height


def remove_rect_from_list(rect_list, rect):
    for idx, val in enumerate(rect_list[:]):
        if val.all() == rect.all():
            rect_list.pop(idx)
            break


class Processor:
    def __init__(self, width, height, fps):
        self.fps = fps
        self.width = width
        self.height = height
        self.screen_area = height * width
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        self.fgbg = cv2.BackgroundSubtractorMOG()

        self.calculator = SpeedCalculator()
        size = (height, width)
        self.prev_background = np.zeros(size, np.uint8)
        self.motion_history = np.zeros(size, np.float32)

        self.frame_count = 0
        frame_interval_normal = int(1000.0 / fps)
        self.frame_interval = frame_interval_normal

        self.se = np.ones((5, 5), dtype='uint8')

    def process_frame(self, frame):
        self.frame_count += 1
        frame = cv2.GaussianBlur(frame, (3, 3), -1)
        display = frame.copy()

        fgmask = self.fgbg.apply(frame)
        diff = cv2.absdiff(fgmask, self.prev_background)
        self.prev_background = fgmask

        diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, self.se)
        diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, self.se)

        timestamp = float(self.frame_count) / self.fps
        cv2.updateMotionHistory(diff, self.motion_history, timestamp, MHI_DURATION)
        mseg_mask, mseg_bounds = cv2.segmentMotion(self.motion_history, timestamp, MAX_TIME_DELTA)

        mseg_bounds = filter_inside(mseg_bounds, self.screen_area)

        if len(mseg_bounds) > 0:
            self.track_human(frame, mseg_bounds)

        people, visible = self.calculator.get_visible()
        draw_detections(display, visible, 3)
        return display, people

    def track_human(self, frame, mseg_bounds):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, w = self.hog.detectMultiScale(gray, winStride=(4, 4), padding=(8, 8), scale=1.05)
        found_filtered = filter_inside(found, self.screen_area)
        self.calculator.update_with_new_detections(found_filtered, mseg_bounds)


class SpeedCalculator:
    def __init__(self):
        self.found_people = []

    def update_with_new_detections(self, found_filtered, mseg_bounds):
        new_list = []

        operated_rects = []
        for rect in found_filtered:
            if is_human_in_movement_area(rect, mseg_bounds):
                operated_rects.append(rect)
                break

        operated_list = self.found_people[:]

        for rect in operated_rects[:]:
            for human in operated_list[:]:
                if human.is_same_human(rect):
                    human.authorised = True
                    operated_list.remove(human)
                    human.set_new_rect(rect)
                    remove_rect_from_list(operated_rects, rect)
                    new_list.append(human)
                    break

        bounds = mseg_bounds[:]
        for rect in bounds[:]:
            for human in operated_list[:]:
                if human.is_same_human(rect):
                    operated_list.remove(human)
                    human.set_new_rect(rect)
                    remove_rect_from_list(operated_rects, bounds)
                    new_list.append(human)
                    break

        for human in operated_list:
            if not human.not_found():
                new_list.append(human)

        for rect in operated_rects:
            human = Human(rect)
            human.authorised = True
            new_list.append(human)

        for rect in bounds:
            new_list.append(Human(rect))

        self.found_people = new_list

    def get_visible(self):
        rects = []
        people = []
        for human in self.found_people:
            if human.frames > 3:
                people.append(human)
                rects.append(human.rect)
        return people, rects


class Human:
    def __init__(self, rect):
        self.rect = rect
        self.found_speed = 0.0
        self.detect_time = time.time()
        self.frames = 1
        self.authorised = False

    def set_new_rect(self, rect):
        self.frames += 1
        current_time = time.time()
        self.found_speed = calc_speed_by_two(self.rect, rect, current_time - self.detect_time)
        self.detect_time = current_time
        if not self.authorised:
            self.rect = rect
        else:
            x, y, width, height = rect
            qx, qy, qw, qh = self.rect
            self.rect = x, y, max(qw, width), max(height, qh)

        self.frames += 1

    def not_found(self):
        self.found_speed = 0.0
        self.frames = 0
        return not self.authorised and time.time() - self.detect_time > HUMAN_SECONDS_TO_LIVE

    def get_coordinates(self):
        return get_center_coordinates(self.rect)

    def calculate_distance(self, rect):
        return calculate_distance_for_rect(self.rect, rect)

    def is_same_human(self, rect):
        user_area = get_rect_area(self.rect)
        new_area = get_rect_area(rect)
        error = user_area * ERROR

        width, height = get_rect_params(self.rect)
        delta = (width + height) / 2 * NEAREST
        distance = self.calculate_distance(rect)

        return (user_area - error) <= new_area <= (user_area + error) and distance <= delta
