import cv2
import time

import math
import sets

DISTANCE_TO_CAMERA = 10
HUMAN_SECONDS_TO_LIVE = 10
ERROR = 0.25
NEAREST = 3


def filter_inside(found):
    found_filtered = []
    for ri, r in enumerate(found):
        for qi, q in enumerate(found):
            if ri != qi and inside(r, q):
                break
        else:
            found_filtered.append(r)
    return found_filtered


def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh


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
    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.calculator = SpeedCalculator()

    def process_frame(self, frame):
        frame = cv2.GaussianBlur(frame, (3, 3), -1)
        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        found, w = self.hog.detectMultiScale(gray, winStride=(4, 4), padding=(8, 8), scale=1.05)
        found_filtered = filter_inside(found)

        self.calculator.update_with_new_detections(found_filtered)

        people, visible = self.calculator.get_visible()
        # draw_detections(frame, found)
        draw_detections(frame, visible, 3)

        return frame, people


class SpeedCalculator:
    def __init__(self):
        self.found_people = []

    def update_with_new_detections(self, found_filtered):
        new_list = []

        operated_list = self.found_people[:]
        operated_rects = found_filtered[:]

        for rect in found_filtered:
            for human in operated_list[:]:
                if human.is_same_human(rect):
                    operated_list.remove(human)
                    human.set_new_rect(rect)
                    remove_rect_from_list(operated_rects, rect)
                    new_list.append(human)
                    break

        for human in operated_list:
            if not human.not_found():
                new_list.append(human)

        for rect in operated_rects:
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

    def set_new_rect(self, rect):
        self.frames += 1
        current_time = time.time()
        self.found_speed = calc_speed_by_two(self.rect, rect, current_time - self.detect_time)
        self.rect = rect
        self.detect_time = current_time

    def not_found(self):
        self.frames = 0
        self.found_speed = 0.0
        return time.time() - self.detect_time > HUMAN_SECONDS_TO_LIVE

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
