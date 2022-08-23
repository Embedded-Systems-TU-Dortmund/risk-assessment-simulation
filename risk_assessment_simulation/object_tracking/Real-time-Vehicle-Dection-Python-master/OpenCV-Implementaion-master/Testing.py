import math
import sys
from collections import OrderedDict
import cv2
import numpy as np
import pygame
from pygame.locals import KEYDOWN, K_q
from shapely.geometry import Polygon
from CentroidTracker import CentroidTracker

tracker = CentroidTracker()


# Draw filled rectangle at coordinates
def drawSquareCell(x, y, dimX, dimY):
    pygame.draw.rect(
        _VARS['surf'], BLACK,
        (x, y, dimX, dimY)
    )

    pygame.display.update()


def drawPerceptionArea(x1, y1, x2, y2):
    points = [(x1, y1), (x1 + 50, y1 - 50), (x2 + 50, y2 + 50), (x2, y2)]
    pygame.draw.polygon(_VARS['surf'], WHITE, points, width=0)
    pygame.display.update()


def drawSquareGrid(origin, gridWH, cells):
    CONTAINER_WIDTH_HEIGHT = gridWH
    cont_x, cont_y = origin
    # DRAW Grid Border:
    # TOP lEFT TO RIGHT
    pygame.draw.line(_VARS['surf'], BLACK,
                     (cont_x, cont_y),
                     (CONTAINER_WIDTH_HEIGHT + cont_x, cont_y), _VARS['lineWidth'])
    # # BOTTOM lEFT TO RIGHT
    pygame.draw.line(
        _VARS['surf'], BLACK,
        (cont_x, CONTAINER_WIDTH_HEIGHT + cont_y),
        (CONTAINER_WIDTH_HEIGHT + cont_x,
         CONTAINER_WIDTH_HEIGHT + cont_y), _VARS['lineWidth'])
    # # LEFT TOP TO BOTTOM
    pygame.draw.line(
        _VARS['surf'], BLACK,
        (cont_x, cont_y),
        (cont_x, cont_y + CONTAINER_WIDTH_HEIGHT), _VARS['lineWidth'])
    # # RIGHT TOP TO BOTTOM
    pygame.draw.line(
        _VARS['surf'], BLACK,
        (CONTAINER_WIDTH_HEIGHT + cont_x, cont_y),
        (CONTAINER_WIDTH_HEIGHT + cont_x,
         CONTAINER_WIDTH_HEIGHT + cont_y), _VARS['lineWidth'])

    # Get cell size, just one since its a square grid.
    cellSize = CONTAINER_WIDTH_HEIGHT / cells

    # VERTICAL DIVISIONS: (0,1,2) for grid(3) for example
    for x in range(cells):
        pygame.draw.line(
            _VARS['surf'], BLACK,
            (cont_x + (cellSize * x), cont_y),
            (cont_x + (cellSize * x), CONTAINER_WIDTH_HEIGHT + cont_y), 2)
        # # HORIZONTAl DIVISIONS
        pygame.draw.line(
            _VARS['surf'], BLACK,
            (cont_x, cont_y + (cellSize * x)),
            (cont_x + CONTAINER_WIDTH_HEIGHT, cont_y + (cellSize * x)), 2)


def checkEvents():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()
            cv2.destroyAllWindows()
            sys.exit()
        elif (event.type == KEYDOWN and event.key == K_q) or (event.type == KEYDOWN and event.key == 27):
            pygame.quit()
            cap.release()
            cv2.destroyAllWindows()
            sys.exit()


def getInfo(x1, y1, x2, y2):
    return x1 * y2 - y1 * x2


def solve(points):
    N = len(points)
    firstx, firsty = points[0]
    prevx, prevy = firstx, firsty
    res = 0

    for i in range(1, N):
        nextx, nexty = points[i]
        res = res + getInfo(prevx, prevy, nextx, nexty)
        prevx = nextx
        prevy = nexty
    res = res + getInfo(prevx, prevy, firstx, firsty)
    return abs(res) / 2.0


tracker = CentroidTracker()
# CONSTANTS:
SCREENSIZE = WIDTH, HEIGHT = 1280, 720
BLACK = (0, 0, 0)
GREY = (160, 160, 160)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

_VARS = {'surf': True, 'gridWH': 1280,
         'gridOrigin': (0, 0), 'gridCells': 100, 'lineWidth': 2}

pygame.init()
_VARS['surf'] = pygame.display.set_mode(SCREENSIZE)

cap = cv2.VideoCapture("intersection_fast.mp4")

# Object detection from Stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
i = 0
old_boxes = OrderedDict()
theta = 0
detections_Old = []

points = [(550, 300), (650, 300), (650, 350), (550, 350)]
polygon = pygame.draw.polygon(_VARS['surf'], RED, points, width=0)
sensor_qualities = [45, 30, 15]

while True:
    new_boxes_id = []
    print(" ")
    ret, frame = cap.read()
    if frame is None:
        break
    height, width, _ = frame.shape
    _VARS = {'surf': True, 'gridWH': width,
             'gridOrigin': (0, 0), 'gridCells': 50, 'lineWidth': 2}
    pygame.init()
    _VARS['surf'] = pygame.display.set_mode(SCREENSIZE)
    checkEvents()
    _VARS['surf'].fill(GREY)
    drawSquareGrid(
        _VARS['gridOrigin'], _VARS['gridWH'], _VARS['gridCells'])

    # Draw Obstacle
    obstacle_points = [(550, 300), (650, 300), (650, 350), (550, 350)]
    obstacle = pygame.draw.polygon(_VARS['surf'], RED, obstacle_points, width=0)
    x1, y1, x2, y2, x3, y3, x4, y4 = 0, 0, 525, 0, 525, 100, 0, 100
    points_pavement1 = [(0, 0), (525, 0), (525, 100), (0, 100)]
    points_pavement2 = [(0, 272.5), (525, 272.5), (525, 457.5), (0, 457.5)]
    points_pavement3 = [(0, 720), (525, 720), (525, 630), (0, 630)]
    points_pavement4 = [(725, 0), (1280, 0), (1280, 100), (725, 100)]
    points_pavement5 = [(775, 272.5), (1280, 272.5), (1280, 457.5), (775, 457.5)]
    points_pavement6 = [(725, 720), (1280, 720), (1280, 630), (725, 630)]
    pavement1 = pygame.draw.polygon(_VARS['surf'], BLACK, points_pavement1, width=0)
    pavement2 = pygame.draw.polygon(_VARS['surf'], BLACK, points_pavement2, width=0)
    pavement3 = pygame.draw.polygon(_VARS['surf'], BLACK, points_pavement3, width=0)
    pavement4 = pygame.draw.polygon(_VARS['surf'], BLACK, points_pavement4, width=0)
    pavement5 = pygame.draw.polygon(_VARS['surf'], BLACK, points_pavement5, width=0)
    pavement6 = pygame.draw.polygon(_VARS['surf'], BLACK, points_pavement6, width=0)

    # Extract Region of interest
    roi = frame[0:, 0:]

    # 1. Object Detection
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    Boxes = []
    Angles1 = []
    Angles2 = []
    for cnt in contours:
        # Calculate area and remove small elements
        area = cv2.contourArea(cnt)
        if 300 < area < 2000:
            # cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append([x, y, x + w, y + h])
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            ang = cv2.minAreaRect(box)
            cv2.drawContours(roi, [box], 0, (0, 0, 255), 2)
            x1, y1, x2, y2 = box[0][0], box[0][1], box[1][0], box[3][1]
            Boxes.append(box)

    # 2. Object Tracking
    boxes_ids = tracker.update(detections)
    # if len(boxes_ids) != 0:
    #     print(list(boxes_ids.items())[0])
    i = 0
    for (objectID, centroid) in boxes_ids.items():
        if objectID % 2 == 1:
            new_boxes_id.append([objectID, centroid, 15])
        else:
            new_boxes_id.append([objectID, centroid, 45])
        if i < len(Boxes):
            pygame.draw.polygon(_VARS['surf'], WHITE, Boxes[i], width=0)
            # cv2.putText(roi, "Theta {}".format(int(Angles1[i])), (centroid[0] - 30, centroid[1] - 30),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            pygame.display.update()
            i += 1
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(roi, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(roi, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        else:
            break
    i = 0
    # Perception Area Drawing
    for (ID_Old, C_Old, sensor_quality_old) in old_boxes:
        for (ID_New, C_New, sensor_quality_new) in new_boxes_id:
            if ID_Old == ID_New and 5 <= np.linalg.norm(C_New - C_Old) <= 200:
                # cv2.line(roi, C_Old, C_New, RED, 5)
                angle = int(math.atan2(C_New[1] - C_Old[1], C_New[0] - C_Old[0]) * 180 / math.pi)
                angle_O_N = "Theta {}".format(angle)
                cv2.putText(roi, angle_O_N, (C_New[0] - 30, C_New[1] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                if angle < 0:
                    points = [(C_New[0], C_New[1]),
                              (C_New[0] + 100 * math.cos(math.radians(sensor_quality_new + abs(angle))),
                               C_New[1] - 100 * math.sin(math.radians(sensor_quality_new + abs(angle)))),
                              (C_New[0] + 100 * math.cos(math.radians(sensor_quality_new - abs(angle))),
                               C_New[1] + 100 * math.sin(math.radians(sensor_quality_new - abs(angle))))]
                    polygon = pygame.draw.polygon(_VARS['surf'], GREEN, points, width=0)
                elif angle > 0:
                    points = [(C_New[0], C_New[1]),
                              (C_New[0] + 100 * math.cos(math.radians(sensor_quality_new - abs(angle))),
                               C_New[1] - 100 * math.sin(math.radians(sensor_quality_new - abs(angle)))),
                              (C_New[0] + 100 * math.cos(math.radians(sensor_quality_new + abs(angle))),
                               C_New[1] + 100 * math.sin(math.radians(sensor_quality_new + abs(angle))))]
                    polygon = pygame.draw.polygon(_VARS['surf'], GREEN, points, width=0)
                pygame.display.update()
                if polygon is not None and pygame.Rect.colliderect(polygon, obstacle):
                    polygon1 = Polygon(points)
                    polygon2 = Polygon(obstacle_points)
                    intersection = polygon1.intersection(polygon2)
                    print("Confidence:", math.ceil(intersection.area), "Car ID:", ID_New, "With Obstacle",
                          "Sensor Quality:", sensor_quality_new, "degrees")
                    polygon = None
                elif polygon is not None and pygame.Rect.colliderect(polygon, pavement1):
                    polygon1 = Polygon(points)
                    polygon2 = Polygon(points_pavement1)
                    intersection = polygon1.intersection(polygon2)
                    print("Confidence:", math.ceil(intersection.area), "Car ID:", ID_New, "With Pavement: 1",
                          "Sensor Quality:", sensor_quality_new, "degrees")
                    polygon = None
                elif polygon is not None and pygame.Rect.colliderect(polygon, pavement2):
                    polygon1 = Polygon(points)
                    polygon2 = Polygon(points_pavement2)
                    intersection = polygon1.intersection(polygon2)
                    print("Confidence:", math.ceil(intersection.area), "Car ID:", ID_New, "With Pavement: 2",
                          "Sensor Quality:", sensor_quality_new, "degrees")
                    polygon = None
                elif polygon is not None and pygame.Rect.colliderect(polygon, pavement3):
                    polygon1 = Polygon(points)
                    polygon2 = Polygon(points_pavement3)
                    intersection = polygon1.intersection(polygon2)
                    print("Confidence:", math.ceil(intersection.area), "Car ID:", ID_New, "With Pavement: 3",
                          "Sensor Quality:", sensor_quality_new, "degrees")
                    polygon = None
                elif polygon is not None and pygame.Rect.colliderect(polygon, pavement4):
                    polygon1 = Polygon(points)
                    polygon2 = Polygon(points_pavement4)
                    intersection = polygon1.intersection(polygon2)
                    print("Confidence:", math.ceil(intersection.area), "Car ID:", ID_New, "With Pavement: 4",
                          "Sensor Quality:", sensor_quality_new, "degrees")
                    polygon = None
                elif polygon is not None and pygame.Rect.colliderect(polygon, pavement5):
                    polygon1 = Polygon(points)
                    polygon2 = Polygon(points_pavement5)
                    intersection = polygon1.intersection(polygon2)
                    print("Confidence:", math.ceil(intersection.area), "Car ID:", ID_New, "With Pavement: 5",
                          "Sensor Quality:", sensor_quality_new, "degrees")
                    polygon = None
                elif polygon is not None and pygame.Rect.colliderect(polygon, pavement6):
                    polygon1 = Polygon(points)
                    polygon2 = Polygon(points_pavement6)
                    intersection = polygon1.intersection(polygon2)
                    print("Confidence:", math.ceil(intersection.area), "Car ID:", ID_New, "With Pavement: 6",
                          "Sensor Quality:", sensor_quality_new, "degrees")
                    polygon = None
    old_boxes = new_boxes_id
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(0)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()
sys.exit()
