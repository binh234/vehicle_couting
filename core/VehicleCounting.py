import cv2
import numpy as np
import math, os

from .sort import Sort
from .helper import intersect, detect_class
from .Config import END_POINT, CLASSES, VEHICLE_CLASSES, YOLOV3_CFG, YOLOV3_WEIGHT, YOLOV3_WIDTH, YOLOV3_HEIGHT

class VehicleCounting(object):

    def __init__(self):
        self.net = cv2.dnn.readNetFromDarknet(YOLOV3_CFG, YOLOV3_WEIGHT)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.output_layers = self.get_output_layers()
        self.colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))
        self.number_vehicle = 0
        self.list_object = []
        self.memory = {}
        self.tracker = Sort()

    def get_output_layers(self):
        """
        Get output layers of darknet
        """
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        return output_layers
    
    def detect(self, frame, confidence_thresh):
        """
        Detect object use yolo3 model
        :param frame: image
        :return:
        """
        yolo_w, yolo_h = YOLOV3_WIDTH, YOLOV3_HEIGHT
        height, width = frame.shape[:2]
        img = cv2.resize(frame, (yolo_w, yolo_h))
        blob = cv2.dnn.blobFromImage(img, 0.00392, (yolo_w, yolo_h), swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_output = self.net.forward(self.output_layers)

        boxes = []
        classes = []
        confidences = []
        mid_points = []

        for out in layer_output:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > confidence_thresh and class_id in VEHICLE_CLASSES:
                    # print("Object name: " + classes[class_id] + " - Confidence: {:0.2f}".format(confidence * 100))
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = center_x - w // 2
                    y = center_y - h // 2
                    boxes.append([x, y, w, h])
                    classes.append(CLASSES[class_id])
                    confidences.append(float(confidence))
                    mid_points.append((center_x, center_y))
        return boxes, classes, confidences, mid_points
    
    def draw_prediction(self, img, label, color, confidence, x, y, width, height):
        """
        Draw bounding box and put class text and confidence
        :param color: color for object
        :param img: image
        :param label: label of this object
        :param confidence: confidence
        :param x: top, left
        :param y: top, left
        :param width: width of bounding box
        :param height: height of bounding box
        :return: None
        """
        try:
            center_x = x + width // 2
            center_y = y + height // 2

            cv2.rectangle(img, (x, y), (x + width, y + height), color, 1)
            cv2.circle(img, (center_x, center_y), 2, (0, 255, 0), -1)
            cv2.putText(img, label + ": {:0.2f}%".format(confidence * 100), (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        except (Exception, cv2.error) as e:
            print(e)
    
    def count_vehicles(self, frame, confidence_thresh, nms_thresh):
        """
        Detect and track vehicles appear in the frame
        :param frame: image
        """
        height, width = frame.shape[:2]
        line = [(0, height - END_POINT), (width, height - END_POINT)]

        # Detect objects
        boxes, classes, confidences, mid_points = self.detect(frame, confidence_thresh)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_thresh, nms_thresh)

        dets = []
        if len(indices) > 0:
            for i in indices.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                dets.append([x, y, x+w, y+h, confidences[i]])
        
        dets = np.asarray(dets)
        tracks = self.tracker.update(dets)

        boxes = []
        indexIDs = []
        previous = self.memory.copy()
        memory = {}

        for track in tracks:
            boxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track[4]))
            self.memory[indexIDs[-1]] = boxes[-1]
        
        for i, box in enumerate(boxes):
            # extract the bounding box coordinates
            (x, y) = (int(box[0]), int(box[1]))
            (w, h) = (int(box[2] - x), int(box[3] - y))
            cur_center = (int(x + w/2), int(y + h/2))

            color = self.colors[indexIDs[i] % len(self.colors)]
            class_label, confidence = detect_class(cur_center, classes, confidences, mid_points)
            self.draw_prediction(frame, class_label, color, confidence, x, y, w, h)

            if indexIDs[i] in previous:
                previous_box = previous[indexIDs[i]]
                (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                prev_center = (int(x2 + (w2 - x2)/2), int(y2 + (h2 - y2)/2))
                cv2.line(frame, cur_center, prev_center, color, 3)

                if intersect(cur_center, prev_center, line[0], line[1]):
                    self.number_vehicle += 1

        # Put summary text
        cv2.putText(frame, "Number : {:03d}".format(self.number_vehicle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        # Draw end line
        cv2.line(frame, line[0], line[1], (255, 0, 0), 2)
        return frame
