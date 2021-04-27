import cv2
import numpy as np
import math

class VehicleCounting(object):
    END_POINT = 150
    CLASSES = open('coco.names').read().strip().split('\n')
            
    # Define vehicle class
    VEHICLE_CLASSES = [1, 2, 3, 5, 6, 7]

    # get it at https://pjreddie.com/darknet/yolo/
    YOLOV3_CFG = 'yolov3-tiny.cfg'
    YOLOV3_WEIGHT = 'yolov3-tiny.weights'

    CONFIDENCE_SETTING = 0.5
    YOLOV3_WIDTH = 416
    YOLOV3_HEIGHT = 416

    MAX_DISTANCE = 80

    def __init__(self):
        self.net = cv2.dnn.readNetFromDarknet(self.YOLOV3_CFG, self.YOLOV3_WEIGHT)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.output_layers = self.get_output_layers()
        self.colors = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))
        self.number_vehicle = 0
        self.list_object = []

    def get_output_layers(self):
        """
        Get output layers of darknet
        :param net: Model
        :return: output_layers
        """
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        return output_layers
    
    def detect(self, frame):
        """
        Detect object use yolo3 model
        :param frame: image
        :return:
        """
        yolo_w, yolo_h = self.YOLOV3_WIDTH, self.YOLOV3_HEIGHT
        height, width = frame.shape[:2]
        img = cv2.resize(frame, (yolo_w, yolo_h))
        blob = cv2.dnn.blobFromImage(img, 0.00392, (yolo_w, yolo_h), swapRB=True, crop=False)
        self.net.setInput(blob)
        layer_output = self.net.forward(self.output_layers)

        boxes = []
        class_ids = []
        confidences = []

        for out in layer_output:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.CONFIDENCE_SETTING and class_id in self.VEHICLE_CLASSES:
                    # print("Object name: " + classes[class_id] + " - Confidence: {:0.2f}".format(confidence * 100))
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = center_x - w // 2
                    y = center_y - h // 2
                    boxes.append([x, y, w, h])
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
        return boxes, class_ids, confidences
    
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
            print("Can't draw prediction for class_id {}: {}".format(class_id, e))

    def check_location(self, box_y, box_height, height):
        """
        Check center point of object that passing end line or not
        :param box_y: y value of bounding box
        :param box_height: height of bounding box
        :param height: height of image
        :return: Boolean
        """
        center_y = box_y + box_height // 2
        return center_y > height - self.END_POINT
    
    def count_vehicles(self, frame):
        height, width = frame.shape[:2]

        # Tracking old object
        tmp_list_object = self.list_object
        self.list_object = []
        for obj in tmp_list_object:
            tracker = obj['tracker']
            class_id = obj['id']
            confidence = obj['confidence']
            check, box = tracker.update(frame)
            if check:
                box_x, box_y, box_width, box_height = box
                self.draw_prediction(frame, self.CLASSES[class_id], self.colors[class_id], confidence,
                                box_x, box_y, box_width, box_height)
                obj['tracker'] = tracker
                obj['box'] = box
                if self.check_location(box_y, box_height, height):
                    # This object passed the end line
                    self.number_vehicle += 1
                else:
                    self.list_object.append(obj)

        # Detect object and check new object
        boxes, class_ids, confidences = self.detect(frame)
        for idx, box in enumerate(boxes):
            box_x, box_y, box_width, box_height = box
            if not self.check_location(box_y, box_height, height):
                # This object doesn't pass the end line
                box_center_x = box_x + box_width // 2
                box_center_y = box_y + box_height // 2
                check_new_object = True
                for tracker in self.list_object:
                    # Check existed object
                    current_box_x, current_box_y, current_box_width, current_box_height = tracker['box']
                    current_box_center_x = current_box_x + current_box_width // 2
                    current_box_center_y = current_box_y + current_box_height // 2
                    # Calculate distance between 2 object
                    distance = math.sqrt((box_center_x - current_box_center_x) ** 2 +
                                            (box_center_y - current_box_center_y) ** 2)
                    if distance < self.MAX_DISTANCE:
                        # Object is existed
                        check_new_object = False
                        break
                if check_new_object:
                    # Append new object to list
                    new_tracker = cv2.TrackerKCF_create()
                    new_tracker.init(frame, (box_x, box_y, box_width, box_height))
                    new_object = {
                        'id': class_ids[idx],
                        'tracker': new_tracker,
                        'confidence': confidences[idx],
                        'box': box
                    }
                    self.list_object.append(new_object)
                    # Draw new object
                    self.draw_prediction(frame, self.CLASSES[new_object['id']], self.colors[new_object['id']], 
                                    new_object['confidence'], box_x, box_y, box_width, box_height)

        # Put summary text
        cv2.putText(frame, "Number : {:03d}".format(self.number_vehicle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        # Draw end line
        cv2.line(frame, (0, height - self.END_POINT), (width, height - self.END_POINT), (255, 0, 0), 2)
        return frame
