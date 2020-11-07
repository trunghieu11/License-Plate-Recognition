import data_utils as utils
import cv2
import numpy as np


class DetectVehicle(object):
    def __init__(self, threshold=0.5):
        print("------- initial detectNumberPlate")
        try:
            self.weight_path = "./weights/yolov3-tiny.weights"
            self.cfg_path = "./cfg/yolov3-tiny.cfg"
            self.labels = utils.get_labels("./cfg/coco.names")
            self.threshold = threshold

            print("------- before DetectVehicle.load_model_readNet")
            # Load model
            self.model = cv2.dnn.readNet(model=self.weight_path, config=self.cfg_path)
            print("------- after DetectVehicle.load_model_readNet")
        except Exception as ex:
            print("############## Error: {} ##############".format(str(ex)))

    def detect(self, image):
        boxes = []
        classes_id = []
        confidences = []
        scale = 0.00392

        blob = cv2.dnn.blobFromImage(image, scalefactor=scale, size=(416, 416), mean=(0, 0), swapRB=True, crop=False)
        height, width = image.shape[:2]

        # take image to model
        self.model.setInput(blob)

        # run forward
        outputs = self.model.forward(utils.get_output_layers(self.model))

        for out in outputs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = float(scores[class_id])

                if confidence > self.threshold:
                    print("class_id {} scores {}".format(class_id, scores[class_id]))
                    # coordinate of bounding boxes
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)

                    detected_width = int(detection[2] * width)
                    detected_height = int(detection[3] * height)

                    x_min = center_x - detected_width / 2
                    y_min = center_y - detected_height / 2

                    boxes.append([x_min, y_min, detected_width, detected_height])
                    classes_id.append(class_id)
                    confidences.append(confidence)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=self.threshold, nms_threshold=0.4)

        coordinates = []
        for i in indices:
            index = i[0]
            x_min, y_min, width, height = boxes[index]
            x_min = round(x_min)
            y_min = round(y_min)

            coordinates.append((x_min, y_min, width, height))

        print("=========== Result from detect_vehicle ===========")
        print("size of indices: ", len(indices))
        indices_set = set(indices)
        for i in range(len(boxes)):
            if i in indices_set:
                print(classes_id[i], confidences[i], coordinates[i])

        return coordinates
