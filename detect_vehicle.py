import data_utils as utils
import cv2
import numpy as np

from src.label import Label
from src.utils import crop_region


class DetectVehicle(object):
    def __init__(self, threshold=0.5):
        print("------- initial detectNumberPlate")
        try:
            self.weight_path = "./weights/yolov3.weights"
            self.cfg_path = "./cfg/yolov3.cfg"
            self.labels = utils.get_labels("./cfg/coco.names")
            self.threshold = threshold

            print("------- before DetectVehicle.load_model_readNet")
            # Load model
            self.model = cv2.dnn.readNet(
                model=self.weight_path, config=self.cfg_path)
            print("------- after DetectVehicle.load_model_readNet")
        except Exception as ex:
            print("############## Error: {} ##############".format(str(ex)))

    def detect(self, img, bname):
        output_dir = "output/tmp"
        boxes = []
        classes_id = []
        confidences = []
        scale = 0.00392

        blob = cv2.dnn.blobFromImage(img, scalefactor=scale, size=(
            416, 416), mean=(0,0,0), swapRB=True, crop=False)
        height, width = img.shape[:2]

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
                    # coordinate of bounding boxes
                    center_x= int(detection[0]*width)
                    center_y= int(detection[1]*height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)
                
                    #cv2.circle(img,(center_x,center_y),10,(0,255,0),2)
                    #rectangle co-ordinaters
                    x=int(center_x - w/2)
                    y=int(center_y - h/2)
                    #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

                    boxes.append([x, y, w, h])
                    classes_id.append(class_id)
                    confidences.append(confidence)

        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, score_threshold=self.threshold, nms_threshold=0.6)

        R = []
        for i in range(len(boxes)):
            if i in indices:
                x, y, w, h = boxes[i]
                label = str(self.labels[classes_id[i]])
                R.append((label, confidences[i], (x, y, w, h)))

        R = sorted(R, key=lambda x: -x[1])
        R = [r for r in R if r[0] in ['car', 'bus']]

        print('\t\t{} cars found by using detect_vehicle'.format(len(R)))

        cars_img = []
        Lcars = []

        if len(R):
            # Iorig = cv2.imread(img_path)
            Iorig = img
            WH = np.array(Iorig.shape[1::-1], dtype=float)

            for i, r in enumerate(R):
                cx, cy, w, h = (np.array(r[2]) / np.concatenate((WH, WH))).tolist()
                tl = np.array([cx, cy])
                br = np.array([cx + w, cy + h])
                label = Label(0, tl, br)
                Icar = crop_region(Iorig, label)

                Lcars.append(label)

                cv2.imwrite(
                    '{}/{}_{}car.png'.format(output_dir, bname, i), Icar)
                cars_img.append(Icar)

            # lwrite('{}/{}_cars.txt'.format(output_dir, bname), Lcars)

        return cars_img, Lcars
