import sys
import os
import keras
import cv2
import traceback

from src.keras_utils import load_model
from glob import glob
from os.path import splitext, basename
from src.utils import im2single
from src.keras_utils import load_model, detect_lp
from src.label import Shape, writeShapes
from time import sleep
import datetime


def adjust_pts(pts, lroi):
    return pts*lroi.wh().reshape((2, 1)) + lroi.tl().reshape((2, 1))


class LicensePlateDetection():
    def __init__(self, wpod_net_path="data/lp-detector/wpod-net_update1.h5", lp_threshold=.5):
        self.wpod_net = load_model(wpod_net_path)
        self.lp_threshold = lp_threshold

    def predict(self, img, bname):
        output_dir = "output/tmp"
        # bname = splitext(basename(img_path))[0]
        # Ivehicle = cv2.imread(img_path)
        Ivehicle = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
        side = int(ratio*288.)
        bound_dim = min(side + (side % (2**4)), 608)
        # print("\t\tBound dim: {}, ratio: {}".format(bound_dim, ratio))

        Llp, LlpImgs, _ = detect_lp(self.wpod_net, im2single(Ivehicle), bound_dim, 2**4, (240, 80), self.lp_threshold)

        if len(LlpImgs):
            Ilp = LlpImgs[0]
            Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
            Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)

            s = Shape(Llp[0].pts)
            
            # cv2.imwrite('{}/{}_lp.png'.format(output_dir, bname), Ilp * 255.)
            # writeShapes('{}/{}_lp.txt'.format(output_dir, bname), [s])

            Ilp = cv2.normalize(src=(Ilp * 255.), dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            return Ilp, s

        return None, None


if __name__ == '__main__':

    try:

        input_dir = sys.argv[1]
        output_dir = input_dir
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        lp_threshold = .5

        wpod_net_path = sys.argv[2]
        wpod_net = load_model(wpod_net_path)

        imgs_paths = glob('%s/*.png' % input_dir)

        print('Searching for license plates using WPOD-NET')

        for i, img_path in enumerate(imgs_paths):

            print('\t Processing %s' % img_path)
            start = datetime.datetime.now()
            bname = splitext(basename(img_path))[0]
            Ivehicle = cv2.imread(img_path)

            ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
            side = int(ratio*288.)
            bound_dim = min(side + (side % (2**4)), 608)
            print("\t\tBound dim: %d, ratio: %f" % (bound_dim, ratio))

            Llp, LlpImgs, _ = detect_lp(wpod_net, im2single(
                Ivehicle), bound_dim, 2**4, (240, 80), lp_threshold)

            if len(LlpImgs):
                Ilp = LlpImgs[0]
                Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
                Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)

                s = Shape(Llp[0].pts)

                cv2.imwrite('%s/%s_lp.png' % (output_dir, bname), Ilp*255.)
                writeShapes('%s/%s_lp.txt' % (output_dir, bname), [s])
            stop = datetime.datetime.now()
            print(stop-start)

    except:
        traceback.print_exc()
        sys.exit(1)
    sleep(10)
    sys.exit(0)
