import datetime
import sys
import traceback
import re

from glob import glob
from os.path import splitext, basename
from time import sleep

import darknet.python.darknet as dn
from darknet.python.darknet import detect, detect_image
from src.label import dknet_label_conversion
from src.utils import nms

class OCR:
    LP_PATTERN = re.compile(r"\d{2}[A-Z]\d{4,5}$")

    def __init__(self, ocr_weights='data/ocr/ocr-net.weights',
                 ocr_netcfg='data/ocr/ocr-net.cfg',
                 ocr_dataset='data/ocr/ocr-net.data',
                 ocr_threshold=.4):

        self.ocr_net = dn.load_net(ocr_netcfg.encode('utf-8'), ocr_weights.encode('utf-8'), 0)
        self.ocr_meta = dn.load_meta(ocr_dataset.encode('utf-8'))
        self.ocr_threshold = ocr_threshold

    def predict(self, img):
        # R, (width, height) = detect(self.ocr_net, self.ocr_meta, img_path.encode('utf-8'), thresh=self.ocr_threshold, nms=None)
        R, (width, height) = detect_image(self.ocr_net, self.ocr_meta, img, thresh=self.ocr_threshold, nms=None)

        if len(R):
            L = dknet_label_conversion(R, width, height)
            L = nms(L, .45)
            L.sort(key=lambda x: (x.line(), x.tl()[0]))

            # print("=========== L ===========")
            # for x in L:
            #     print(x.cl(), x.br(), x.tl(), x.line(), x.prob())
            # print("=========================")

            lp_str = ''.join([chr(l.cl()) for l in L])

            # with open('%s/%s_str.txt' % (output_dir, bname), 'w') as f:
            # 	f.write(lp_str + '\n')

            if OCR.LP_PATTERN.match(lp_str):
                return lp_str
            else:
                return None
        else:
            print('No characters found')
            return None


ocr = OCR()

if __name__ == '__main__':
    try:

        # input_dir  = sys.argv[1]
        # output_dir = input_dir

        input_dir = "output/lp/"
        output_dir = "output/lp_result/"

        ocr_threshold = .4

        ocr_weights = 'data/ocr/ocr-net.weights'
        ocr_netcfg = 'data/ocr/ocr-net.cfg'
        ocr_dataset = 'data/ocr/ocr-net.data'

        ocr_net = dn.load_net(ocr_netcfg.encode('utf-8'), ocr_weights.encode('utf-8'), 0)
        ocr_meta = dn.load_meta(ocr_dataset.encode('utf-8'))

        imgs_paths = sorted(glob('{}/*.jpg'.format(input_dir)))

        print('Performing OCR...')
        print(imgs_paths)
        print(output_dir)

        for i, img_path in enumerate(imgs_paths):

            print('\tScanning {}'.format(img_path))
            start = datetime.datetime.now()
            bname = basename(splitext(img_path)[0])

            R, (width, height) = detect(ocr_net, ocr_meta, img_path.encode('utf-8'), thresh=ocr_threshold, nms=None)

            if len(R):

                L = dknet_label_conversion(R, width, height)
                L = nms(L, .45)

                L.sort(key=lambda x: x.tl()[0])
                lp_str = ''.join([chr(l.cl()) for l in L])

                # with open('%s/%s_str.txt' % (output_dir, bname),'w') as f:
                # 	f.write(lp_str + '\n')

                print('\t\tLP: {}'.format(lp_str))

            else:

                print('No characters found')
            stop = datetime.datetime.now()
            print(stop - start)

    except Exception as ex:
        traceback.print_exc()
        print(ex)
        sys.exit(1)
    sleep(10)
    sys.exit(0)
