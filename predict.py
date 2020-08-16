import argparse
import os
import cv2
import time

from recognition import E2E
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def get_arguments():
    arg = argparse.ArgumentParser()
    arg.add_argument('-i', '--input_folder', help='link to folder contains images', default='./images/')

    return arg.parse_args()


def predict_one_image(img):

    # start
    start = time.time()

    # load model
    model = E2E()

    # recognize license plate
    processed_image = model.predict(img)

    # end
    end = time.time()

    print('Model process on %.2f s' % (end - start))
    return processed_image


if __name__ == "__main__":
    args = get_arguments()
    img_folder = args.input_folder
    for img_file in os.listdir(img_folder):
        if img_file.endswith(".jpg") or img_file.endswith(".png"):
            try:
                img_path = os.path.join(img_folder, img_file)

                # read image
                img = cv2.imread(str(img_path))
                processed_img = predict_one_image(img)

                # save image
                output_path = os.path.join("output", img_file)
                print(output_path)
                cv2.imwrite(output_path, processed_img)
            except:
                print("Error image {}".format(img_file))