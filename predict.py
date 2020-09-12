import argparse
import os
import cv2
import time
import subprocess as sp
import os
import traceback
import sys

from recognition import E2E
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

print("Updated at 11:28 AM")


def get_arguments():
    arg = argparse.ArgumentParser()
    arg.add_argument('-i', '--input_folder', help='link to folder contains images', default='./images/')

    return arg.parse_args()


def get_gpu_memory():
  _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

  ACCEPTABLE_AVAILABLE_MEMORY = 1024
  COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
  memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
  memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
  print(memory_free_values)
  return memory_free_values


def predict_one_image(img, model, name):

    # start
    start = time.time()

    # recognize license plate
    processed_image = model.predict(img, name)

    # end
    end = time.time()

    print('Model process on %.2f s' % (end - start))
    return processed_image


if __name__ == "__main__":
    args = get_arguments()
    img_folder = args.input_folder

    # load model
    model = E2E()

    for img_file in os.listdir(img_folder):
        if img_file.endswith(".jpg") or img_file.endswith(".png"):
            try:
                img_path = os.path.join(img_folder, img_file)

                # read image
                img = cv2.imread(str(img_path))
                processed_img = predict_one_image(img, model, name=img_file)
                # processed_img = img

                # save image
                output_path = os.path.join("output", img_file)
                # print(output_path)
                cv2.imwrite(output_path, processed_img)
            except:
                # printing stack trace
                traceback.print_exception(*sys.exc_info())
                # print("Error image {}".format(img_file))