import argparse
import os
import cv2
import time

from collections import defaultdict
from recognition import E2E
from pathlib import Path

print("Updated at 01:37 PM")

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def get_arguments():
    arg = argparse.ArgumentParser()
    arg.add_argument('-i', '--input_video', help='link to input video', default='test_video/test.MOV')
    arg.add_argument('-o', '--output_video', help='link to output video', default='output/output_test.MOV')
    arg.add_argument('-f', '--output_file', help='output file path', default='output/output_file.txt')

    return arg.parse_args()


def predict_one_image(img, model, name):
    # recognize license plate
    return model.predict(img, name)


def select_license_plate(license_list, queue_size=4):
    result = set()
    license_count = defaultdict(int)

    for i, child_list in enumerate(license_list):
        if i - queue_size >= 0:
            for cur_license in license_list[i - queue_size]:
                license_count[cur_license] -= 1

        for cur_license in child_list:
            license_count[cur_license] += 1

        for key, value in license_count.items():
            if value == queue_size:
                result.add(key)
    return list(result)


if __name__ == "__main__":

    # start
    start = time.time()

    # args = get_arguments()
    # input_video = args.input_video
    # output_video = args.output_video
    # output_file = args.output_file

    input_video = "test_video/test.MOV"
    output_video = "output/output_test.avi"
    output_file = "output/output_file.txt"
    # input_video = "test_video/IMG_0119.MOV"
    # output_video = "output/output_IMG_0119.MOV"

    # video_size = (540, 960)
    # video_size = (1080, 1920)
    video_size = (1920, 1080)

    # remove existed output video
    if os.path.exists(output_video):
        os.remove(output_video)
        print("Removed {}".format(output_video))

    # video path
    cap = cv2.VideoCapture(input_video)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video, fourcc, 20.0, video_size)

    # load model
    model = E2E()
    frame_count = 0

    license_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            print("[INFO] End of Video")
            break

        frame_count += 1

        # skip frame for faster processing
        if frame_count % 4 != 0:
            # out.write(frame)
            continue

        # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = cv2.resize(frame, video_size)
        try:
            processed_frame, all_license_plates = predict_one_image(frame, model, name="frame_{}.jpg".format(frame_count))
            license_list.append(all_license_plates)
        except:
            processed_frame = frame

        # cv2.imshow('video', processed_frame)
        out.write(processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # choose license plate
    selected_license_plates = select_license_plate(license_list, queue_size=4)

    # write to result file
    writer = open(output_file, "w")
    for cur_license in selected_license_plates:
        writer.write(cur_license + "\n")
    writer.close()

    # end
    end = time.time()
    print("Total running time: {}s".format(end - start))
