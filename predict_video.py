import argparse
import os
import cv2
import time

from recognition import E2E
from pathlib import Path

print("Updated at 01:13 PM")

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def get_arguments():
    arg = argparse.ArgumentParser()
    arg.add_argument('-i', '--input_video', help='link to input video', default='test_video/test.MOV')
    arg.add_argument('-o', '--output_video', help='link to output video', default='output/output_test.MOV')

    return arg.parse_args()


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
    # args = get_arguments()
    # input_video = args.input_video
    # output_video = args.output_video

    input_video = "test_video/test.MOV"
    output_video = "output/output_test.avi"
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
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video, fourcc, 20.0, video_size)

    # load model
    model = E2E()
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            print("[INFO] End of Video")
            break
        frame_count += 1

        # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = cv2.resize(frame, video_size)
        try:
            processed_frame = predict_one_image(frame, model, name="frame_{}.jpg".format(frame_count))
        except:
            processed_frame = frame

        # cv2.imshow('video', processed_frame)
        out.write(processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
