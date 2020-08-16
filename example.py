from recognition import E2E
import cv2
from pathlib import Path
import argparse
import time
import os


def get_arguments():
    arg = argparse.ArgumentParser()
    arg.add_argument('-i', '--image_path', help='link to image', default='./images/1.jpg')

    return arg.parse_args()


args = get_arguments()
img_path = Path(args.image_path)

# get image name
image_name = os.path.basename(img_path)

# read image
img = cv2.imread(str(img_path))

# start
start = time.time()

# load model
model = E2E()

# recognize license plate
image = model.predict(img)

# end
end = time.time()

print('Model process on %.2f s' % (end - start))

# save image
output_path = os.path.join("output", image_name)
print(output_path)
cv2.imwrite(output_path, image) 

# show image
cv2.imshow('License Plate', image)
if cv2.waitKey(0) & 0xFF == ord('q'):
    exit(0)


cv2.destroyAllWindows()
