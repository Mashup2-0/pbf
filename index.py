import numpy as np
from PIL import Image

from align.align_trans import warp_and_crop_face, get_reference_facial_points
from align.detector import detect_faces
from align.visualization_utils import show_results

img = Image.open('testImages/stp.jpeg') # modify the image path to yours
bounding_boxes, landmarks = detect_faces(img) # detect bboxes and landmarks for all faces in the image
image = show_results(img, bounding_boxes, landmarks) # visualize the results
crop_size = 112  # specify size of aligned faces, align and crop with padding
scale = crop_size / 112.
reference = get_reference_facial_points(default_square=True) * scale

for landmark in landmarks:
    facial5points = [[landmark[j], landmark[j + 5]] for j in range(5)]
    warped_face = warp_and_crop_face(np.array(img), facial5points, reference, crop_size=(96, 112))
    img_warped = Image.fromarray(warped_face)

    img_warped.show()



