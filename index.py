from PIL import Image
from align.detector import detect_faces
from align.visualization_utils import show_results

img = Image.open('testImages/stp.jpeg') # modify the image path to yours
bounding_boxes, landmarks = detect_faces(img) # detect bboxes and landmarks for all faces in the image
image = show_results(img, bounding_boxes, landmarks) # visualize the results

image.show()