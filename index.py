import numpy as np
import torch
from PIL import Image
from models import IR_50

from align.align_trans import warp_and_crop_face, get_reference_facial_points
from align.detector import detect_faces
from align.visualization_utils import show_results

img = Image.open('testImages/stp.jpeg') # modify the image path to yours
bounding_boxes, landmarks = detect_faces(img) # detect bboxes and landmarks for all faces in the image
image = show_results(img, bounding_boxes, landmarks) # visualize the results
crop_size = 112  # specify size of aligned faces, align and crop with padding
scale = crop_size / 112.
reference = get_reference_facial_points(default_square=True) * scale

model = IR_50((112, 112)).eval()
model.load_state_dict(torch.load('backbone_ir50_ms1m_epoch120.pth'))

for landmark in landmarks:
    facial5points = [[landmark[j], landmark[j + 5]] for j in range(5)]
    warped_face = warp_and_crop_face(np.array(img), facial5points, reference, crop_size=(112, 112))
    img_warped = torch.Tensor(warped_face).unsqueeze(0).transpose(1, 3)
    out = model(img_warped).detach().cpu().numpy()
    norm = np.linalg.norm(out, axis=1)
    out = out / norm
    i = 5




