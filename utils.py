import math

import cv2 as cv
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from align_faces import get_reference_facial_points, warp_and_crop_face
from config import device
from config import image_h, image_w
from mtcnn.detector import detect_faces

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}
transformer = data_transforms['val']

checkpoint = 'BEST_checkpoint.tar'
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
model = model.to(device)
model.eval()

threshold = 73.18799151798612


class FaceNotFoundError(Exception):
    """Base class for other exceptions"""
    pass


def align_face(img_fn, facial5points):
    raw = cv.imread(img_fn, True)
    facial5points = np.reshape(facial5points, (2, 5))

    crop_size = (image_h, image_w)

    default_square = True
    inner_padding_factor = 0.25
    outer_padding = (0, 0)
    output_size = (image_h, image_w)

    # get the reference 5 landmarks position in the crop settings
    reference_5pts = get_reference_facial_points(
        output_size, inner_padding_factor, outer_padding, default_square)

    # dst_img = warp_and_crop_face(raw, facial5points)
    dst_img = warp_and_crop_face(raw, facial5points, reference_pts=reference_5pts, crop_size=crop_size)
    return dst_img


def get_face_all_attributes(full_path):
    try:
        img = Image.open(full_path).convert('RGB')
        bounding_boxes, landmarks = detect_faces(img)

        if len(landmarks) > 0:
            i = select_central_face(img.size, bounding_boxes)
            return True, [bounding_boxes[i]], [landmarks[i]]

    except KeyboardInterrupt:
        raise
    except Exception as ex:
        print(ex)
        pass
    return False, None, None


def select_central_face(im_size, bounding_boxes):
    width, height = im_size
    nearest_index = -1
    nearest_distance = 100000
    for i, b in enumerate(bounding_boxes):
        x_box_center = (b[0] + b[2]) / 2
        y_box_center = (b[0] + b[2]) / 2
        x_img = width / 2
        y_img = height / 2
        distance = math.sqrt((x_box_center - x_img) ** 2 + (y_box_center - y_img) ** 2)
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_index = i

    return nearest_index


def get_image(filename):
    has_face, bboxes, landmarks = get_face_all_attributes(filename)
    if not has_face:
        raise FaceNotFoundError(filename)

    img = align_face(filename, landmarks)
    img = transforms.ToPILImage()(img)
    img = transformer(img)
    img = img.to(device)
    return img


def compare(fn_0, fn_1):
    print('fn_0: ' + fn_0)
    print('fn_1: ' + fn_1)
    img0 = get_image(fn_0)
    img1 = get_image(fn_1)
    imgs = torch.zeros([2, 3, 112, 112], dtype=torch.float)
    imgs[0] = img0
    imgs[1] = img1

    with torch.no_grad():
        output = model(imgs)

        feature0 = output[0].cpu().numpy()
        feature1 = output[1].cpu().numpy()
        x0 = feature0 / np.linalg.norm(feature0)
        x1 = feature1 / np.linalg.norm(feature1)
        cosine = np.dot(x0, x1)
        theta = math.acos(cosine)
        theta = theta * 180 / math.pi

    print('theta: ' + str(theta))
    prob = get_prob(theta)
    print('prob: ' + str(prob))
    return prob, theta < threshold


def get_prob(theta):
    mu_0 = 89.6058
    sigma_0 = 4.5451
    mu_1 = 43.5357
    sigma_1 = 8.83
    from scipy.stats import norm
    prob_0 = norm.pdf(theta, mu_0, sigma_0)
    prob_1 = norm.pdf(theta, mu_1, sigma_1)
    return prob_1/(prob_0 + prob_1)


def ensure_folder(folder):
    import os
    if not os.path.isdir(folder):
        os.mkdir(folder)


if __name__ == "__main__":
    compare('id_card.jpg', 'photo_1.jpg')
    compare('id_card.jpg', 'photo_2.jpg')
    compare('id_card.jpg', 'photo_3.jpg')
    compare('id_card.jpg', 'photo_4.jpg')
