
import numpy as np
import glob
import imageio
import random
import cv2
from sklearn.model_selection import train_test_split


def load_batch(images, labels, shape=64, channels=1, h_flip=True, vflip=True, rotation=True):
    x1, y1 = [], []
    for x, y in zip(images, labels):
        # Load images
        img = imageio.imread(x)
        lab = imageio.imread(y)
        # Augmentation
        img, lab = data_augmentation(img, lab, h_flip, vflip, rotation)
        # Reshape
        lab = lab.reshape(shape, shape, 1)
        img = img.reshape(shape, shape, channels)
        # Mask verification
        lab[lab > 0.5] = 1
        lab[lab <= 0.5] = 0
        # Store
        x1.append(img / 255.0)
        y1.append(lab)
    return x1, np.array(y1).astype(np.float32)


def prepare_data(train_img_path, train_mask_path, test_image_path, test_mask_path):

    img = np.array(sorted(glob.glob(r'{}'.format(train_img_path) + r'*.png')))
    test_img = np.array(sorted(glob.glob(r'{}'.format(test_image_path) + r'*.png')))
    label = np.array(sorted(glob.glob(r'{}'.format(train_mask_path) + r'*.png')))
    test_label = np.array(sorted(glob.glob(r'{}'.format(test_mask_path) + r'*.png')))

    return img, label, test_img, test_label


def data_augmentation(image, label, h_flip, vflip, rotation):
    # Data augmentation

    if h_flip and random.randint(0, 1):
        image = np.fliplr(image)
        label = np.fliplr(label)

    if vflip and random.randint(0, 1):
        image = np.flipud(image)
        label = np.flipud(label)

    if rotation and random.randint(0, 1):
        angle = random.randint(0, 3)*90
        if angle != 0:
            M = cv2.getRotationMatrix2D(
                (image.shape[1] // 2, image.shape[0] // 2), angle, 1.0)
            image = cv2.warpAffine(
                image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST)
            label = cv2.warpAffine(
                label, M, (label.shape[1], label.shape[0]), flags=cv2.INTER_NEAREST)

    return image, label
